import os
import contextlib
import pytorch_lightning as pl
import math
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict

from infgen.utils.func import angle_between_2d_vectors
from infgen.modules.layers import OccLoss
from infgen.modules.attr_tokenizer import Attr_Tokenizer
from infgen.modules.infgen_decoder import InfGenDecoder
from infgen.datasets.preprocess import TokenProcessor
from infgen.metrics.compute_metrics import *
from infgen.utils.metrics import *
from infgen.utils.visualization import *
from infgen.utils.func import wrap_angle


class InfGen(pl.LightningModule):

    def __init__(self, model_config, save_path: os.PathLike="", logger=None, **kwargs) -> None:
        super(InfGen, self).__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.warmup_steps = model_config.warmup_steps
        self.lr = model_config.lr
        self.total_steps = model_config.total_steps
        self.dataset = model_config.dataset
        self.input_dim = model_config.input_dim
        self.hidden_dim = model_config.hidden_dim
        self.output_dim = model_config.output_dim
        self.output_head = model_config.output_head
        self.num_historical_steps = model_config.num_historical_steps
        self.num_future_steps = model_config.decoder.num_future_steps
        self.num_freq_bands = model_config.num_freq_bands
        self.save_path = save_path
        self.vis_map = False
        self.noise = True
        self.local_logger = logger
        self.max_epochs = kwargs.get('max_epochs', 0)
        module_dir = os.path.dirname(os.path.dirname(__file__))

        self.map_token_traj_path = os.path.join(module_dir, 'tokens/map_traj_token5.pkl')
        self.init_map_token()

        self.predict_motion = model_config.predict_motion
        self.predict_state = model_config.predict_state
        self.predict_map = model_config.predict_map
        self.predict_occ = model_config.predict_occ
        self.pl2seed_radius = model_config.decoder.pl2seed_radius
        self.token_size = model_config.decoder.token_size

        self.disable_grid_token = getattr(model_config, 'disable_grid_token') \
            if hasattr(model_config, 'disable_grid_token') else False
        self.use_grid_token = not self.disable_grid_token
        if self.disable_grid_token:
            self.predict_occ = False

        self.disable_head_token = getattr(model_config, 'disable_head_token') \
            if hasattr(model_config, 'disable_head_token') else False
        self.use_head_token = not self.disable_head_token

        self.disable_state_token = getattr(model_config, 'disable_state_token') \
            if hasattr(model_config, 'disable_state_token') else False
        self.use_state_token = not self.disable_state_token

        self.disable_insertion = getattr(model_config, 'disable_insertiion') \
            if hasattr(model_config, 'disable_insertion') else False

        self.token_processer = TokenProcessor(self.token_size,
                                              training=self.training,
                                              predict_motion=self.predict_motion,
                                              predict_state=self.predict_state,
                                              predict_map=self.predict_map,
                                              state_token=model_config.state_token,
                                              pl2seed_radius=self.pl2seed_radius)

        self.attr_tokenizer = Attr_Tokenizer(grid_range=self.model_config.grid_range,
                                             grid_interval=self.model_config.grid_interval,
                                             radius=model_config.decoder.pl2seed_radius,
                                             angle_interval=self.model_config.angle_interval)

        # state tokens
        self.invalid_state = int(self.model_config.state_token['invalid'])
        self.valid_state = int(self.model_config.state_token['valid'])
        self.enter_state = int(self.model_config.state_token['enter'])
        self.exit_state = int(self.model_config.state_token['exit'])

        self.seed_size = int(model_config.decoder.seed_size)

        self.encoder = InfGenDecoder(
            decoder_type=model_config.decoder_type,
            dataset=model_config.dataset,
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            num_historical_steps=model_config.num_historical_steps,
            num_freq_bands=model_config.num_freq_bands,
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            dropout=model_config.dropout,
            num_map_layers=model_config.decoder.num_map_layers,
            num_agent_layers=model_config.decoder.num_agent_layers,
            pl2pl_radius=model_config.decoder.pl2pl_radius,
            pl2a_radius=model_config.decoder.pl2a_radius,
            pl2seed_radius=model_config.decoder.pl2seed_radius,
            a2a_radius=model_config.decoder.a2a_radius,
            a2sa_radius=model_config.decoder.a2sa_radius,
            pl2sa_radius=model_config.decoder.pl2sa_radius,
            time_span=model_config.decoder.time_span,
            map_token={'traj_src': self.map_token['traj_src']},
            token_size=self.token_size,
            attr_tokenizer=self.attr_tokenizer,
            predict_motion=self.predict_motion,
            predict_state=self.predict_state,
            predict_map=self.predict_map,
            predict_occ=self.predict_occ,
            state_token=model_config.state_token,
            use_grid_token=self.use_grid_token,
            use_head_token=self.use_head_token,
            use_state_token=self.use_state_token,
            disable_insertion=self.disable_insertion,
            seed_size=self.seed_size,
            buffer_size=model_config.decoder.buffer_size,
            num_recurrent_steps_val=model_config.num_recurrent_steps_val,
            loss_weight=model_config.loss_weight,
            logger=logger,
        )
        self.minADE = minADE(max_guesses=1)
        self.minFDE = minFDE(max_guesses=1)
        self.TokenCls = TokenCls(max_guesses=1)
        self.StateCls = TokenCls(max_guesses=1)
        self.StateAccuracy = StateAccuracy(state_token=self.model_config.state_token)
        self.GridOverlapRate = GridOverlapRate(num_step=18,
                                               state_token=self.model_config.state_token,
                                               seed_size=self.encoder.agent_encoder.num_seed_feature)
        # self.NumInsertAccuracy = NumInsertAccuracy()
        self.loss_weight = model_config.loss_weight

        self.token_cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        if self.predict_map:
            self.map_token_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        if self.predict_state:
            self.state_cls_loss = nn.CrossEntropyLoss(
                torch.tensor(self.loss_weight['state_weight']))
            self.state_cls_loss_seed = nn.CrossEntropyLoss(
                torch.tensor(self.loss_weight['seed_state_weight']))
            self.type_cls_loss_seed = nn.CrossEntropyLoss(
                torch.tensor(self.loss_weight['seed_type_weight']))
            self.pos_cls_loss_seed = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.head_cls_loss_seed = nn.CrossEntropyLoss()
            self.offset_reg_loss_seed = nn.MSELoss()
            self.shape_reg_loss_seed = nn.MSELoss()
            self.pos_reg_loss_seed = nn.MSELoss()
            self.head_reg_loss_seed = nn.MSELoss()
        if self.predict_occ:
            self.occ_cls_loss = nn.CrossEntropyLoss()
            self.agent_occ_loss_seed = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.loss_weight['agent_occ_pos_weight']]))
            self.pt_occ_loss_seed = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.loss_weight['pt_occ_pos_weight']]))
            # self.agent_occ_loss_seed = OccLoss()
            # self.pt_occ_loss_seed = OccLoss()
            # self.agent_occ_loss_seed = nn.BCEWithLogitsLoss()
            # self.pt_occ_loss_seed = nn.BCEWithLogitsLoss()
        self.rollout_num = 1

        self.val_open_loop = model_config.val_open_loop
        self.val_close_loop = model_config.val_close_loop
        self.val_insert = model_config.val_insert or bool(os.getenv('VAL_INSERT'))
        self.n_rollout_close_val = model_config.n_rollout_close_val
        self.t = kwargs.get('t', 2)

        # for validation / test
        self._mode = 'training'
        self._long_metrics = None
        self._online_metric = False
        self._save_validate_reuslts = False
        self._plot_rollouts = False

    def set(self, mode: str = 'train'):
        self._mode = mode

        if mode == 'validation':
            self._online_metric = True
            self._save_validate_reuslts = True
            self._long_metrics = LongMetric('val_close_long')

        elif mode == 'test':
            self._save_validate_reuslts = True

        elif mode == 'plot_rollouts':
            self._plot_rollouts = True

    def init_map_token(self):
        self.argmin_sample_len = 3
        map_token_traj = pickle.load(open(self.map_token_traj_path, 'rb'))
        self.map_token = {'traj_src': map_token_traj['traj_src'], }
        traj_end_theta = np.arctan2(self.map_token['traj_src'][:, -1, 1]-self.map_token['traj_src'][:, -2, 1],
                                    self.map_token['traj_src'][:, -1, 0]-self.map_token['traj_src'][:, -2, 0])
        indices = torch.linspace(0, self.map_token['traj_src'].shape[1]-1, steps=self.argmin_sample_len).long()
        self.map_token['sample_pt'] = torch.from_numpy(self.map_token['traj_src'][:, indices]).to(torch.float)
        self.map_token['traj_end_theta'] = torch.from_numpy(traj_end_theta).to(torch.float)
        self.map_token['traj_src'] = torch.from_numpy(self.map_token['traj_src']).to(torch.float)

    def get_agent_inputs(self, data: HeteroData):
        res = self.encoder.get_agent_inputs(data)
        return res

    def forward(self, data: HeteroData):
        res = self.encoder(data)
        return res

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def check_inputs(self, data: HeteroData):
        inputs = self.get_agent_inputs(data)
        next_token_idx_gt = inputs['next_token_idx_gt']
        next_state_idx_gt = inputs['next_state_idx_gt'].clone()
        next_token_eval_mask = inputs['next_token_eval_mask'].clone()
        raw_agent_valid_mask = inputs['raw_agent_valid_mask'].clone()

        self.StateAccuracy.update(state_idx=next_state_idx_gt,
                                  valid_mask=raw_agent_valid_mask)

        state_token = inputs['state_token']
        grid_index = inputs['grid_index']
        self.GridOverlapRate.update(state_token=state_token,
                                    grid_index=grid_index)

        print(self.StateAccuracy)
        print(self.GridOverlapRate)
        # self.log('valid_accuracy', self.StateAccuracy.compute()['valid'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        # self.log('invalid_accuracy', self.StateAccuracy.compute()['invalid'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

    def training_step(self,
                      data,
                      batch_idx):

        data = self.token_processer(data)

        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)

        # find map tokens for entering agents
        data = self._fetch_enterings(data)

        data['batch_size_a'] = data['agent']['ptr'][1:] - data['agent']['ptr'][:-1]
        data['batch_size_pl'] = data['pt_token']['ptr'][1:] - data['pt_token']['ptr'][:-1]
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        if int(os.getenv("CHECK_INPUTS", 0)):
            return self.check_inputs(data)

        pred = self(data)

        loss = 0

        log_params = dict(prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        if pred.get('occ_decoder', False):

            agent_occ = pred['agent_occ']
            agent_occ_gt = pred['agent_occ_gt']
            agent_occ_eval_mask = pred['agent_occ_eval_mask']
            pt_occ = pred['pt_occ']
            pt_occ_gt = pred['pt_occ_gt']
            pt_occ_eval_mask = pred['pt_occ_eval_mask']

            agent_occ_cls_loss = self.occ_cls_loss(agent_occ[agent_occ_eval_mask],
                                                   agent_occ_gt[agent_occ_eval_mask])
            pt_occ_cls_loss = self.occ_cls_loss(pt_occ[pt_occ_eval_mask],
                                                pt_occ_gt[pt_occ_eval_mask])
            self.log('agent_occ_cls_loss', agent_occ_cls_loss, **log_params)
            self.log('pt_occ_cls_loss', pt_occ_cls_loss, **log_params)
            loss = loss + agent_occ_cls_loss + pt_occ_cls_loss

            # plot
            # plot_scenario_ids = ['74ad7b76d5906d39', '1351ea8b8333ddcb', '1352066cc3c0508d', '135436833ce5b9e7', '13570a32432d449', '13577c32a81336fb']
            if random.random() < 4e-5 or os.getenv('DEBUG'):
                num_step = pred['num_step']
                num_agent = pred['num_agent']
                num_pt = pred['num_pt']
                with torch.no_grad():
                    agent_occ = agent_occ.reshape(num_step, num_agent, -1).transpose(0, 1)
                    agent_occ_gt = agent_occ_gt.reshape(num_step, num_agent).transpose(0, 1)
                    agent_occ_gt[agent_occ_gt == -1] = self.encoder.agent_encoder.grid_size // 2
                    agent_occ_gt = torch.nn.functional.one_hot(agent_occ_gt, num_classes=self.encoder.agent_encoder.grid_size)
                    agent_occ = self.attr_tokenizer.pad_square(agent_occ.softmax(-1).detach().cpu().numpy())[0]
                    agent_occ_gt = self.attr_tokenizer.pad_square(agent_occ_gt.detach().cpu().numpy())[0]
                    plot_occ_grid(pred['scenario_id'][0],
                                  agent_occ,
                                  gt_occ=agent_occ_gt,
                                  mode='agent',
                                  save_path=self.save_path,
                                  prefix=f'training_{self.global_step:06d}_')
                    pt_occ = pt_occ.reshape(num_step, num_pt, -1).transpose(0, 1)
                    pt_occ_gt = pt_occ_gt.reshape(num_step, num_pt).transpose(0, 1)
                    pt_occ_gt[pt_occ_gt == -1] = self.encoder.agent_encoder.grid_size // 2
                    pt_occ_gt = torch.nn.functional.one_hot(pt_occ_gt, num_classes=self.encoder.agent_encoder.grid_size)
                    pt_occ = self.attr_tokenizer.pad_square(pt_occ.sigmoid().detach().cpu().numpy())[0]
                    pt_occ_gt = self.attr_tokenizer.pad_square(pt_occ_gt.detach().cpu().numpy())[0]
                    plot_occ_grid(pred['scenario_id'][0],
                                  pt_occ,
                                  gt_occ=pt_occ_gt,
                                  mode='pt',
                                  save_path=self.save_path,
                                  prefix=f'training_{self.global_step:06d}_')

            return loss

        train_mask = data['agent']['train_mask']
        # remove_ina_mask = data['agent']['remove_ina_mask']

        # motion token loss
        if self.predict_motion:

            next_token_idx = pred['next_token_idx']
            next_token_prob = pred['next_token_prob'] # (a, t, token_size)
            next_token_idx_gt = pred['next_token_idx_gt'] # (a, t)
            next_token_eval_mask = pred['next_token_eval_mask'] # (a, t)
            next_token_eval_mask &= train_mask[:, None]

            token_cls_loss = self.token_cls_loss(next_token_prob[next_token_eval_mask],
                                                 next_token_idx_gt[next_token_eval_mask]) * self.loss_weight['token_cls_loss']
            self.log('token_cls_loss', token_cls_loss, **log_params)

            loss = loss + token_cls_loss

            # record motion predict precision of certain timesteps of centain type of agents
            with torch.no_grad():
                agent_state_idx_gt = data['agent']['state_idx']
                index = torch.nonzero(agent_state_idx_gt == self.enter_state)
                for i in range(10):
                    index[:, 1] += 1
                    index = index[index[:, 1] < agent_state_idx_gt.shape[1] - 1]
                    prob = next_token_prob[index[:, 0], index[:, 1]]
                    gt = next_token_idx_gt[index[:, 0], index[:, 1]]
                    mask = next_token_eval_mask[index[:, 0], index[:, 1]]
                    step_token_cls_loss = self.token_cls_loss(prob[mask], gt[mask])
                    self.log(f's{i}', step_token_cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        # state token loss
        if self.predict_state:

            next_state_idx = pred['next_state_idx']
            next_state_prob = pred['next_state_prob']
            next_state_idx_gt = pred['next_state_idx_gt']
            next_state_eval_mask = pred['next_state_eval_mask'] # (num_agent, num_timestep)

            state_cls_loss = self.state_cls_loss(next_state_prob[next_state_eval_mask],
                                                 next_state_idx_gt[next_state_eval_mask]) * self.loss_weight['state_cls_loss']
            if torch.isnan(state_cls_loss):
                print("Found NaN in state_cls_loss!!!")
                print(next_state_prob.shape)
                print(next_state_idx_gt.shape)
                print(next_state_eval_mask.shape)
                print(next_state_idx_gt[next_state_eval_mask].shape)
            self.log('state_cls_loss', state_cls_loss, **log_params)

            loss = loss + state_cls_loss

            next_state_idx_seed     = pred['next_state_idx_seed']
            next_state_prob_seed    = pred['next_state_prob_seed']
            next_state_idx_gt_seed  = pred['next_state_idx_gt_seed']
            next_type_prob_seed     = pred['next_type_prob_seed']
            next_type_idx_gt_seed   = pred['next_type_idx_gt_seed']
            next_shape_seed         = pred['next_shape_seed']
            next_shape_gt_seed      = pred['next_shape_gt_seed']
            next_state_eval_mask_seed = pred['next_state_eval_mask_seed']
            next_attr_eval_mask_seed = pred['next_attr_eval_mask_seed']
            next_head_eval_mask_seed    = pred['next_head_eval_mask_seed']

            # when num_seed_gt=0 loss term will be NaN
            state_cls_loss_seed = self.state_cls_loss_seed(next_state_prob_seed[next_state_eval_mask_seed],
                                                           next_state_idx_gt_seed[next_state_eval_mask_seed]) * self.loss_weight['state_cls_loss']
            state_cls_loss_seed = torch.nan_to_num(state_cls_loss_seed)
            self.log('seed_state_cls_loss', state_cls_loss_seed, **log_params)

            type_cls_loss_seed = self.type_cls_loss_seed(next_type_prob_seed[next_attr_eval_mask_seed],
                                                         next_type_idx_gt_seed[next_attr_eval_mask_seed]) * self.loss_weight['type_cls_loss']
            shape_reg_loss_seed = self.shape_reg_loss_seed(next_shape_seed[next_attr_eval_mask_seed],
                                                           next_shape_gt_seed[next_attr_eval_mask_seed]) * self.loss_weight['shape_reg_loss']
            type_cls_loss_seed = torch.nan_to_num(type_cls_loss_seed)
            shape_reg_loss_seed = torch.nan_to_num(shape_reg_loss_seed)
            self.log('seed_type_cls_loss', type_cls_loss_seed, **log_params)
            self.log('seed_shape_reg_loss', shape_reg_loss_seed, **log_params)

            loss = loss + state_cls_loss_seed + type_cls_loss_seed + shape_reg_loss_seed


            if self.use_grid_token:
                next_pos_rel_prob_seed      = pred['next_pos_rel_prob_seed']
                next_pos_rel_index_gt_seed  = pred['next_pos_rel_index_gt_seed']
                next_offset_xy_seed         = pred['next_offset_xy_seed']
                next_offset_xy_gt_seed      = pred['next_offset_xy_gt_seed']

                pos_cls_loss_seed = self.pos_cls_loss_seed(next_pos_rel_prob_seed[next_attr_eval_mask_seed],
                                                           next_pos_rel_index_gt_seed[next_attr_eval_mask_seed]) * self.loss_weight['pos_cls_loss']
                offset_reg_loss_seed = self.offset_reg_loss_seed(next_offset_xy_seed[next_head_eval_mask_seed],
                                                                next_offset_xy_gt_seed[next_head_eval_mask_seed]) * self.loss_weight['offset_reg_loss']
                pos_cls_loss_seed = torch.nan_to_num(pos_cls_loss_seed)
                self.log('seed_pos_cls_loss', pos_cls_loss_seed, **log_params)
                self.log('seed_offset_reg_loss', offset_reg_loss_seed, **log_params)

                loss = loss + pos_cls_loss_seed + offset_reg_loss_seed

            else:
                next_pos_rel_xy_seed       = pred['next_pos_rel_xy_seed']
                next_pos_rel_xy_gt_seed    = pred['next_pos_rel_xy_gt_seed']
                pos_reg_loss_seed = self.pos_reg_loss_seed(next_pos_rel_xy_seed[next_attr_eval_mask_seed],
                                                           next_pos_rel_xy_gt_seed[next_attr_eval_mask_seed]) * self.loss_weight['pos_reg_loss']
                pos_reg_loss_seed = torch.nan_to_num(pos_reg_loss_seed)
                self.log('seed_pos_reg_loss', pos_reg_loss_seed, **log_params)

                loss = loss + pos_reg_loss_seed

            if self.use_head_token:
                next_head_rel_prob_seed     = pred['next_head_rel_prob_seed']
                next_head_rel_index_gt_seed = pred['next_head_rel_index_gt_seed']

                head_cls_loss_seed = self.head_cls_loss_seed(next_head_rel_prob_seed[next_head_eval_mask_seed],
                                                             next_head_rel_index_gt_seed[next_head_eval_mask_seed]) * self.loss_weight['head_cls_loss']
                self.log('seed_head_cls_loss', head_cls_loss_seed, **log_params)

                loss = loss + head_cls_loss_seed

            else:
                next_head_rel_theta_seed    = pred['next_head_rel_theta_seed']
                next_head_rel_theta_gt_seed = pred['next_head_rel_theta_gt_seed']

                head_reg_loss_seed = self.head_reg_loss_seed(next_head_rel_theta_seed[next_head_eval_mask_seed],
                                                             next_head_rel_theta_gt_seed[next_head_eval_mask_seed]) * self.loss_weight['head_reg_loss']
                self.log('seed_head_reg_loss', head_reg_loss_seed, **log_params)

                loss = loss + head_reg_loss_seed

            # plot_scenario_ids = ['74ad7b76d5906d39', '1351ea8b8333ddcb', '1352066cc3c0508d', '135436833ce5b9e7', '13570a32432d449', '13577c32a81336fb']
            if random.random() < 4e-5 or int(os.getenv('DEBUG', 0)):
                with torch.no_grad():
                    # plot probability of inserting new agent (agent-timestep)
                    raw_next_state_prob_seed = pred['raw_next_state_prob_seed']
                    plot_prob_seed(pred['scenario_id'][0],
                                   torch.softmax(raw_next_state_prob_seed, dim=-1
                                                 )[..., -1].detach().cpu().numpy(),
                                   self.save_path,
                                   prefix=f'training_{self.global_step:06d}_',
                                   indices=pred['target_indices'].cpu().numpy())

                    # plot heatmap of inserting new agent
                    if self.use_grid_token:
                        next_pos_rel_prob_seed = pred['next_pos_rel_prob_seed']
                        if next_pos_rel_prob_seed.shape[0] > 0:
                            next_pos_rel_prob_seed = torch.softmax(next_pos_rel_prob_seed, dim=-1).detach().cpu().numpy()
                            indices = next_pos_rel_index_gt_seed.detach().cpu().numpy()
                            mask = next_attr_eval_mask_seed.detach().cpu().numpy().astype(np.bool_)
                            indices[~mask] = -1
                            prob, indices = self.attr_tokenizer.pad_square(next_pos_rel_prob_seed, indices)
                            plot_insert_grid(pred['scenario_id'][0],
                                            prob,
                                            indices=indices,
                                            save_path=self.save_path,
                                            prefix=f'training_{self.global_step:06d}_')

        if self.predict_occ:

            neighbor_agent_grid_idx             = pred['neighbor_agent_grid_idx']
            neighbor_agent_grid_index_gt        = pred['neighbor_agent_grid_index_gt']
            neighbor_agent_grid_index_eval_mask = pred['neighbor_agent_grid_index_eval_mask']
            neighbor_pt_grid_idx                = pred['neighbor_pt_grid_idx']
            neighbor_pt_grid_index_gt           = pred['neighbor_pt_grid_index_gt']
            neighbor_pt_grid_index_eval_mask    = pred['neighbor_pt_grid_index_eval_mask']

            neighbor_agent_grid_cls_loss = self.occ_cls_loss(neighbor_agent_grid_idx[neighbor_agent_grid_index_eval_mask],
                                                             neighbor_agent_grid_index_gt[neighbor_agent_grid_index_eval_mask])
            neighbor_pt_grid_cls_loss = self.occ_cls_loss(neighbor_pt_grid_idx[neighbor_pt_grid_index_eval_mask],
                                                          neighbor_pt_grid_index_gt[neighbor_pt_grid_index_eval_mask])
            # self.log('neighbor_agent_grid_cls_loss', neighbor_agent_grid_cls_loss, **log_params)
            # self.log('neighbor_pt_grid_cls_loss', neighbor_pt_grid_cls_loss, **log_params)
            # loss = loss + neighbor_agent_grid_cls_loss + neighbor_pt_grid_cls_loss

            grid_agent_occ_seed = pred['grid_agent_occ_seed']
            grid_pt_occ_seed = pred['grid_pt_occ_seed']
            grid_agent_occ_gt_seed = pred['grid_agent_occ_gt_seed'].float()
            grid_pt_occ_gt_seed = pred['grid_pt_occ_gt_seed'].float()
            grid_agent_occ_eval_mask_seed = pred['grid_agent_occ_eval_mask_seed']
            grid_pt_occ_eval_mask_seed = pred['grid_pt_occ_eval_mask_seed']

            # plot_scenario_ids = ['74ad7b76d5906d39', '1351ea8b8333ddcb', '1352066cc3c0508d', '135436833ce5b9e7', '13570a32432d449', '13577c32a81336fb']
            if random.random() < 4e-5 or os.getenv('DEBUG'):
                with torch.no_grad():
                    agent_occ = self.attr_tokenizer.pad_square(grid_agent_occ_seed.sigmoid().detach().cpu().numpy())[0]
                    agent_occ_gt = self.attr_tokenizer.pad_square(grid_agent_occ_gt_seed.detach().cpu().numpy())[0]
                    plot_occ_grid(pred['scenario_id'][0],
                                 agent_occ,
                                 gt_occ=agent_occ_gt,
                                 mode='agent',
                                 save_path=self.save_path,
                                 prefix=f'training_{self.global_step:06d}_')
                    pt_occ = self.attr_tokenizer.pad_square(grid_pt_occ_seed.sigmoid().detach().cpu().numpy())[0]
                    pt_occ_gt = self.attr_tokenizer.pad_square(grid_pt_occ_gt_seed.detach().cpu().numpy())[0]
                    plot_occ_grid(pred['scenario_id'][0],
                                 pt_occ,
                                 gt_occ=pt_occ_gt,
                                 mode='pt',
                                 save_path=self.save_path,
                                 prefix=f'training_{self.global_step:06d}_')

            grid_agent_occ_gt_seed[grid_agent_occ_gt_seed == -1] = 0
            if grid_agent_occ_gt_seed.min() < 0 or grid_agent_occ_gt_seed.max() > 1 or \
               grid_pt_occ_gt_seed.min() < 0 or grid_pt_occ_gt_seed.max() > 1:
                   raise RuntimeError("Occurred invalid values in occ gt")

            agent_occ_loss = self.agent_occ_loss_seed(grid_agent_occ_seed[grid_agent_occ_eval_mask_seed],
                                                      grid_agent_occ_gt_seed[grid_agent_occ_eval_mask_seed]) * self.loss_weight['agent_occ_loss']
            pt_occ_loss = self.pt_occ_loss_seed(grid_pt_occ_seed[grid_pt_occ_eval_mask_seed],
                                                grid_pt_occ_gt_seed[grid_pt_occ_eval_mask_seed]) * self.loss_weight['pt_occ_loss']

            self.log('agent_occ_loss', agent_occ_loss, **log_params)
            self.log('pt_occ_loss', pt_occ_loss, **log_params)
            loss = loss + agent_occ_loss + pt_occ_loss

        if os.getenv('LOG_TRAIN', False) and (self.predict_motion or self.predict_state):
            for a in range(next_token_idx.shape[0]):
                print(f"agent: {a}")
                if self.predict_motion:
                    print(f"pred motion: {next_token_idx[a, :, 0].tolist()}, \ngt motion:   {next_token_idx_gt[a, :].tolist()}")
                    print(f"train mask: {next_token_eval_mask[a].long().tolist()}")
                if self.predict_state:
                    print(f"pred state: {next_state_idx[a, :, 0].tolist()}, \ngt state:   {next_state_idx_gt[a, :].tolist()}")
                    print(f"train mask: {next_state_eval_mask[a].long().tolist()}")
            num_sa = next_state_idx_seed[..., 0].sum(dim=-1).bool().sum()
            for sa in range(num_sa):
                print(f"seed agent: {sa}")
                print(f"seed pred state: {next_state_idx_seed[sa, :, 0].tolist()}, \ngt seed state:   {next_state_idx_gt_seed[sa, :].tolist()}")
                # if sa < next_pos_rel_seed.shape[0]:
                #     print(f"pred pos: {next_pos_rel_seed[sa, :, 0].tolist()}, \ngt pos:   {next_pos_rel_gt_seed[sa, :, 0].tolist()}")
                #     print(f"pred head: {next_head_rel_seed[sa].tolist()}, \ngt head:   {next_head_rel_gt_seed[sa].tolist()}")
                # print(f"seed train mask: {next_state_eval_mask_seed[sa].long().tolist()}")

        # map token loss
        if self.predict_map:

            map_next_token_prob = pred['map_next_token_prob']
            map_next_token_idx_gt = pred['map_next_token_idx_gt']
            map_next_token_eval_mask = pred['map_next_token_eval_mask']

            map_token_loss = self.map_token_loss(map_next_token_prob[map_next_token_eval_mask], map_next_token_idx_gt[map_next_token_eval_mask]) * self.loss_weight['map_token_loss']
            self.log('map_token_loss', map_token_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            loss = loss + map_token_loss

        allocated = torch.cuda.memory_allocated(device='cuda:0') / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device='cuda:0') / (1024 ** 3)
        self.log('allocated', allocated, **log_params)
        self.log('reserved', reserved, **log_params)

        return loss

    def validation_step(self,
                        data,
                        batch_idx):

        self.debug = int(os.getenv('DEBUG', 0))

        # ! validation in training process
        if (
            self._mode == 'training' and (
            self.current_epoch not in [5, 10, 20, 25, self.max_epochs] or random.random() > 5e-4) and
            not self.debug
        ):
            self.val_open_loop = False
            self.val_close_loop = False
            return

        if int(os.getenv('NO_VAL', 0)) or int(os.getenv("CHECK_INPUTS", 0)):
            return

        # ! check if save exists
        if not self._plot_rollouts:
            rollouts_path = os.path.join(self.save_path, f'idx_{self.trainer.global_rank}_{batch_idx}_rollouts.pkl')
            if os.path.exists(rollouts_path):
                tqdm.write(f'Skipped batch {batch_idx}')
                return
        else:
            rollouts_path = os.path.join(self.save_path, f'{data["scenario_id"][0]}.gif')
            if os.path.exists(rollouts_path):
                tqdm.write(f'Skipped scenario {data["scenario_id"][0]}')
                return

        # ! data preparation
        data = self.token_processer(data)

        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)

        # find map tokens for entering agents
        data = self._fetch_enterings(data)

        data['batch_size_a'] = data['agent']['ptr'][1:] - data['agent']['ptr'][:-1]
        data['batch_size_pl'] = data['pt_token']['ptr'][1:] - data['pt_token']['ptr'][:-1]
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        if int(os.getenv('NEAREST_POS', 0)):
            pred = self.encoder.predict_nearest_pos(data, rank=self.local_rank)
            return

        # if self.insert_agent:
        #     pred = self.encoder.insert_agent(data)
        #     return

        # ! open-loop validation
        if self.val_open_loop or int(os.getenv('OPEN_LOOP', 0)):

            pred = self(data)

            # pred['next_state_prob_seed'] = torch.softmax(pred['next_state_prob_seed'], dim=-1)[..., -1]
            # plot_prob_seed(pred, self.save_path, suffix=f'_training')

            loss = 0

            if self.predict_motion:

                # motion token
                next_token_idx = pred['next_token_idx']
                next_token_idx_gt = pred['next_token_idx_gt'] # (num_agent, num_step, 10)
                next_token_prob = pred['next_token_prob']
                next_token_eval_mask = pred['next_token_eval_mask']

                token_cls_loss = self.token_cls_loss(next_token_prob[next_token_eval_mask], next_token_idx_gt[next_token_eval_mask])
                loss = loss + token_cls_loss

            if self.predict_state:

                # state token
                next_state_idx = pred['next_state_idx']
                next_state_idx_gt = pred['next_state_idx_gt']
                next_state_prob = pred['next_state_prob']
                next_state_eval_mask = pred['next_state_eval_mask']

                state_cls_loss = self.state_cls_loss(next_state_prob[next_state_eval_mask], next_state_idx_gt[next_state_eval_mask])
                loss = loss + state_cls_loss

                # seed state token
                next_state_idx_seed = pred['next_state_idx_seed']
                next_state_idx_gt_seed = pred['next_state_idx_gt_seed']

            if self.predict_occ:

                grid_agent_occ_seed = pred['grid_agent_occ_seed']
                grid_pt_occ_seed = pred['grid_pt_occ_seed']
                grid_agent_occ_gt_seed = pred['grid_agent_occ_gt_seed'].float()
                grid_pt_occ_gt_seed = pred['grid_pt_occ_gt_seed'].float()

                agent_occ = self.attr_tokenizer.pad_square(grid_agent_occ_seed.sigmoid().detach().cpu().numpy())[0]
                agent_occ_gt = self.attr_tokenizer.pad_square(grid_agent_occ_gt_seed.detach().cpu().numpy())[0]
                plot_occ_grid(pred['scenario_id'][0],
                              agent_occ,
                              gt_occ=agent_occ_gt,
                              mode='agent',
                              save_path=self.save_path,
                              prefix=f'eval_')
                pt_occ = self.attr_tokenizer.pad_square(grid_pt_occ_seed.sigmoid().detach().cpu().numpy())[0]
                pt_occ_gt = self.attr_tokenizer.pad_square(grid_pt_occ_gt_seed.detach().cpu().numpy())[0]
                plot_occ_grid(pred['scenario_id'][0],
                              pt_occ,
                              gt_occ=pt_occ_gt,
                              mode='pt',
                              save_path=self.save_path,
                              prefix=f'eval_')

            self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        if self.val_insert:

            pred = self(data)

            next_state_idx_seed = pred['next_state_idx_seed']
            next_state_idx_gt_seed = pred['next_state_idx_gt_seed']

            self.NumInsertAccuracy.update(next_state_idx_seed=next_state_idx_seed,
                                          next_state_idx_gt_seed=next_state_idx_gt_seed)

            return

        # ! close-loop validation
        if self.val_close_loop and (self.predict_motion or self.predict_state):

            rollouts = []
            for _ in tqdm(range(self.n_rollout_close_val), leave=False, desc='Rollout ...'):
                rollout = self.encoder.inference(data.clone())
            rollouts.append(rollout)

            av_index = int(rollout['ego_index'])
            scenario_id = rollout['scenario_id'][0]

            # motion tokens
            if self.predict_motion:

                if self._plot_rollouts:  # only plot gifs for last 2 epochs for efficiency
                    plot_val(data, rollout, av_index, self.save_path, pl2seed_radius=self.pl2seed_radius, attr_tokenizer=self.attr_tokenizer)

                # next_token_idx = pred['next_token_idx'][..., None]
                # next_token_idx_gt = pred['next_token_idx_gt'][:, 2:] # hard code 2=11//5
                # next_token_eval_mask = pred['next_token_eval_mask'][:, 2:]

                # gt_traj = pred['gt_traj']
                # pred_traj = pred['pred_traj']
                # pred_head = pred['pred_head']

                # self.TokenCls.update(pred=next_token_idx[next_token_eval_mask], target=next_token_idx_gt[next_token_eval_mask],
                #                      valid_mask=next_token_eval_mask[next_token_eval_mask])
                # self.log('val_token_cls_acc', self.TokenCls, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

                # remove the agents which are unseen at current step
                # eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]

                # self.minADE.update(pred=pred_traj[eval_mask], target=gt_traj[eval_mask], valid_mask=valid_mask[eval_mask])
                # self.minFDE.update(pred=pred_traj[eval_mask], target=gt_traj[eval_mask], valid_mask=valid_mask[eval_mask])
                # print('ade: ', self.minADE.compute(), 'fde: ', self.minFDE.compute())

                # self.log('val_minADE', self.minADE, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
                # self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

            # state tokens
            if self.predict_state:

                if self.use_grid_token:
                    next_pos_rel_prob_seed = rollout['next_pos_rel_prob_seed'].cpu().numpy() # (s, t, grid_size)
                    prob, _ = self.attr_tokenizer.pad_square(next_pos_rel_prob_seed)

                if self._plot_rollouts:
                    if self.use_grid_token:
                        plot_insert_grid(scenario_id,
                                        prob,
                                        save_path=self.save_path,
                                        prefix=f'inference_')
                    plot_prob_seed(scenario_id,
                                   rollout['next_state_prob_seed'].cpu().numpy(),
                                   self.save_path,
                                   prefix=f'inference_')

                next_state_idx = rollout['next_state_idx'][..., None]
                # next_state_idx_gt = rollout['next_state_idx_gt'][:, 2:]
                # next_state_eval_mask = rollout['next_state_eval_mask'][:, 2:]

                # self.StateCls.update(pred=next_state_idx[next_token_eval_mask], target=next_state_idx_gt[next_token_eval_mask],
                #                      valid_mask=next_token_eval_mask[next_token_eval_mask])
                # self.log('val_state_cls_acc', self.TokenCls, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

                self.StateAccuracy.update(state_idx=next_state_idx[..., 0])
                self.log('valid_accuracy', self.StateAccuracy.compute()['valid'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
                self.log('invalid_accuracy', self.StateAccuracy.compute()['invalid'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
                self.local_logger.info(rollout['log_message'])
                # print(rollout['log_message'])
                # print(self.StateAccuracy)

            if self.predict_occ:

                grid_agent_occ_seed = rollout['grid_agent_occ_seed']
                grid_pt_occ_seed = rollout['grid_pt_occ_seed']
                grid_agent_occ_gt_seed = rollout['grid_agent_occ_gt_seed']

                agent_occ = self.attr_tokenizer.pad_square(grid_agent_occ_seed.sigmoid().cpu().numpy())[0]
                agent_occ_gt = self.attr_tokenizer.pad_square(grid_agent_occ_gt_seed.sigmoid().cpu().numpy())[0]
                if self._plot_rollouts:
                    plot_occ_grid(scenario_id,
                                  agent_occ,
                                  gt_occ=agent_occ_gt,
                                  mode='agent',
                                  save_path=self.save_path,
                                  prefix=f'inference_')

            if self._online_metric or self._save_validate_reuslts:

                # ! format results
                pred_valid, token_pos, token_head = [], [], []
                pred_traj, pred_head, pred_z = [], [], []
                pred_shape, pred_type, pred_state = [], [], []
                agent_id = []
                for rollout in rollouts:
                    pred_valid.append(rollout['pred_valid'])
                    token_pos.append(rollout['pos_a'])
                    token_head.append(rollout['head_a'])
                    pred_traj.append(rollout['pred_traj'])
                    pred_head.append(rollout['pred_head'])
                    pred_z.append(rollout['pred_z'])
                    pred_shape.append(rollout['eval_shape'])
                    pred_type.append(rollout['pred_type'])
                    pred_state.append(rollout['next_state_idx'])
                    agent_id.append(rollout['agent_id'])

                pred_valid = torch.stack(pred_valid, dim=1)
                token_pos = torch.stack(token_pos, dim=1)
                token_head = torch.stack(token_head, dim=1)
                pred_traj = torch.stack(pred_traj, dim=1)  # (n_agent, n_rollout, n_step, 2)
                pred_head = torch.stack(pred_head, dim=1)
                pred_z = torch.stack(pred_z, dim=1)
                pred_shape = torch.stack(pred_shape, dim=1)  # [n_agent, n_rollout, 3]
                pred_type = torch.stack(pred_type, dim=1)  # [n_agent, n_rollout]
                pred_state = torch.stack(pred_state, dim=1)  # [n_agent, n_rollout, n_step // shift]
                agent_id = torch.stack(agent_id, dim=1)  # [n_agent, n_rollout]

                agent_batch = torch.zeros((pred_traj.shape[0]), dtype=torch.long)
                rollouts = dict(
                    _scenario_id=data['scenario_id'],
                    scenario_id=get_scenario_id_int_tensor(data['scenario_id']),
                    av_id=int(rollouts[0]['agent_id'][rollouts[0]['ego_index']]),  # NOTE: hard code!!!
                    agent_id=agent_id.cpu(),
                    agent_batch=agent_batch.cpu(),
                    pred_traj=pred_traj.cpu(),
                    pred_z=pred_z.cpu(),
                    pred_head=pred_head.cpu(),
                    pred_shape=pred_shape.cpu(),
                    pred_type=pred_type.cpu(),
                    pred_state=pred_state.cpu(),
                    pred_valid=pred_valid.cpu(),
                    token_pos=token_pos.cpu(),
                    token_head=token_head.cpu(),
                    tfrecord_path=data['tfrecord_path'],
                )

                if self._save_validate_reuslts:
                    with open(rollouts_path, 'wb') as f:
                        pickle.dump(rollouts, f)

                if self._online_metric:
                    self._long_metrics.update(rollouts)

    def on_validation_start(self):
        self.scenario_rollouts = []
        self.batch_metric = defaultdict(list)

    def on_validation_epoch_end(self):
        if self.val_close_loop:

            if self._long_metrics is not None:
                epoch_long_metrics = self._long_metrics.compute()
                if self.global_rank == 0:
                    epoch_long_metrics['epoch'] = self.current_epoch
                    self.logger.log_metrics(epoch_long_metrics)

                self._long_metrics.reset()

            self.minADE.reset()
            self.minFDE.reset()
            self.StateAccuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            if current_step + 1 < self.warmup_steps:
                return float(current_step + 1) / float(max(1, self.warmup_steps))
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def load_state_from_file(self, filename, to_cpu=False):
        logger = self.local_logger

        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)


        model_state_disk = checkpoint['state_dict']
        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')

        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter
        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)

        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'The number of missing keys: {len(missing_keys)}')
        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch

    def match_token_map(self, data):
        traj_pos = data['map_save']['traj_pos'].to(torch.float)
        traj_theta = data['map_save']['traj_theta'].to(torch.float)
        pl_idx_list = data['map_save']['pl_idx_list']
        token_sample_pt = self.map_token['sample_pt'].to(traj_pos.device)
        token_src = self.map_token['traj_src'].to(traj_pos.device)
        max_traj_len = self.map_token['traj_src'].shape[1]
        pl_num = traj_pos.shape[0]

        pt_token_pos = traj_pos[:, 0, :].clone()
        pt_token_orientation = traj_theta.clone()
        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = -sin
        rot_mat[..., 1, 0] = sin
        rot_mat[..., 1, 1] = cos
        traj_pos_local = torch.bmm((traj_pos - traj_pos[:, 0:1]), rot_mat.view(-1, 2, 2))
        distance = torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1)) ** 2, dim=(-2, -1))
        pt_token_id = torch.argmin(distance, dim=1)

        if self.noise:
            topk_indices = torch.argsort(torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1))**2, dim=(-2, -1)), dim=1)[:, :8]
            sample_topk = torch.randint(0, topk_indices.shape[-1], size=(topk_indices.shape[0], 1), device=topk_indices.device)
            pt_token_id = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)

        # cos, sin = traj_theta.cos(), traj_theta.sin()
        # rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        # rot_mat[..., 0, 0] = cos
        # rot_mat[..., 0, 1] = sin
        # rot_mat[..., 1, 0] = -sin
        # rot_mat[..., 1, 1] = cos
        # token_src_world = torch.bmm(token_src[None, ...].repeat(pl_num, 1, 1, 1).reshape(pl_num, -1, 2),
        #                             rot_mat.view(-1, 2, 2)).reshape(pl_num, token_src.shape[0], max_traj_len, 2) + traj_pos[:, None, [0], :]
        # token_src_world_select = token_src_world.view(-1, 1024, 11, 2)[torch.arange(pt_token_id.view(-1).shape[0]), pt_token_id.view(-1)].view(pl_num, max_traj_len, 2)

        pl_idx_full = pl_idx_list.clone()
        token2pl = torch.stack([torch.arange(len(pl_idx_list), device=traj_pos.device), pl_idx_full.long()])
        count_nums = []
        for pl in pl_idx_full.unique():
            pt = token2pl[0, token2pl[1, :] == pl]
            left_side = (data['pt_token']['side'][pt] == 0).sum()
            right_side = (data['pt_token']['side'][pt] == 1).sum()
            center_side = (data['pt_token']['side'][pt] == 2).sum()
            count_nums.append(torch.Tensor([left_side, right_side, center_side]))
        count_nums = torch.stack(count_nums, dim=0)
        num_polyline = int(count_nums.max().item())
        traj_mask = torch.zeros((int(len(pl_idx_full.unique())), 3, num_polyline), dtype=bool)
        idx_matrix = torch.arange(traj_mask.size(2)).unsqueeze(0).unsqueeze(0)
        idx_matrix = idx_matrix.expand(traj_mask.size(0), traj_mask.size(1), -1)
        counts_num_expanded = count_nums.unsqueeze(-1)
        mask_update = idx_matrix < counts_num_expanded
        traj_mask[mask_update] = True

        data['pt_token']['traj_mask'] = traj_mask
        data['pt_token']['position'] = torch.cat([pt_token_pos, torch.zeros((data['pt_token']['num_nodes'], 1),
                                                                            device=traj_pos.device, dtype=torch.float)], dim=-1)
        data['pt_token']['orientation'] = pt_token_orientation
        data['pt_token']['height'] = data['pt_token']['position'][:, -1]
        data[('pt_token', 'to', 'map_polygon')] = {}
        data[('pt_token', 'to', 'map_polygon')]['edge_index'] = token2pl  # (2, num_points)
        data['pt_token']['token_idx'] = pt_token_id

        # data['pt_token']['batch'] = torch.zeros(data['pt_token']['num_nodes'], device=traj_pos.device).long()
        # data['pt_token']['ptr'] = torch.tensor([0, data['pt_token']['num_nodes']], device=traj_pos.device).long()

        return data

    def sample_pt_pred(self, data):
        traj_mask = data['pt_token']['traj_mask']
        raw_pt_index = torch.arange(1, traj_mask.shape[2]).repeat(traj_mask.shape[0], traj_mask.shape[1], 1)
        masked_pt_index = raw_pt_index.view(-1)[torch.randperm(raw_pt_index.numel())[:traj_mask.shape[0] * traj_mask.shape[1] * ((traj_mask.shape[2] - 1) // 3)].reshape(traj_mask.shape[0], traj_mask.shape[1], (traj_mask.shape[2] - 1) // 3)]
        masked_pt_index = torch.sort(masked_pt_index, -1)[0]
        pt_valid_mask = traj_mask.clone()
        pt_valid_mask.scatter_(2, masked_pt_index, False)
        pt_pred_mask = traj_mask.clone()
        pt_pred_mask.scatter_(2, masked_pt_index, False)
        tmp_mask = pt_pred_mask.clone()
        tmp_mask[:, :, :] = True
        tmp_mask.scatter_(2, masked_pt_index-1, False)
        pt_pred_mask.masked_fill_(tmp_mask, False)
        pt_pred_mask = pt_pred_mask * torch.roll(traj_mask, shifts=-1, dims=2)
        pt_target_mask = torch.roll(pt_pred_mask, shifts=1, dims=2)

        data['pt_token']['pt_valid_mask'] = pt_valid_mask[traj_mask]
        data['pt_token']['pt_pred_mask'] = pt_pred_mask[traj_mask]
        data['pt_token']['pt_target_mask'] = pt_target_mask[traj_mask]

        return data

    def _fetch_enterings(self, data: HeteroData, plot: bool=False):
        data['agent']['grid_token_idx']     = torch.zeros_like(data['agent']['state_idx']).long()
        data['agent']['grid_offset_xy']     = torch.zeros_like(data['agent']['token_pos'])
        data['agent']['heading_token_idx']  = torch.zeros_like(data['agent']['state_idx']).long()
        data['agent']['sort_indices']       = torch.zeros_like(data['agent']['state_idx']).long()
        data['agent']['inrange_mask']       = torch.zeros_like(data['agent']['state_idx']).bool()
        data['agent']['bos_mask']           = torch.zeros_like(data['agent']['state_idx']).bool()

        data['agent']['pos_xy']     = torch.zeros_like(data['agent']['token_pos'])
        data['agent']['heading_theta'] = torch.zeros_like(data['agent']['token_heading'])
        if self.predict_occ:
            num_step = data['agent']['state_idx'].shape[1]
            data['agent']['pt_grid_token_idx'] = torch.zeros_like(data['pt_token']['token_idx'])[None].repeat(num_step, 1).long()

        for b in range(data.num_graphs):
            av_index = int(data['agent']['av_index'][b])
            agent_batch_mask = data['agent']['batch'] == b
            pt_batch_mask = data['pt_token']['batch'] == b
            pt_token_idx = data['pt_token']['token_idx'][pt_batch_mask]
            pt_pos = data['pt_token']['position'][pt_batch_mask]
            agent_token_pos = data['agent']['token_pos'][agent_batch_mask]
            agent_token_heading = data['agent']['token_heading'][agent_batch_mask]
            state_idx = data['agent']['state_idx'][agent_batch_mask]
            ego_pos = agent_token_pos[av_index]  # NOTE: `av_index` will be added by `ptr` later
            ego_heading = agent_token_heading[av_index]

            grid_token_idx = torch.full(state_idx.shape, -1, device=state_idx.device)
            offset_xy = torch.zeros_like(agent_token_pos)
            sort_indices = torch.zeros_like(grid_token_idx)
            pt_grid_token_idx = torch.full((state_idx.shape[1], *pt_token_idx.shape), -1, device=pt_token_idx.device)

            pos_xy = torch.zeros((*state_idx.shape, 2), device=state_idx.device)

            is_bos = []
            is_inrange = []
            for t in range(agent_token_pos.shape[1]): # num_step

                # tokenize position
                is_bos_t = state_idx[:, t] == self.enter_state
                is_invalid_t = state_idx[:, t] == self.invalid_state
                is_inrange_t = ((agent_token_pos[:, t] - ego_pos[[t]]) ** 2).sum(-1).sqrt() <= self.pl2seed_radius
                grid_index_t, offset_xy_t = self.attr_tokenizer.encode_pos(x=agent_token_pos[~is_invalid_t & is_inrange_t, t],
                                                                           y=ego_pos[[t]],
                                                                           theta_y=ego_heading[[t]])
                grid_token_idx[~is_invalid_t & is_inrange_t, t] = grid_index_t
                offset_xy[~is_invalid_t & is_inrange_t, t] = offset_xy_t

                pos_xy[~is_invalid_t & is_inrange_t, t] = agent_token_pos[~is_invalid_t & is_inrange_t, t] - ego_pos[[t]]

                # distance = ((agent_token_pos[:, t] - ego_pos[[t]]) ** 2).sum(-1).sqrt()
                head_vector = torch.stack([ego_heading[[t]].cos(), ego_heading[[t]].sin()], dim=-1)
                distance = angle_between_2d_vectors(ctr_vector=head_vector,
                                                    nbr_vector=agent_token_pos[:, t] - ego_pos[[t]])
                # distance = torch.rand(agent_token_pos.shape[0], device=agent_token_pos.device)
                distance[~(is_bos_t & is_inrange_t)] = torch.inf
                sort_dist, sort_indice = distance.sort()
                sort_indice[torch.isinf(sort_dist)] = av_index
                sort_indices[:, t] = sort_indice

                is_bos.append(is_bos_t)
                is_inrange.append(is_inrange_t)

                # tokenize pt token
                if self.predict_occ:
                    is_inrange_t = ((pt_pos[:, :2] - ego_pos[None, t]) ** 2).sum(-1).sqrt() <= self.pl2seed_radius
                    grid_index_t, _ = self.attr_tokenizer.encode_pos(x=pt_pos[is_inrange_t, :2],
                                                                     y=ego_pos[[t]],
                                                                     theta_y=ego_heading[[t]])

                    pt_grid_token_idx[t, is_inrange_t] = grid_index_t

            # tokenize heading
            rel_heading = agent_token_heading - ego_heading[None, ...]
            heading_token_idx = self.attr_tokenizer.encode_heading(rel_heading)

            data['agent']['grid_token_idx'][agent_batch_mask] = grid_token_idx
            data['agent']['grid_offset_xy'][agent_batch_mask] = offset_xy
            data['agent']['heading_token_idx'][agent_batch_mask] = heading_token_idx
            data['agent']['pos_xy'][agent_batch_mask] = pos_xy
            data['agent']['heading_theta'][agent_batch_mask] = wrap_angle(rel_heading)
            data['agent']['sort_indices'][agent_batch_mask] = sort_indices
            data['agent']['inrange_mask'][agent_batch_mask] = torch.stack(is_inrange, dim=1)
            data['agent']['bos_mask'][agent_batch_mask] = torch.stack(is_bos, dim=1)
            if self.predict_occ:
                data['agent']['pt_grid_token_idx'][:, pt_batch_mask] = pt_grid_token_idx

            plot = False
            if plot:
                scenario_id = data['scenario_id'][b]
                dummy_prob = np.zeros((ego_pos.shape[0], self.attr_tokenizer.grid.shape[0])) + .5
                indices = grid_token_idx[:, 1:][state_idx[:, 1:] == self.enter_state].reshape(-1).cpu().numpy()
                dummy_prob, indices = self.attr_tokenizer.pad_square(dummy_prob, indices)
                # plot_insert_grid(scenario_id, dummy_prob,
                #                  self.attr_tokenizer.grid.cpu().numpy(),
                #                  ego_pos.cpu().numpy(),
                #                  None,
                #                  save_path=os.path.join(self.save_path, 'vis'),
                #                  indices=indices[np.newaxis, ...],
                #                  inference=True,
                #                  all_t_in_one=True)

                enter_index = [grid_token_idx[:, i][state_idx[:, i] == self.enter_state].tolist()
                               for i in range(agent_token_pos.shape[1])]
                agent_labels = [[f'A{i}'] * agent_token_pos.shape[1] for i in range(agent_token_pos.shape[0])]
                plot_scenario(scenario_id,
                              data['map_point']['position'].cpu().numpy(),
                              agent_token_pos.cpu().numpy(),
                              agent_token_heading.cpu().numpy(),
                              state_idx.cpu().numpy(),
                              types=list(map(lambda i: self.encoder.agent_encoder.agent_type[i],
                                             data['agent']['type'].tolist())),
                              av_index=av_index,
                              pl2seed_radius=self.pl2seed_radius,
                              attr_tokenizer=self.attr_tokenizer,
                              enter_index=enter_index,
                              save_gif=False,
                              save_path=os.path.join(self.save_path, 'vis'),
                              agent_labels=agent_labels,
                              tokenized=True)

        return data
