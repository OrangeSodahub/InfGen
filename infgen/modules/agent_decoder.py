import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Mapping, Optional, Literal
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import dense_to_sparse, subgraph
from scipy.optimize import linear_sum_assignment

from infgen.modules.attr_tokenizer import Attr_Tokenizer
from infgen.modules.layers import *
from infgen.utils.visualization import *
from infgen.datasets.preprocess import AGENT_SHAPE, AGENT_TYPE
from infgen.utils.func import angle_between_2d_vectors, wrap_angle, weight_init


class HungarianMatcher(nn.Module):

    def __init__(self, loss_weight: dict, enter_state: int = 0):
        super().__init__()
        self.enter_state = enter_state
        self.cost_state = loss_weight['state_cls_loss']
        self.cost_pos = loss_weight['pos_cls_loss']
        self.cost_head = loss_weight['head_cls_loss']
        self.cost_shape = loss_weight['shape_reg_loss']
        self.seed_state_weight = loss_weight['seed_state_weight']
        self.seed_type_weight = loss_weight['seed_type_weight']

    @torch.no_grad()
    def forward(self, outputs, targets, ptr_pred, ptr_gt, valid_mask=None):

        pred_indices = []
        gt_indices = []

        for b in range(len(ptr_gt) - 1):

            start_pred, end_pred = ptr_pred[b], ptr_pred[b + 1]
            start_gt, end_gt = ptr_gt[b], ptr_gt[b + 1]

            pos_pred   = outputs['pos_pred'][start_pred : end_pred] # (n, s, l)
            shape_pred = outputs['shape_pred'][start_pred : end_pred]

            pos_gt   = targets['pos_gt'][start_gt : end_gt]
            shape_gt = targets['shape_gt'][start_gt : end_gt]

            num_pred = pos_pred.shape[0]
            num_gt = pos_gt.shape[0]

            cost_pos = F.cross_entropy(pos_pred[:, None].repeat(1, num_gt, 1, 1).reshape(-1, pos_pred.shape[-1]),
                                       pos_gt[None, ...].repeat(num_pred, 1, 1).reshape(-1),
                                       label_smoothing=0.1, ignore_index=-1, reduction='none'
                        ).reshape(num_pred, num_gt, -1)        
            cost_shape = ((shape_pred[:, None] - shape_gt[None, ...]) ** 2).sum(-1)

            C = (
                self.cost_pos * cost_pos
                + self.cost_shape * cost_shape
            )

            C = C.reshape(num_pred, num_gt, -1).cpu().numpy()

            if valid_mask is not None:
                # in case of seed size is smaller than the maximum number of gt among all steps
                C[:, ~valid_mask[start_gt : end_gt].cpu().numpy().astype(np.bool_)] = 1 << 15

            _indices = []
            for t in range(C.shape[-1]): # num_step
                _indices.append(linear_sum_assignment(C[..., t]))

            _indices = (
                torch.as_tensor(np.array([indices_t[0] for indices_t in _indices]) + int(start_pred), dtype=torch.long).transpose(-1, -2),
                torch.as_tensor(np.array([indices_t[1] for indices_t in _indices]) + int(start_gt), dtype=torch.long).transpose(-1, -2),
            )

            pred_indices.append(_indices[0])
            gt_indices.append(_indices[1])

        pred_indices = torch.cat(pred_indices)
        gt_indices = torch.cat(gt_indices)

        return pred_indices, gt_indices

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_pos: {}".format(self.cost_pos),
            "cost_head: {}".format(self.cost_head),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class InfGenAgentDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 pl2seed_radius: float,
                 a2a_radius: float,
                 a2sa_radius: float,
                 pl2sa_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 token_size: int,
                 attr_tokenizer: Attr_Tokenizer=None,
                 predict_motion: bool=False,
                 predict_state: bool=False,
                 predict_map: bool=False,
                 predict_occ: bool=False,
                 state_token: Dict[str, int]=None,
                 use_grid_token: bool=True,
                 use_head_token: bool=True,
                 use_state_token: bool=True,
                 disable_insertion: bool=False,
                 seed_size: int=5,
                 buffer_size: int=32,
                 num_recurrent_steps_val: int=-1,
                 loss_weight: dict=None,
                 logger=None) -> None:

        super(InfGenAgentDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.pl2seed_radius = pl2seed_radius
        self.a2a_radius = a2a_radius
        self.a2sa_radius = a2sa_radius
        self.pl2sa_radius = pl2sa_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.predict_motion = predict_motion
        self.predict_state = predict_state
        self.predict_map = predict_map
        self.predict_occ = predict_occ
        self.use_grid_token = use_grid_token
        self.use_head_token = use_head_token
        self.use_state_token = use_state_token
        self.disable_insertion = disable_insertion
        self.num_recurrent_steps_val = num_recurrent_steps_val
        self.loss_weight = loss_weight
        self.logger = logger

        self.attr_tokenizer = attr_tokenizer

        # state tokens
        self.state_type = list(state_token.keys())
        self.state_token = state_token
        self.invalid_state = int(state_token['invalid'])
        self.valid_state = int(state_token['valid'])
        self.enter_state = int(state_token['enter'])
        self.exit_state = int(state_token['exit'])

        self.seed_state_type = ['invalid', 'enter']
        self.valid_state_type = ['invalid', 'valid', 'exit']

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_pt2sa = 3
        input_dim_r_a2a = 3
        input_dim_r_a2sa = 3#4
        input_dim_motion_token = 8 # tokens: (token_size, 4, 2)
        input_dim_offset_token = 2

        self.seed_size = seed_size
        self.buffer_size = buffer_size

        # self.agent_type = ['veh', 'ped', 'cyc', 'seed']
        self.type_a_emb = nn.Embedding(len(AGENT_TYPE), hidden_dim)
        self.shape_emb = MLPEmbedding(input_dim=3, hidden_dim=hidden_dim)
        self.state_a_emb = nn.Embedding(len(self.state_type), hidden_dim)
        self.motion_gap = 1.
        self.heading_gap = 1.
        self.invalid_shape_value = .1
        self.invalid_motion_value = -2.
        self.invalid_head_value = -2.

        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2a_emb = FourierEmbedding(input_dim=input_dim_r_pt2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        # self.r_sa2sa_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim,
        #                                     num_freq_bands=num_freq_bands)
        self.r_pt2sa_emb = FourierEmbedding(input_dim=input_dim_r_pt2sa, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_a2sa_emb = FourierEmbedding(input_dim=input_dim_r_a2sa, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.token_emb_veh = MLPEmbedding(input_dim=input_dim_motion_token, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=input_dim_motion_token, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=input_dim_motion_token, hidden_dim=hidden_dim)
        self.token_emb_grid = MLPEmbedding(input_dim=input_dim_offset_token, hidden_dim=hidden_dim)
        self.no_token_emb = nn.Embedding(1, hidden_dim)
        self.bos_token_emb = nn.Embedding(1, hidden_dim)
        self.invalid_offset_token_emb = nn.Embedding(1, hidden_dim)

        if self.use_grid_token:
            num_inputs = 4
        else:
            num_inputs = 3
        self.fusion_emb = MLPEmbedding(input_dim=self.hidden_dim * num_inputs, hidden_dim=self.hidden_dim)

        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )

        self.seed_layers = 3
        self.pt2sa_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.seed_layers)]
        )
        self.a2sa_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(self.seed_layers)]
        )
        self.occ2sa_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=False) for _ in range(self.seed_layers)]
        )

        self.token_size = token_size # 2048
        # agent motion prediction head
        self.token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.token_size)
        # agent state prediction head
        self.state_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=len(self.valid_state_type))

        self.seed_state_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=len(self.seed_state_type))
        self.seed_type_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=len(AGENT_TYPE) - 1)
        self.seed_shape_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=3)

        self.grid_size = self.attr_tokenizer.grid_size
        self.angle_size = self.attr_tokenizer.angle_size

        if self.use_grid_token:
            self.seed_pos_rel_token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                            output_dim=self.grid_size)
            self.seed_offset_xy_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                        output_dim=2)
            self.seed_agent_occ_embed = MLPLayer(input_dim=self.grid_size, hidden_dim=hidden_dim,
                                                 output_dim=hidden_dim)
        else:
            self.seed_pos_rel_xy_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                         output_dim=2)
        if self.use_head_token:
            self.seed_heading_rel_token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                                output_dim=self.angle_size)
        else:
            self.seed_heading_rel_theta_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                                output_dim=1)

        if self.predict_occ:
            self.grid_agent_occ_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=self.grid_size)
            self.grid_pt_occ_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=self.grid_size)
        self.grid_index_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                        output_dim=self.grid_size)

        self.num_seed_feature = 10

        # self.trajectory_token = token_data['token'] # dict('veh', 'ped', 'cyc') (2048, 4, 2)
        # self.trajectory_token_traj = token_data['traj'] # (2048, 6, 3)
        # self.trajectory_token_all = token_data['token_all'] # (2048, 6, 4, 2)
        self.apply(weight_init)

        self.shift = 5
        self.motion_beam_size = 5
        self.insert_beam_size = 10
        self.hist_mask = True
        self.temporal_attn_to_invalid = False
        self.use_rel = False
        self.inference_filter_overlap = True
        assert self.num_recurrent_steps_val % self.shift == 0 or self.num_recurrent_steps_val == -1, \
                            f"Invalid num_recurrent_steps_val: {num_recurrent_steps_val}."

        # seed agent
        self.temporal_attn_seed = False
        self.seed_attn_to_av = True
        self.seed_use_ego_motion = False

        self.matcher = HungarianMatcher(loss_weight=loss_weight, enter_state=self.enter_state)

    def transform_rel(self, token_traj, prev_pos, prev_heading=None):
        if prev_heading is None:
            diff_xy = prev_pos[:, :, -1, :] - prev_pos[:, :, -2, :]
            prev_heading = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

        num_agent, num_step, traj_num, traj_dim = token_traj.shape
        cos, sin = prev_heading.cos(), prev_heading.sin()
        rot_mat = torch.zeros((num_agent, num_step, 2, 2), device=prev_heading.device)
        rot_mat[:, :, 0, 0] = cos
        rot_mat[:, :, 0, 1] = -sin
        rot_mat[:, :, 1, 0] = sin
        rot_mat[:, :, 1, 1] = cos
        agent_diff_rel = torch.bmm(token_traj.view(-1, traj_num, 2), rot_mat.view(-1, 2, 2)).view(num_agent, num_step, traj_num, traj_dim)
        agent_pred_rel = agent_diff_rel + prev_pos[:, :, -1:, :]
        return agent_pred_rel

    def _agent_token_embedding(self, data, agent_token_index, agent_state, agent_offset_token_idx, pos_a, head_a,
                               inference=False, filter_mask=None, av_index=None):

        if filter_mask is None:
            filter_mask = torch.ones_like(agent_state[:, 2], dtype=torch.bool)

        num_agent, num_step, traj_dim = pos_a.shape # traj_dim=2
        agent_type = data['agent']['type'][filter_mask]
        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2

        motion_vector_a, head_vector_a = self._build_vector_a(pos_a, head_a, agent_state)


        trajectory_token_veh = data['agent']['trajectory_token_veh']  # [n_token, 6, 4, 2]
        trajectory_token_ped = data['agent']['trajectory_token_ped']
        trajectory_token_cyc = data['agent']['trajectory_token_cyc']
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh[:, -1].flatten(1, 2))
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped[:, -1].flatten(1, 2))
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc[:, -1].flatten(1, 2))

        # add bos token embedding
        agent_token_emb_veh = torch.cat([agent_token_emb_veh, self.bos_token_emb(torch.zeros(1, device=pos_a.device).long())])
        agent_token_emb_ped = torch.cat([agent_token_emb_ped, self.bos_token_emb(torch.zeros(1, device=pos_a.device).long())])
        agent_token_emb_cyc = torch.cat([agent_token_emb_cyc, self.bos_token_emb(torch.zeros(1, device=pos_a.device).long())])

        # add invalid token embedding
        agent_token_emb_veh = torch.cat([agent_token_emb_veh, self.no_token_emb(torch.zeros(1, device=pos_a.device).long())])
        agent_token_emb_ped = torch.cat([agent_token_emb_ped, self.no_token_emb(torch.zeros(1, device=pos_a.device).long())])
        agent_token_emb_cyc = torch.cat([agent_token_emb_cyc, self.no_token_emb(torch.zeros(1, device=pos_a.device).long())])

        # additional token embeddings are already added -> -1: invalid, -2: bos
        agent_token_emb = torch.zeros((num_agent, num_step, self.hidden_dim), device=pos_a.device)
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        # grid embedding
        self.grid_token_emb = self.token_emb_grid(self.attr_tokenizer.grid)
        self.grid_token_emb = torch.cat([self.grid_token_emb, self.invalid_offset_token_emb(torch.zeros(1, device=pos_a.device).long())])
        offset_token_emb = self.grid_token_emb[agent_offset_token_idx]

        # 'vehicle', 'pedestrian', 'cyclist', 'background'
        is_invalid = agent_state == self.invalid_state
        agent_types = data['agent']['type'].clone()[filter_mask].long().repeat_interleave(repeats=num_step, dim=0)
        agent_types[is_invalid.reshape(-1)] = AGENT_TYPE.index('seed')
        agent_shapes = data['agent']['shape'].clone()[filter_mask, self.num_historical_steps - 1, :].repeat_interleave(repeats=num_step, dim=0)
        agent_shapes[is_invalid.reshape(-1)] = self.invalid_shape_value

        # TODO: fix ego_pos in inference mode
        offset_pos = pos_a - pos_a[av_index].repeat_interleave(repeats=data['batch_size_a'], dim=0)
        feat_a, categorical_embs = self._build_agent_feature(num_step, pos_a.device,
                                                             motion_vector_a,
                                                             head_vector_a,
                                                             agent_token_emb,
                                                             offset_token_emb,
                                                             offset_pos=offset_pos,
                                                             type=agent_types,
                                                             shape=agent_shapes,
                                                             state=agent_state,
                                                             n=num_agent)

        if inference:
            return (
                feat_a,
                agent_token_emb,
                agent_token_emb_veh,
                agent_token_emb_ped,
                agent_token_emb_cyc,
                categorical_embs,
                trajectory_token_veh,
                trajectory_token_ped,
                trajectory_token_cyc,
            )

        else:

            # seed agent feature
            if self.seed_use_ego_motion:
                motion_vector_seed = motion_vector_a[av_index].repeat_interleave(repeats=self.num_seed_feature, dim=0)
                head_vector_seed = head_vector_a[av_index].repeat_interleave(repeats=self.num_seed_feature, dim=0)
            else:
                motion_vector_seed = head_vector_seed = None
            feat_seed, _ = self._build_agent_feature(num_step, pos_a.device,
                                                     motion_vector_seed,
                                                     head_vector_seed,
                                                     state_index=self.invalid_state,
                                                     n=data.num_graphs * self.num_seed_feature)

            feat_a = torch.cat([feat_a, feat_seed], dim=0) # (a + s, t, d)

            return feat_a

    def _build_vector_a(self, pos_a, head_a, state_a):
        num_agent = pos_a.shape[0]

        motion_vector_a = torch.cat([pos_a.new_zeros(num_agent, 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

        motion_vector_a[state_a == self.invalid_state] = self.invalid_motion_value

        # invalid -> valid
        is_last_invalid = (state_a.roll(shifts=1, dims=1) == self.invalid_state) & (state_a != self.invalid_state)
        is_last_invalid[:, 0] = state_a[:, 0] == self.enter_state
        motion_vector_a[is_last_invalid] = self.motion_gap

        # valid -> invalid
        is_last_valid = (state_a.roll(shifts=1, dims=1) != self.invalid_state) & (state_a == self.invalid_state)
        is_last_valid[:, 0] = False
        motion_vector_a[is_last_valid] = -self.motion_gap

        head_a[state_a == self.invalid_state] == self.invalid_head_value
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        return motion_vector_a, head_vector_a

    def _build_agent_feature(self, num_step, device,
                             motion_vector=None,
                             head_vector=None,
                             agent_token_emb=None,
                             agent_grid_emb=None,
                             offset_pos=None,
                             type=None,
                             shape=None,
                             categorical_embs_a=None,
                             state=None,
                             state_index=None,
                             n=1):

        if agent_token_emb is None:
            agent_token_emb = self.no_token_emb(torch.zeros(1, device=device).long())[:, None].repeat(n, num_step, 1)
            if state is not None:
                agent_token_emb[state == self.enter_state] = self.bos_token_emb(torch.zeros(1, device=device).long())

        if agent_grid_emb is None:
            agent_grid_emb = self.grid_token_emb[None, None, self.grid_size // 2].repeat(n, num_step, 1)

        if motion_vector is None or head_vector is None:
            pos_a = torch.zeros((n, num_step, 2), device=device)
            head_a = torch.zeros((n, num_step), device=device)
            if state is None:
                state = torch.full((n, num_step), self.invalid_state, device=device)
            motion_vector, head_vector = self._build_vector_a(pos_a, head_a, state)

        if offset_pos is None:
            offset_pos = torch.zeros_like(motion_vector)

        feature_a = torch.stack(
            [torch.norm(motion_vector[:, :, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector, nbr_vector=motion_vector[:, :, :2]),
             # torch.norm(offset_pos[:, :, :2], p=2, dim=-1),
             ], dim=-1)

        if categorical_embs_a is None:
            if type is None:
                type = torch.tensor([AGENT_TYPE.index('seed')], device=device)
            if shape is None:
                shape = torch.full((1, 3), self.invalid_shape_value, device=device)

            categorical_embs_a = [self.type_a_emb(type.reshape(-1)), self.shape_emb(shape.reshape(-1, shape.shape[-1]))]

        x_a = self.x_a_emb(continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
                           categorical_embs=categorical_embs_a)
        x_a = x_a.view(-1, num_step, self.hidden_dim) # (a, t, d)

        if state is None:
            assert state_index is not None, f"state index need to be set when state tensor is None!"
            state = torch.tensor([state_index], device=device)[:, None].repeat(n, num_step, 1) # do not use `expand`
        s_a = self.state_a_emb(state.reshape(-1).long()).reshape(n, num_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a, s_a), dim=-1)
        if self.use_grid_token:
            feat_a = torch.cat([feat_a, agent_grid_emb], dim=-1)

        feat_a = self.fusion_emb(feat_a) # (a, t, d)

        return feat_a, categorical_embs_a

    def _pad_feat(self, num_graph, av_index, *feats, num_seed_feature=None):

        if num_seed_feature is None:
            num_seed_feature = self.num_seed_feature

        padded_feats = tuple()
        for i in range(len(feats)):
            padded_feats += (torch.cat([feats[i], feats[i][av_index].repeat_interleave(
                repeats=num_seed_feature, dim=0)],
                dim=0
            ),)

        pad_mask = torch.ones(*padded_feats[0].shape[:2], device=feats[0].device).bool() # (a, t)
        pad_mask[-num_graph * num_seed_feature:] = False

        return padded_feats + (pad_mask,)

    # def _build_seed_feat(self, data, pos_a, head_a, state_a, head_vector_a, mask, sort_indices, av_index):
    #     seed_mask = sort_indices != av_index.repeat_interleave(repeats=data['batch_size_a'], dim=0)[:, None]
    #     print(mask.shape, sort_indices.shape, seed_mask.shape)
    #     mask[-data.num_graphs * self.num_seed_feature:] = seed_mask[:self.num_seed_feature]

    #     insert_pos_a = torch.gather(pos_a, dim=0, index=sort_indices[:self.num_seed_feature, :, None].expand(-1, -1, pos_a.shape[-1]))
    #     pos_a[mask] = insert_pos_a[mask[-self.num_seed_feature:]]

    #     state_a[-data.num_graphs * self.num_seed_feature:] = self.enter_state

    #     return pos_a, head_a, state_a, head_vector_a, mask

    def _build_temporal_edge(self, data, pos_a, head_a, state_a, head_vector_a, mask, inference_mask=None):

        num_graph = data.num_graphs
        num_agent = pos_a.shape[0]
        hist_mask = mask.clone()

        if not self.temporal_attn_to_invalid:
            is_bos = state_a == self.enter_state
            bos_index = torch.where(is_bos.any(dim=1), torch.argmax(is_bos.long(), dim=1), torch.tensor(0))
            history_invalid_mask = torch.arange(mask.shape[1]).expand(mask.shape[0], mask.shape[1]).to(mask.device)
            history_invalid_mask = (history_invalid_mask < bos_index[:, None])
            hist_mask[history_invalid_mask] = False

        if not self.temporal_attn_seed:
            hist_mask[-num_graph * self.num_seed_feature:] = False
            if inference_mask is not None:
                inference_mask[-num_graph * self.num_seed_feature:] = False
        else:
            # WARNING: if use temporal attn to seed
            # we need to fix the pos/head of seed!!!
            raise RuntimeError("Wrong settings!")

        pos_t = pos_a.reshape(-1, self.input_dim) # (num_agent * num_step, ...)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)

        # for those invalid agents won't predict any motion token, we don't attend to them
        is_bos = state_a == self.enter_state
        is_bos[-num_graph * self.num_seed_feature:] = False
        bos_index = torch.where(is_bos.any(dim=1), torch.argmax(is_bos.long(), dim=1), torch.tensor(0))
        motion_predict_start_index = torch.clamp(bos_index - self.time_span / self.shift + 1, min=0)
        motion_predict_mask = torch.arange(hist_mask.shape[1]).expand(hist_mask.shape[0], -1).to(hist_mask.device)
        motion_predict_mask = motion_predict_mask >= motion_predict_start_index[:, None]
        hist_mask[~motion_predict_mask] = False

        if self.hist_mask and self.training:
            hist_mask[
                torch.arange(mask.shape[0]).unsqueeze(1), torch.randint(0, mask.shape[1], (num_agent, 10))] = False
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)
        elif inference_mask is not None:
            mask_t = hist_mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)

        # mask_t: (num_agent, 18, 18), edge_index_t: (2, num_edge)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, (edge_index_t[1] - edge_index_t[0] > 0) &
                                       (edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift)]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])

        # handle the invalid steps
        is_invalid = state_a == self.invalid_state
        is_invalid_t = is_invalid.reshape(-1)

        rel_pos_t[is_invalid_t[edge_index_t[0]] & ~is_invalid_t[edge_index_t[1]]] = -self.motion_gap
        rel_pos_t[~is_invalid_t[edge_index_t[0]] & is_invalid_t[edge_index_t[1]]] = self.motion_gap
        rel_head_t[is_invalid_t[edge_index_t[0]] & ~is_invalid_t[edge_index_t[1]]] = -self.heading_gap
        rel_head_t[~is_invalid_t[edge_index_t[1]] & is_invalid_t[edge_index_t[1]]] = self.heading_gap

        rel_pos_t[is_invalid_t[edge_index_t[0]] & is_invalid_t[edge_index_t[1]]] = self.invalid_motion_value
        rel_head_t[is_invalid_t[edge_index_t[0]] & is_invalid_t[edge_index_t[1]]] = self.invalid_head_value

        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        return edge_index_t, r_t

    def _build_interaction_edge(self, data, pos_a, head_a, state_a, head_vector_a, batch_s, mask, pad_mask=None, inference_mask=None,
                                av_index=None, seq_mask=None, seq_index=None, grid_index_a=None, **plot_kwargs):
        num_graph = data.num_graphs
        num_agent, num_step, _ = pos_a.shape
        is_training = inference_mask is None

        mask_a = mask.clone()

        if pad_mask is None:
            pad_mask = torch.ones_like(state_a).bool()

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pad_mask_s = pad_mask.transpose(0, 1).reshape(-1)
        if inference_mask is not None:
            mask_a = mask_a & inference_mask
        mask_s = mask_a.transpose(0, 1).reshape(-1)

        # build agent2agent bilateral connection
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s & pad_mask_s, edge_index=edge_index_a2a)[0]

        if int(os.getenv('PLOT_EDGE', 0)):
            plot_interact_edge(edge_index_a2a, data['scenario_id'], data['batch_size_a'].cpu(), self.num_seed_feature, num_step,
                               av_index=av_index, **plot_kwargs)

        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])

        # handle the invalid steps
        is_invalid = state_a == self.invalid_state
        is_invalid_s = is_invalid.transpose(0, 1).reshape(-1)

        rel_pos_a2a[is_invalid_s[edge_index_a2a[0]] & ~is_invalid_s[edge_index_a2a[1]]] = -self.motion_gap
        rel_pos_a2a[~is_invalid_s[edge_index_a2a[0]] & is_invalid_s[edge_index_a2a[1]]] = self.motion_gap
        rel_head_a2a[is_invalid_s[edge_index_a2a[0]] & ~is_invalid_s[edge_index_a2a[1]]] = -self.heading_gap
        rel_head_a2a[~is_invalid_s[edge_index_a2a[1]] & is_invalid_s[edge_index_a2a[1]]] = self.heading_gap

        rel_pos_a2a[is_invalid_s[edge_index_a2a[0]] & is_invalid_s[edge_index_a2a[1]]] = self.invalid_motion_value
        rel_head_a2a[is_invalid_s[edge_index_a2a[0]] & is_invalid_s[edge_index_a2a[1]]] = self.invalid_head_value

        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        # add the edges which connect seed agents
        if is_training:
            mask_av = torch.ones_like(mask_a).bool()
            if not self.seed_attn_to_av:
                mask_av[av_index] = False
            mask_a &= mask_av
            edge_index_seed2a, r_seed2a = self._build_a2sa_edge(data, pos_a, head_a, head_vector_a, batch_s,
                                                                mask_a.clone(), ~pad_mask.clone(), inference_mask=inference_mask,
                                                                r=self.pl2seed_radius, max_num_neighbors=300,
                                                                seq_mask=seq_mask, seq_index=seq_index, grid_index_a=grid_index_a, mode='insert')

            if os.getenv('PLOT_EDGE', False):
                plot_interact_edge(edge_index_seed2a, data['scenario_id'], data['batch_size_a'].cpu(), self.num_seed_feature, num_step,
                                   'interact_edge_map_seed', av_index=av_index, **plot_kwargs)

            edge_index_a2a = torch.cat([edge_index_a2a, edge_index_seed2a], dim=-1)
            r_a2a = torch.cat([r_a2a, r_seed2a])

            return edge_index_a2a, r_a2a, (edge_index_a2a.shape[1], edge_index_seed2a.shape[1])

        return edge_index_a2a, r_a2a

    def _build_map2agent_edge(self, data, pos_a, head_a, state_a, head_vector_a, batch_s, batch_pl,
                              mask, pad_mask=None, inference_mask=None, av_index=None, **kwargs):
        num_graph = data.num_graphs
        num_agent, num_step, _ = pos_a.shape
        is_training = inference_mask is None

        mask_pl2a = mask.clone()

        if pad_mask is None:
            pad_mask = torch.ones_like(state_a).bool()

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pad_mask_s = pad_mask.transpose(0, 1).reshape(-1)
        if inference_mask is not None:
            mask_pl2a = mask_pl2a & inference_mask
        mask_pl2a = mask_pl2a.transpose(0, 1).reshape(-1)

        ori_pos_pl = data['pt_token']['position'][:, :self.input_dim].contiguous()
        ori_orient_pl = data['pt_token']['orientation'].contiguous()
        pos_pl = ori_pos_pl.repeat(num_step, 1) # not `repeat_interleave`
        orient_pl = ori_orient_pl.repeat(num_step)

        # build map2agent directed graph
        # edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius,
        #                          batch_x=batch_s, batch_y=batch_pl, max_num_neighbors=300)
        edge_index_pl2a = radius(x=pos_pl[:, :2], y=pos_s[:, :2], r=self.pl2a_radius,
                                 batch_x=batch_pl, batch_y=batch_s, max_num_neighbors=5)
        edge_index_pl2a = edge_index_pl2a[[1, 0]]
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]] &
                                             pad_mask_s[edge_index_pl2a[1]]]

        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])

        # handle the invalid steps
        is_invalid = state_a == self.invalid_state
        is_invalid_s = is_invalid.transpose(0, 1).reshape(-1)
        rel_pos_pl2a[is_invalid_s[edge_index_pl2a[1]]] = self.motion_gap
        rel_orient_pl2a[is_invalid_s[edge_index_pl2a[1]]] = self.heading_gap

        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        # add the edges which connect seed agents
        if is_training:
            edge_index_pl2seed, r_pl2seed = self._build_map2sa_edge(data, pos_a, head_a, head_vector_a, batch_s, batch_pl,
                                                                    ~pad_mask.clone(), inference_mask=inference_mask,
                                                                    r=self.pl2seed_radius, max_num_neighbors=2048, mode='insert')

            # ! sanity check
            # pl2a_index = torch.zeros(pos_a.shape[0], num_step)
            # pl2a_r = torch.zeros(pos_a.shape[0], num_step)
            # for src_index in torch.unique(edge_index_pl2seed[1]):
            #     src_row = src_index % pos_a.shape[0]
            #     src_col = src_index // pos_a.shape[0]
            #     pl2a_index[src_row, src_col] = edge_index_pl2seed[0, edge_index_pl2seed[1] == src_index].sum()
            #     pl2a_r[src_row, src_col] = r_pl2seed[edge_index_pl2seed[1] == src_index].sum()
            # print(pl2a_index)
            # print(pl2a_r)
            # exit(1)

            if os.getenv('PLOT_EDGE', False):
                plot_interact_edge(edge_index_pl2seed, data['scenario_id'], data['batch_size_a'].cpu(), self.num_seed_feature, num_step,
                                'interact_edge_map_seed', av_index=av_index)

            edge_index_pl2a = torch.cat([edge_index_pl2a, edge_index_pl2seed], dim=-1)
            r_pl2a = torch.cat([r_pl2a, r_pl2seed])

            return edge_index_pl2a, r_pl2a, (edge_index_pl2a.shape[1], edge_index_pl2seed.shape[1])

        return edge_index_pl2a, r_pl2a

    def _build_a2sa_edge(self, data, pos_a, head_a, head_vector_a, batch_s, mask_a, mask_sa,
                         inference_mask=None, r=None, max_num_neighbors=8, seq_mask=None, seq_index=None,
                         grid_index_a=None, mode: Literal['insert', 'feature']='feature', **plot_kwargs):

        num_agent, num_step, _ = pos_a.shape
        is_training = inference_mask is None

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        if inference_mask is not None:
            mask_a = mask_a & inference_mask
            mask_sa = mask_sa & inference_mask
        mask_s = mask_a.transpose(0, 1).reshape(-1)
        mask_s_sa = mask_sa.transpose(0, 1).reshape(-1)

        # build seed_agent2agent unilateral connection
        assert r is not None, "r needs to be specified!"
        # edge_index_a2sa = radius(x=pos_s[mask_s_sa, :2], y=pos_s[:, :2], r=r,
        #                          batch_x=batch_s[mask_s_sa], batch_y=batch_s, max_num_neighbors=max_num_neighbors)
        edge_index_a2sa = radius(x=pos_s[:, :2], y=pos_s[mask_s_sa, :2], r=r,
                                 batch_x=batch_s, batch_y=batch_s[mask_s_sa], max_num_neighbors=max_num_neighbors)
        edge_index_a2sa = edge_index_a2sa[[1, 0]]
        edge_index_a2sa = edge_index_a2sa[:, ~mask_s_sa[edge_index_a2sa[0]] & mask_s[edge_index_a2sa[0]]]

        # only for seed agent sequence training
        if seq_mask is not None:
            edge_mask = seq_mask[edge_index_a2sa[1]]
            edge_mask = torch.gather(edge_mask, dim=1, index=edge_index_a2sa[0, :, None] % num_agent)[:, 0]
            edge_index_a2sa = edge_index_a2sa[:, edge_mask]

        if seq_index is None:
            seq_index = torch.zeros(num_agent, device=pos_a.device).long()
        if seq_index.dim() == 1:
            seq_index = seq_index[:, None].repeat(1, num_step)
        seq_index = seq_index.transpose(0, 1).reshape(-1)
        assert seq_index.shape[0] == pos_s.shape[0], f"Inconsistent lenght {seq_index.shape[0]} and {pos_s.shape[0]}!"

        # convert to global index
        all_index = torch.arange(pos_s.shape[0], device=pos_a.device).long()
        sa_index = all_index[mask_s_sa]
        edge_index_a2sa[1] = sa_index[edge_index_a2sa[1]]

        # plot edge index TODO: now only support bs=1
        if os.getenv('PLOT_EDGE_INFERENCE', False) and not is_training:
            num_agent, num_step, _ = pos_a.shape
            # plot_interact_edge(edge_index_a2sa, data['scenario_id'], data['batch_size_a'].cpu(), 1, num_step,
            #                    'interact_a2sa_edge_map', **plot_kwargs)
            plot_interact_edge(edge_index_a2sa, data['scenario_id'], torch.tensor([num_agent - 1]), 1, num_step,
                               f"interact_a2sa_edge_map_infer_{plot_kwargs['tag']}", **plot_kwargs)

        rel_pos_a2sa = pos_s[edge_index_a2sa[0]] - pos_s[edge_index_a2sa[1]]
        rel_head_a2sa = wrap_angle(head_s[edge_index_a2sa[0]] - head_s[edge_index_a2sa[1]])

        if mode == 'insert':

            # assert grid_index_a is not None, f"Missing input: grid_index_a!"
            # grid_index_s = grid_index_a.transpose(0, 1).reshape(-1)
            # assert grid_index_s[edge_index_a2sa[0]].min() >= 0, "Found invalid values in grid index"

            # r_a2sa = torch.stack(
            #     [self.attr_tokenizer.dist[grid_index_s[edge_index_a2sa[0]]],
            #     self.attr_tokenizer.dir[grid_index_s[edge_index_a2sa[0]]],
            #     rel_head_a2sa,
            #     seq_index[edge_index_a2sa[0]] - seq_index[edge_index_a2sa[1]]], dim=-1)

            # r_a2sa = torch.stack(
            #     [torch.norm(rel_pos_a2sa[:, :2], p=2, dim=-1),
            #     angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2sa[1]], nbr_vector=rel_pos_a2sa[:, :2]),
            #     rel_head_a2sa,
            #     seq_index[edge_index_a2sa[0]] - seq_index[edge_index_a2sa[1]]], dim=-1)
            r_a2sa = torch.stack(
                [torch.norm(rel_pos_a2sa[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2sa[1]], nbr_vector=rel_pos_a2sa[:, :2]),
                rel_head_a2sa], dim=-1)
            # TODO: try categorical embs
            r_a2sa = self.r_a2sa_emb(continuous_inputs=r_a2sa, categorical_embs=None)

        elif mode == 'feature':

            r_a2sa = torch.stack(
                [torch.norm(rel_pos_a2sa[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2sa[1]], nbr_vector=rel_pos_a2sa[:, :2]),
                rel_head_a2sa], dim=-1)
            r_a2sa = self.r_a2a_emb(continuous_inputs=r_a2sa, categorical_embs=None)

        else:
            raise ValueError(f"Unsupport mode {mode}.")

        return edge_index_a2sa, r_a2sa

    def _build_map2sa_edge(self, data, pos_a, head_a, head_vector_a, batch_s, batch_pl,
                           mask_sa, inference_mask=None, r=None, max_num_neighbors=32, mode: Literal['insert', 'feature']='feature'):

        _, num_step, _ = pos_a.shape

        mask_pl2sa = torch.ones_like(mask_sa).bool()
        if inference_mask is not None:
            mask_pl2sa = mask_pl2sa & inference_mask
        mask_pl2sa = mask_pl2sa.transpose(0, 1).reshape(-1)
        mask_s_sa = mask_sa.transpose(0, 1).reshape(-1)

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        ori_pos_pl = data['pt_token']['position'][:, :self.input_dim].contiguous()
        ori_orient_pl = data['pt_token']['orientation'].contiguous()
        pos_pl = ori_pos_pl.repeat(num_step, 1) # not `repeat_interleave`
        orient_pl = ori_orient_pl.repeat(num_step)

        # build map2agent directed graph
        assert r is not None, "r needs to be specified!"
        # edge_index_pl2sa = radius(x=pos_s[mask_s_sa, :2], y=pos_pl[:, :2], r=r,
        #                           batch_x=batch_s[mask_s_sa], batch_y=batch_pl, max_num_neighbors=max_num_neighbors)
        edge_index_pl2sa = radius(x=pos_pl[:, :2], y=pos_s[mask_s_sa, :2], r=r,
                                  batch_x=batch_pl, batch_y=batch_s[mask_s_sa], max_num_neighbors=max_num_neighbors)
        edge_index_pl2sa = edge_index_pl2sa[[1, 0]]
        edge_index_pl2sa = edge_index_pl2sa[:, mask_pl2sa[mask_s_sa][edge_index_pl2sa[1]]]

        # convert to global index
        all_index = torch.arange(pos_s.shape[0], device=pos_a.device).long()
        sa_index = all_index[mask_s_sa]
        edge_index_pl2sa[1] = sa_index[edge_index_pl2sa[1]]

        # ! plot edge map
        # if os.getenv('PLOT_EDGE', False):
        #     plot_map_edge(edge_index_pl2sa, pos_s[:, :2], data, save_path='map2sa_edge_map')

        rel_pos_pl2sa = pos_pl[edge_index_pl2sa[0]] - pos_s[edge_index_pl2sa[1]]
        rel_orient_pl2sa = wrap_angle(orient_pl[edge_index_pl2sa[0]] - head_s[edge_index_pl2sa[1]])

        r_pl2sa = torch.stack(
            [torch.norm(rel_pos_pl2sa[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2sa[1]], nbr_vector=rel_pos_pl2sa[:, :2]),
            rel_orient_pl2sa], dim=-1)

        if mode == 'insert':
            r_pl2sa = self.r_pt2sa_emb(continuous_inputs=r_pl2sa, categorical_embs=None)
        elif mode == 'feature':
            r_pl2sa = self.r_pt2a_emb(continuous_inputs=r_pl2sa, categorical_embs=None)
        else:
            raise ValueError(f"Unsupport mode {mode}.")

        return edge_index_pl2sa, r_pl2sa

    # def _build_sa2sa_edge(self, data, pos_a, head_a, state_a, head_vector_a, batch_s, mask, inference_mask=None, **plot_kwargs):

    #     num_agent = pos_a.shape[0]

    #     pos_t = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
    #     head_t = head_a.reshape(-1)
    #     head_vector_t = head_vector_a.reshape(-1, 2)

    #     if inference_mask is not None:
    #         mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
    #     else:
    #         mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

    #     edge_index_sa2sa = dense_to_sparse(mask_t)[0]
    #     edge_index_sa2sa = edge_index_sa2sa[:, edge_index_sa2sa[1] - edge_index_sa2sa[0] > 0]
    #     rel_pos_t = pos_t[edge_index_sa2sa[0]] - pos_t[edge_index_sa2sa[1]]
    #     rel_head_t = wrap_angle(head_t[edge_index_sa2sa[0]] - head_t[edge_index_sa2sa[1]])

    #     r_t = torch.stack(
    #         [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
    #          angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_sa2sa[1]], nbr_vector=rel_pos_t[:, :2]),
    #          rel_head_t,
    #          edge_index_sa2sa[0] - edge_index_sa2sa[1]], dim=-1)
    #     r_sa2sa = self.r_sa2sa_emb(continuous_inputs=r_t, categorical_embs=None)

    #     return edge_index_sa2sa, r_sa2sa

    def get_inputs(self, data: HeteroData) -> Dict[str, torch.Tensor]:

        pos_a = data['agent']['token_pos'].clone()
        head_a = data['agent']['token_heading'].clone()
        agent_token_index = data['agent']['token_idx'].clone()
        agent_state_index = data['agent']['state_idx'].clone()
        mask = data['agent']['raw_agent_valid_mask'].clone()

        agent_grid_token_idx = data['agent']['grid_token_idx']
        agent_grid_offset_xy = data['agent']['grid_offset_xy']
        agent_head_token_idx = data['agent']['heading_token_idx']
        sort_indices = data['agent']['sort_indices']

        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)
        next_state_index_gt = agent_token_index.roll(shifts=-1, dims=1)

        # next token prediction mask
        bos_token_index = torch.nonzero(agent_state_index == self.enter_state)
        eos_token_index = torch.nonzero(agent_state_index == self.exit_state)

        # mask for motion tokens
        next_token_eval_mask = mask.clone()
        next_token_eval_mask = next_token_eval_mask * next_token_eval_mask.roll(shifts=-1, dims=1) * next_token_eval_mask.roll(shifts=1, dims=1)
        for bos_token_index_ in bos_token_index:
            next_token_eval_mask[bos_token_index_[0], bos_token_index_[1] : bos_token_index_[1] + 1] = 1
            next_token_eval_mask[bos_token_index_[0], bos_token_index_[1] + 1 : bos_token_index_[1] + 2] = \
                            mask[bos_token_index_[0], bos_token_index_[1] + 2 : bos_token_index_[1] + 3]
        next_token_eval_mask[eos_token_index[:, 0], eos_token_index[:, 1]] = 0

        # mask for state tokens
        next_state_eval_mask = mask.clone()
        next_state_eval_mask = next_state_eval_mask * next_state_eval_mask.roll(shifts=-1, dims=1) * next_state_eval_mask.roll(shifts=1, dims=1)
        for bos_token_index_ in bos_token_index:
            next_state_eval_mask[bos_token_index_[0], :bos_token_index_[1]] = 0
            next_state_eval_mask[bos_token_index_[0], bos_token_index_[1] : bos_token_index_[1] + 1] = 1
            next_state_eval_mask[bos_token_index_[0], bos_token_index_[1] + 1 : bos_token_index_[1] + 2] = \
                            mask[bos_token_index_[0], bos_token_index_[1] + 2 : bos_token_index_[1] + 3]
        for eos_token_index_ in eos_token_index:
            next_state_eval_mask[eos_token_index_[0], eos_token_index_[1] + 1:] = 1
            next_state_eval_mask[eos_token_index_[0], eos_token_index_[1] : eos_token_index_[1] + 1] = \
                            mask[eos_token_index_[0], eos_token_index_[1] - 1 : eos_token_index_[1]]

        # the last timestep is the beginning of the sequence (also the input)
        next_token_eval_mask[:, 0] = mask[:, 0] * mask[:, 1]
        next_state_eval_mask[:, 0] = mask[:, 0] * mask[:, 1]
        next_token_eval_mask[:, -1] = 0
        next_state_eval_mask[:, -1] = 0

        if next_token_index_gt[next_token_eval_mask].min() < 0:
            raise RuntimeError()

        return {'token_pos': pos_a,
                'token_heading': head_a,
                'next_token_idx_gt': next_token_index_gt,
                'next_state_idx_gt': next_state_index_gt,
                'next_token_eval_mask': next_token_eval_mask,
                'raw_agent_valid_mask': data['agent']['raw_agent_valid_mask'],
                'state_token': agent_state_index,
                'grid_index': agent_grid_token_idx,
                }

    def _build_seq(self, device, data, num_agent, num_step, av_index, sort_indices):
        """
        Args:
            sort_indices (torch.Tensor): shape (num_agent, num_atep)
        """
        ptr = data['agent']['ptr']
        num_graph = len(ptr) - 1

        # sort_indices = sort_indices[:self.num_seed_feature]
        seq_mask = torch.ones(num_graph * self.num_seed_feature, num_step, num_agent + num_graph * self.num_seed_feature, device=device).bool()
        seq_mask[..., -num_graph * self.num_seed_feature:] = False
        for b in range(num_graph):
            batch_sort_indices = sort_indices[ptr[b] : ptr[b + 1]]
            for t in range(num_step):
                for s in range(self.num_seed_feature):
                    seq_mask[b * self.num_seed_feature + s, t, batch_sort_indices[s:, t].flatten().long()] = False
        if self.seed_attn_to_av:
            seq_mask[..., av_index] = True
        seq_mask = seq_mask.transpose(0, 1).reshape(-1, num_agent + num_graph * self.num_seed_feature)

        seq_index = torch.cat([torch.zeros(num_agent), (torch.arange(self.num_seed_feature) + 1).repeat(num_graph)]).to(device)
        seq_index = seq_index[:, None].repeat(1, num_step)
        # 0, 0, 0, ..., 1, 2, 3, ...
        for b in range(num_graph):
            batch_sort_indices = sort_indices[ptr[b] : ptr[b + 1]]
            for t in range(num_step):
                for s in range(self.num_seed_feature):
                    seq_index[batch_sort_indices[s : s + 1, t].flatten().long() + ptr[b], t] = s + 1

        # 0, 2, 1, ..., N+1, N+2, ...
        # for b in range(num_graph):
        #     batch_sort_indices = sort_indices[ptr[b] : ptr[b + 1]]
        #     batch_agent_valid_mask = data['agent']['inrange_mask'][ptr[b] : ptr[b + 1]] & \
        #                              data['agent']['raw_agent_valid_mask'][ptr[b] : ptr[b + 1]] & \
        #                             ~data['agent']['bos_mask'][ptr[b] : ptr[b + 1]]
        #     batch_agent_valid_mask[av_index[b]] = False
        #     for t in range(num_step):
        #         batch_num_valid_agent_t = batch_agent_valid_mask[:, t].sum()
        #         seq_index[num_agent + b * self.num_seed_feature : num_agent + (b + 1) * self.num_seed_feature, t] += batch_num_valid_agent_t
        #         random_seq_index = torch.zeros(ptr[b + 1] - ptr[b], device=device)
        #         random_seq_index[batch_agent_valid_mask[:, t]] = torch.randperm(batch_num_valid_agent_t, device=device).float() + 1  # starts from 1
        #         seq_index[ptr[b] : ptr[b + 1], t] = random_seq_index
        #         for s in range(self.num_seed_feature):
        #             seq_index[batch_sort_indices[s : s + 1, t].flatten().long() + ptr[b], t] = s + 1 + batch_num_valid_agent_t.float()

        # 0, 0, 0, ..., N+1, N+2, ...
        # for b in range(num_graph):
        #     batch_sort_indices = sort_indices[ptr[b] : ptr[b + 1]]
        #     batch_agent_valid_mask = data['agent']['inrange_mask'][ptr[b] : ptr[b + 1]] & \
        #                              data['agent']['raw_agent_valid_mask'][ptr[b] : ptr[b + 1]] & \
        #                             ~data['agent']['bos_mask'][ptr[b] : ptr[b + 1]]
        #     batch_agent_valid_mask[av_index[b]] = False
        #     for t in range(num_step):
        #         batch_num_valid_agent_t = batch_agent_valid_mask[:, t].sum()
        #         seq_index[num_agent + b * self.num_seed_feature : num_agent + (b + 1) * self.num_seed_feature, t] += batch_num_valid_agent_t
        #         for s in range(self.num_seed_feature):
        #             seq_index[batch_sort_indices[s : s + 1, t].flatten().long() + ptr[b], t] = s + 1 + batch_num_valid_agent_t.float()

        seq_index[av_index] = 0

        return seq_mask, seq_index

    def _build_occ_gt(self, data, seq_mask, pos_rel_index_gt, pos_rel_index_gt_seed=None, mask_seed=None,
                      edge_index=None, mode='edge_index'):
        """
        Args:
            seq_mask (torch.Tensor): shape (num_step * num_seed_feature, num_agent + self.num_seed_feature)
            pos_rel_index_gt (torch.Tensor): shape (num_agent, num_step)
            pos_rel_index_gt_seed (torch.Tensor): shape (num_seed, num_step)
        """
        num_agent = data['agent']['state_idx'].shape[0] + data.num_graphs * self.num_seed_feature
        num_step = data['agent']['state_idx'].shape[1]
        data['agent']['agent_occ'] = torch.zeros(data.num_graphs * self.num_seed_feature, num_step, self.attr_tokenizer.grid_size,
                                                            device=data['agent']['state_idx'].device).long()
        data['agent']['map_occ'] = torch.zeros(data.num_graphs, num_step, self.attr_tokenizer.grid_size,
                                                            device=data['agent']['state_idx'].device).long()

        if mode == 'edge_index':

            assert edge_index is not None, f"Need edge_index input!"
            for src_index in torch.unique(edge_index[1]):
                # decode src
                src_row = src_index % num_agent - (num_agent - data.num_graphs * self.num_seed_feature)
                src_col = src_index // num_agent
                # decode tgt
                tgt_indexes = edge_index[0, edge_index[1] == src_index]
                tgt_rows = tgt_indexes % num_agent
                tgt_cols = tgt_indexes // num_agent
                assert tgt_rows.max() < num_agent - data.num_graphs * self.num_seed_feature, f"Invalid {tgt_rows}"
                assert torch.unique(tgt_cols).shape[0] == 1 and torch.unique(tgt_cols)[0] == src_col
                data['agent']['agent_occ'][src_row, src_col, pos_rel_index_gt[tgt_rows, tgt_cols]] = 1

        else:

            seq_mask = seq_mask.reshape(num_step, self.num_seed_feature, -1).transpose(0, 1)[..., :-self.num_seed_feature]
            for s in range(self.num_seed_feature):
                for t in range(num_step):
                    index = pos_rel_index_gt[seq_mask[s, t], t]
                    data['agent']['agent_occ'][s, t, index[index != -1]] = 1
                    if t > 0 and s < pos_rel_index_gt_seed.shape[0] and mask_seed[s, t - 1]: # insert agents
                        data['agent']['agent_occ'][s, t, pos_rel_index_gt_seed[s, t - 1]] = -1

        ptr = data['pt_token']['ptr']
        pt_grid_token_idx = data['agent']['pt_grid_token_idx'] # (t, num_pt)
        for b in range(data.num_graphs):
            batch_pt_grid_token_idx = pt_grid_token_idx[:, ptr[b] : ptr[b + 1]]
            for t in range(num_step):
                data['agent']['map_occ'][b, t, batch_pt_grid_token_idx[t][batch_pt_grid_token_idx[t] != -1]] = 1
        data['agent']['map_occ'] = data['agent']['map_occ'].repeat_interleave(repeats=self.num_seed_feature, dim=0)

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        pos_a = data['agent']['token_pos'].clone() # (a, t, 2)
        head_a = data['agent']['token_heading'].clone() # (a, t)
        num_agent, num_step, traj_dim = pos_a.shape # e.g. (50, 18, 2)
        agent_shape = data['agent']['shape'][:, self.num_historical_steps - 1].clone() # (a, 3)
        agent_token_index = data['agent']['token_idx'].clone() # (a, t)
        agent_state_index = data['agent']['state_idx'].clone()
        agent_type_index = data['agent']['type'].clone()

        av_index = data['agent']['av_index'].long()
        ego_pos = pos_a[av_index]
        ego_head = head_a[av_index]

        _, head_vector_a = self._build_vector_a(pos_a, head_a, agent_state_index)

        agent_grid_token_idx = data['agent']['grid_token_idx']
        agent_grid_offset_xy = data['agent']['grid_offset_xy']
        agent_head_token_idx = data['agent']['heading_token_idx']
        agent_pos_xy = data['agent']['pos_xy']
        agent_heading_theta = data['agent']['heading_theta']
        sort_indices = data['agent']['sort_indices']

        device = pos_a.device

        feat_a = self._agent_token_embedding(data,
                                             agent_token_index,
                                             agent_state_index,
                                             agent_grid_token_idx,
                                             pos_a,
                                             head_a,
                                             av_index=av_index)

        raw_feat_a = feat_a[:-data.num_graphs * self.num_seed_feature].clone()
        raw_feat_seed = feat_a[-data.num_graphs * self.num_seed_feature:].clone()

        # build masks
        mask = data['agent']['raw_agent_valid_mask'].clone()
        temporal_mask = mask.clone()
        interact_mask = mask.clone()

        is_bos = agent_state_index == self.enter_state
        is_eos = agent_state_index == self.exit_state
        bos_index = torch.where(is_bos.any(dim=1), torch.argmax(is_bos.long(), dim=1), torch.tensor(0))
        eos_index = torch.where(is_eos.any(dim=1), torch.argmax(is_eos.long(), dim=1), torch.tensor(num_step - 1)) # not `-1`

        temporal_mask = torch.ones_like(mask)
        motion_mask = torch.arange(mask.shape[1]).expand(mask.shape[0], -1).to(device)
        motion_mask = (motion_mask > bos_index[:, None]) & (motion_mask <= eos_index[:, None])
        temporal_mask[motion_mask] = mask[motion_mask]
        temporal_mask = torch.cat([temporal_mask, torch.ones(data.num_graphs * self.num_seed_feature, *temporal_mask.shape[1:], device=device)]).bool()

        interact_mask[agent_state_index == self.enter_state] = True
        interact_mask = torch.cat([interact_mask, torch.ones(data.num_graphs * self.num_seed_feature, *interact_mask.shape[1:], device=device)]).bool() # placeholder

        pos_a_p, head_a_p, state_a_p, head_vector_a_p, grid_index_a_p, pad_mask = \
            self._pad_feat(data.num_graphs, av_index, pos_a, head_a, agent_state_index, head_vector_a, agent_grid_token_idx)
        edge_index_t, r_t = self._build_temporal_edge(data, pos_a_p, head_a_p, state_a_p, head_vector_a_p, temporal_mask)

        # placeholder for seed agent
        batch_s = torch.cat([
            torch.cat([data['agent']['batch'], torch.arange(data.num_graphs, device=device
                        ).repeat_interleave(repeats=self.num_seed_feature, dim=0)], dim=0)
            + data.num_graphs * t for t in range(num_step)
        ], dim=0)
        batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t for t in range(num_step)], dim=0)

        seq_mask, seq_index = self._build_seq(device, data, num_agent, num_step, av_index, sort_indices)
        plot_kwargs = dict(is_bos=agent_state_index == self.enter_state)
        edge_index_a2a, r_a2a, (na2a, na2sa) = self._build_interaction_edge(data, pos_a_p, head_a_p, state_a_p, head_vector_a_p, batch_s,
                                                             interact_mask, pad_mask=pad_mask, av_index=av_index,
                                                             seq_mask=seq_mask, seq_index=seq_index, grid_index_a=grid_index_a_p, **plot_kwargs)

        edge_index_pl2a, r_pl2a, (npl2a, npl2sa) = self._build_map2agent_edge(data, pos_a_p, head_a_p, state_a_p, head_vector_a_p, batch_s, batch_pl,
                                                             interact_mask, pad_mask=pad_mask, av_index=av_index)
        interact_mask = interact_mask[:-data.num_graphs * self.num_seed_feature]

        # pos_a_s, head_a_s, state_a_s, head_vector_a_s, mask_a_s = self._build_seed_feat(data, pos_a_p, head_a_p, state_a_p, head_vector_a_p, ~pad_mask,
        #                                                                                 sort_indices, av_index=av_index)
        # edge_index_sa2sa, r_sa2sa = self._build_sa2sa_edge(data, pos_a_s, head_a_s, state_a_s, head_vector_a_s, batch_s, mask=mask_a_s)

        # for i in range(self.num_layers):

        #     feat_a = feat_a.reshape(-1, self.hidden_dim) # (a, t, d) -> (a*t, d)
        #     feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)

        #     feat_a = feat_a.reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        #     feat_a = self.pt2a_attn_layers[i]((
        #         map_enc['x_pt'].repeat_interleave(repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
        #         -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)

        #     feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
        #     feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

        # predict next motions
        for i in range(self.num_layers):

            feat_a = feat_a.reshape(-1, self.hidden_dim) # (a, t, d) -> (a*t, d)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)

            feat_a = feat_a.reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            feat_a = self.pt2a_attn_layers[i]((
                map_enc['x_pt'].repeat_interleave(repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                -1, self.hidden_dim), feat_a), r_pl2a[:npl2a], edge_index_pl2a[:, :npl2a])

            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a[:na2a], edge_index_a2a[:, :na2a])
            feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

        feat_ea = feat_a[:-data.num_graphs * self.num_seed_feature]

        # next motion token
        next_token_prob = self.token_predict_head(feat_ea) # (a, t, token_size)
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1) # (a, t, 10)

        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)

        # next state token
        next_state_prob = self.state_predict_head(feat_ea)
        next_state_idx = next_state_prob.softmax(dim=-1).argmax(dim=-1, keepdim=True) # (a, t, 1)

        next_state_index_gt = agent_state_index.roll(shifts=-1, dims=1) # (invalid, valid, exit)

        # predict next agents: coarse stage
        grid_agent_occ_gt_seed = grid_pt_occ_gt_seed = None
        if self.use_grid_token:
            self._build_occ_gt(data, seq_mask, agent_grid_token_idx.long(), edge_index=edge_index_a2a[:, -na2sa:], mode='edge_index')
            grid_agent_occ_gt_seed = data['agent']['agent_occ']
            grid_pt_occ_gt_seed = data['agent']['map_occ']

        if self.use_grid_token:
            occ_embed_a = self.seed_agent_occ_embed(grid_agent_occ_gt_seed.transpose(0, 1).reshape(-1, self.grid_size).float())
            # occ_embed_pt = self.seed_pt_occ_embed(grid_pt_occ_gt_seed.transpose(0, 1).reshape(-1, self.grid_size).float())
            edge_index_occ2sa_src = torch.arange(feat_a.shape[0] * feat_a.shape[1], device=device).long()
            edge_index_occ2sa_src = edge_index_occ2sa_src[~pad_mask.transpose(0, 1).reshape(-1)]
            edge_index_occ2sa_tgt = torch.arange(occ_embed_a.shape[0], device=device).long()
            edge_index_occ2sa = torch.stack([edge_index_occ2sa_tgt, edge_index_occ2sa_src], dim=0)

        feat_sa = torch.cat([raw_feat_a, raw_feat_seed])
        # feat_sa = feat_a
        for i in range(self.seed_layers):

            # feat_sa = feat_a.reshape(-1, self.hidden_dim)
            # feat_sa = self.sa2sa_attn_layers[i](feat_sa, r_sa2sa, edge_index_sa2sa)

            feat_sa = feat_sa.reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            if self.use_grid_token:
                feat_sa = self.occ2sa_attn_layers[i]((occ_embed_a, feat_sa), None, edge_index_occ2sa)
            feat_sa = self.pt2sa_attn_layers[i]((
                map_enc['x_pt'].repeat_interleave(repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                -1, self.hidden_dim), feat_sa), r_pl2a[-npl2sa:], edge_index_pl2a[:, -npl2sa:])
            feat_sa = self.a2sa_attn_layers[i](feat_sa, r_a2a[-na2sa:], edge_index_a2a[:, -na2sa:])
            feat_sa = feat_sa.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

        feat_seed = feat_sa[-data.num_graphs * self.num_seed_feature:]

        # seed agent
        next_state_prob_seed = self.seed_state_predict_head(feat_seed)
        raw_next_state_prob_seed = next_state_prob_seed.clone()
        next_state_idx_seed = next_state_prob_seed.softmax(dim=-1).argmax(dim=-1, keepdim=True) # (seed_size, t, 1)

        next_type_prob_seed = self.seed_type_predict_head(feat_seed)
        next_type_idx_seed = next_type_prob_seed.softmax(dim=-1).argmax(dim=-1, keepdim=True)

        next_type_index_gt = agent_type_index[:, None].repeat(1, num_step).long()

        next_shape_seed = self.seed_shape_predict_head(feat_seed)

        next_shape_gt = agent_shape[:, None].repeat(1, num_step, 1).float()

        if self.use_grid_token:
            next_pos_rel_prob_seed = self.seed_pos_rel_token_predict_head(feat_seed)
            next_pos_rel_idx_seed = next_pos_rel_prob_seed.softmax(dim=-1).argmax(dim=-1, keepdim=True)
        else:
            next_pos_rel_prob_seed = self.seed_pos_rel_xy_predict_head(feat_seed)
            next_pos_rel_xy_seed = torch.tanh(next_pos_rel_prob_seed)

        next_pos_rel_index_gt = agent_grid_token_idx.long()
        next_pos_rel_xy_gt = agent_pos_xy.float() / self.pl2seed_radius

        # decode grid index of neighbor agents
        if self.use_grid_token:
            neighbor_agent_grid_index_gt = grid_index_a_p.transpose(0, 1).reshape(-1)[edge_index_a2a[0, -na2sa:]]
            neighbor_pt_grid_index_gt = data['agent']['pt_grid_token_idx'].reshape(-1)[edge_index_pl2a[0, -npl2sa:]]
            neighbor_agent_grid_idx = self.grid_index_head(r_a2a[-na2sa:])
            neighbor_pt_grid_idx = self.grid_index_head(r_pl2a[-npl2sa:])
            neighbor_agent_grid_index_eval_mask = torch.zeros_like(neighbor_agent_grid_index_gt).bool()
            neighbor_pt_grid_index_eval_mask = torch.zeros_like(neighbor_pt_grid_index_gt).bool()
            neighbor_agent_grid_index_eval_mask[torch.randperm(neighbor_agent_grid_index_gt.shape[0])[:180]] = True
            neighbor_pt_grid_index_eval_mask[torch.randperm(neighbor_pt_grid_index_gt.shape[0])[:600]] = True

        # occupancy prediction
        grid_agent_occ_seed = grid_pt_occ_seed = grid_agent_occ_eval_mask_seed = grid_pt_occ_eval_mask_seed = None
        if self.predict_occ:
            # grid_occ_embed = self.grid_occ_embed(self.grid_token_emb[:-1])
            grid_agent_occ_seed = self.grid_agent_occ_head(feat_seed) # (s, t, d)
            grid_pt_occ_seed = self.grid_pt_occ_head(feat_seed)

        # refine stage
        batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t for t in range(num_step)], dim=0)
        batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t for t in range(num_step)], dim=0)

        mask_sa = torch.zeros_like(agent_state_index).bool()
        for t in range(mask_sa.shape[1]):
            availabel_rows = ((agent_state_index[:, t] != self.invalid_state) &
                              (agent_grid_token_idx[:, t] != -1)).nonzero()[..., 0]
            mask_sa[availabel_rows[torch.randperm(availabel_rows.shape[0])[:data.num_graphs * 10]], t] = True
        mask_sa[agent_state_index == self.enter_state] = True
        mask_sa[:, 0] = False # ignore the first step
        mask_sa[av_index] = False # ignore self

        state_sa = torch.full_like(agent_state_index, self.invalid_state).long()
        state_sa[mask_sa] = self.enter_state

        sa_indices = torch.nonzero(mask_sa)
        pos_sa = pos_a.clone()
        head_sa = head_a.clone()
        expanded_av_index = av_index.repeat_interleave(repeats=data['batch_size_a'], dim=0)
        head_sa[sa_indices[:, 0], sa_indices[:, 1]] = head_a[expanded_av_index[sa_indices[:, 0]], sa_indices[:, 1]]

        motion_vector_sa, head_vector_sa = self._build_vector_a(pos_a, head_sa, state_sa)
        motion_vector_sa[mask_sa] = self.motion_gap # fix the case e.g. [0, 0, 1, '1', 0, 1]

        offset_pos = pos_a - data['ego_pos'].repeat_interleave(repeats=data['batch_size_a'], dim=0)
        agent_grid_emb = self.grid_token_emb[agent_grid_token_idx.long()]
        feat_sa, _ = self._build_agent_feature(num_step, pos_a.device,
                                               motion_vector_sa,
                                               head_vector_sa,
                                               agent_grid_emb=agent_grid_emb,
                                               offset_pos=offset_pos,
                                               type=next_type_index_gt.long(),
                                               shape=next_shape_gt,
                                               state=state_sa,
                                               n=num_agent)
        feat_sa[~mask_sa] = raw_feat_a[~mask_sa].clone()

        edge_index_a2sa, r_a2sa = self._build_a2sa_edge(data, pos_a, head_sa, head_vector_sa, batch_s,
                                                        interact_mask, mask_sa=mask_sa, r=self.a2sa_radius)
        edge_index_pl2sa, r_pl2sa = self._build_map2sa_edge(data, pos_a, head_sa, head_vector_sa, batch_s, batch_pl,
                                                            mask_sa=mask_sa, r=self.pl2sa_radius)

        # sanity check
        global_index = set(torch.nonzero(mask_sa.transpose(0, 1).reshape(-1).int())[:, 0].tolist())
        a2sa_index = set(edge_index_a2sa[1].tolist())
        pl2sa_index = set(edge_index_pl2sa[1].tolist())
        assert a2sa_index.issubset(global_index) and pl2sa_index.issubset(global_index), "Invalid index!"

        select_mask = torch.zeros_like(mask_sa.view(-1)).bool()
        select_mask[torch.unique(edge_index_a2sa[1])] = True
        select_mask[torch.unique(edge_index_pl2sa[1])] = True
        mask_sa[~select_mask.reshape(num_step, -1).transpose(0, 1)] = False

        for i in range(self.seed_layers):

            feat_sa = feat_sa.transpose(0, 1).reshape(-1, self.hidden_dim)
            feat_sa = self.pt2a_attn_layers[i]((
                map_enc['x_pt'].repeat_interleave(repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                -1, self.hidden_dim), feat_sa), r_pl2sa, edge_index_pl2sa)

            feat_sa = self.a2a_attn_layers[i](feat_sa, r_a2sa, edge_index_a2sa)
            feat_sa = feat_sa.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

        if self.use_head_token:
            next_head_rel_theta_seed = None
            next_head_rel_prob_seed = self.seed_heading_rel_token_predict_head(feat_sa)
            next_head_rel_idx_seed = next_head_rel_prob_seed.softmax(dim=-1).argmax(dim=-1, keepdim=True)
        else:
            next_head_rel_prob_seed = None
            next_head_rel_theta_seed = self.seed_heading_rel_theta_predict_head(feat_sa)
            next_head_rel_theta_seed = torch.tanh(next_head_rel_theta_seed)[..., 0]

        next_head_rel_index_gt_seed = agent_head_token_idx.long()
        next_head_rel_theta_gt_seed = agent_heading_theta.float() / torch.pi  # [-1, 1]

        next_offset_xy_seed = None
        if self.use_grid_token:
            next_offset_xy_seed = self.seed_offset_xy_predict_head(feat_sa)
            next_offset_xy_seed = torch.tanh(next_offset_xy_seed) * 2  # [-2, 2]

        next_offset_xy_gt_seed = agent_grid_offset_xy.float()

        # next token prediction mask
        bos_token_index = torch.nonzero(agent_state_index == self.enter_state)
        eos_token_index = torch.nonzero(agent_state_index == self.exit_state)

        # mask for motion tokens
        next_token_eval_mask = mask.clone()
        next_token_eval_mask = next_token_eval_mask * next_token_eval_mask.roll(shifts=-1, dims=1) * next_token_eval_mask.roll(shifts=1, dims=1)
        for bos_token_index_ in bos_token_index:
            next_token_eval_mask[bos_token_index_[0], bos_token_index_[1] : bos_token_index_[1] + 1] = 1
            next_token_eval_mask[bos_token_index_[0], bos_token_index_[1] + 1 : bos_token_index_[1] + 2] = \
                            mask[bos_token_index_[0], bos_token_index_[1] + 2 : bos_token_index_[1] + 3]
        next_token_eval_mask[eos_token_index[:, 0], eos_token_index[:, 1]] = 0

        # mask for state tokens
        next_state_eval_mask = mask.clone()
        next_state_eval_mask = next_state_eval_mask * next_state_eval_mask.roll(shifts=-1, dims=1) * next_state_eval_mask.roll(shifts=1, dims=1)
        for bos_token_index_ in bos_token_index:
            next_state_eval_mask[bos_token_index_[0], :bos_token_index_[1]] = 0
            next_state_eval_mask[bos_token_index_[0], bos_token_index_[1] : bos_token_index_[1] + 1] = 1
            next_state_eval_mask[bos_token_index_[0], bos_token_index_[1] + 1 : bos_token_index_[1] + 2] = \
                            mask[bos_token_index_[0], bos_token_index_[1] + 2 : bos_token_index_[1] + 3]
        for eos_token_index_ in eos_token_index:
            next_state_eval_mask[eos_token_index_[0], eos_token_index_[1] + 1:] = 1
            next_state_eval_mask[eos_token_index_[0], eos_token_index_[1] : eos_token_index_[1] + 1] = \
                            mask[eos_token_index_[0], eos_token_index_[1] - 1 : eos_token_index_[1]]

        next_state_eval_mask_seed = torch.ones_like(next_state_idx_seed[..., 0])

        # the last timestep is the beginning of the sequence (also the input)
        next_token_eval_mask[:, 0] = mask[:, 0] * mask[:, 1]
        next_state_eval_mask[:, 0] = mask[:, 0] * mask[:, 1]
        next_token_eval_mask[:, -1] = 0
        next_state_eval_mask[:, -1] = 0
        next_state_eval_mask_seed[:, 0] = 0

        # no invalid motion token will be supervised
        if (next_token_index_gt[next_token_eval_mask] < 0).any():
            raise RuntimeError("Found invalid motion index.")

        # seed agents
        # is_next_bos = next_state_index_gt.roll(shifts=1, dims=1) == self.enter_state
        # is_next_bos[:, 0] = False # we filter out the last timestep
        # is_next_bos[av_index] = False

        # num_seed_gt = is_next_bos.sum(dim=0).max()

        # pred_indices = torch.zeros((num_seed_gt, num_step, 1), device=device).long()
        # gt_indices = torch.zeros((num_seed_gt, num_step), device=device).long()
        # if num_seed_gt > 0:
        #     outputs = dict(state_pred=next_state_prob_seed,
        #                    pos_pred=next_pos_rel_prob_seed,
        #                    shape_pred=next_shape_seed)
        #     targets = dict(state_gt=next_state_index_gt.clone(),
        #                    pos_gt=next_pos_rel_index_gt.clone(),
        #                    shape_gt=next_shape_gt.clone())

        #     indices = self.matcher(outputs, targets,
        #                            valid_mask=is_next_bos,
        #                            ptr_gt=data['agent']['ptr'],
        #                            ptr_pred=torch.arange(data.num_graphs + 1, device=device) * self.num_seed_feature)

        #     pred_indices = indices[0][..., None].to(device)
        #     gt_indices = indices[1].to(device)

        pred_indices = []
        gt_indices = []
        agent_ptr = data['agent']['ptr']
        num_seed_gt = 0
        for b in range(data.num_graphs):
            batch_sort_indices = sort_indices[agent_ptr[b] : agent_ptr[b + 1]]
            batch_num_seed_gt = min(self.num_seed_feature, batch_sort_indices.shape[0])
            num_seed_gt += batch_num_seed_gt
            pred_indices.append((torch.arange(batch_num_seed_gt, device=device) + b * self.num_seed_feature
                                )[:, None, None].repeat(1, num_step, 1).long())
            gt_indices.append(batch_sort_indices[:batch_num_seed_gt] + agent_ptr[b])
        pred_indices = torch.concat(pred_indices)
        gt_indices = torch.concat(gt_indices)

        n = pred_indices.shape[0]

        res_pred_indices = []
        for t in range(next_state_idx_seed.shape[1]):
            indices_t = torch.arange(next_state_idx_seed.shape[0]).to(device)
            selected_pred_mask = torch.zeros_like(indices_t)
            selected_pred_mask[pred_indices[:, t]] = 1
            res_pred_indices.append(indices_t[~selected_pred_mask.bool()])
        res_pred_indices = torch.stack(res_pred_indices, dim=1)
        padded_pred_indices = torch.concat([pred_indices, res_pred_indices[..., None]])
        next_state_idx_seed   = torch.gather(next_state_idx_seed, dim=0, index=padded_pred_indices)
        next_state_prob_seed  = torch.gather(next_state_prob_seed, dim=0, index=padded_pred_indices.expand(
                                                            -1, -1, next_state_prob_seed.shape[-1]))
        next_state_index_gt_seed = torch.gather(agent_state_index, dim=0, index=gt_indices)
        next_state_index_gt_seed = torch.concat([next_state_index_gt_seed,
                torch.zeros((next_state_prob_seed.shape[0] - next_state_index_gt_seed.shape[0], next_state_index_gt_seed.shape[1]), device=device)]).long()
        seed_enter_mask = next_state_index_gt_seed == self.enter_state
        next_state_index_gt_seed = torch.full(next_state_index_gt_seed.shape, self.seed_state_type.index('invalid'), device=device)
        next_state_index_gt_seed[seed_enter_mask] = self.seed_state_type.index('enter')

        next_type_idx_seed         = torch.gather(next_type_idx_seed, dim=0, index=pred_indices)
        next_type_prob_seed        = torch.gather(next_type_prob_seed, dim=0, index=pred_indices.expand(
                                                            -1, -1, next_type_prob_seed.shape[-1]))
        next_type_index_gt_seed    = torch.gather(next_type_index_gt, dim=0, index=gt_indices)

        if self.use_grid_token:
            next_pos_rel_xy_seed = None
            next_pos_rel_prob_seed     = torch.gather(next_pos_rel_prob_seed, dim=0, index=pred_indices.expand(
                                                                -1, -1, next_pos_rel_prob_seed.shape[-1]))
        else:
            next_pos_rel_prob_seed = None
            next_pos_rel_xy_seed       = torch.gather(next_pos_rel_xy_seed, dim=0, index=pred_indices.expand(
                                                                -1, -1, next_pos_rel_xy_seed.shape[-1]))
        next_pos_rel_index_gt_seed = torch.gather(next_pos_rel_index_gt, dim=0, index=gt_indices)
        next_pos_rel_xy_gt_seed    = torch.gather(next_pos_rel_xy_gt, dim=0, index=gt_indices[..., None].expand(
                                                                -1, -1, next_pos_rel_xy_gt.shape[-1]))

        next_shape_seed            = torch.gather(next_shape_seed, dim=0, index=pred_indices.expand(
                                                            -1, -1, next_shape_seed.shape[-1]))
        next_shape_gt_seed         = torch.gather(next_shape_gt, dim=0, index=gt_indices[..., None].expand(
                                                            -1, -1, next_shape_gt.shape[-1]))

        next_attr_eval_mask_seed = seed_enter_mask[:n]
        next_attr_eval_mask_seed[:, 0] = False  # we ignore the first step
        next_attr_eval_mask_seed[next_pos_rel_index_gt_seed == self.grid_size // 2] = False

        next_state_eval_mask[av_index] = 0 # we dont predict state for ego agent

        if (torch.any(next_type_index_gt_seed[next_attr_eval_mask_seed] == AGENT_TYPE.index('seed')) \
            or torch.any(torch.all(next_shape_gt_seed[next_attr_eval_mask_seed] == self.invalid_shape_value, dim=-1)) \
            or torch.any(next_pos_rel_index_gt_seed[next_attr_eval_mask_seed] < 0)) and num_seed_gt > 0:
            raise ValueError(f"Found invalid gt values in scenario {data['scenario_id'][0]}.")

        next_state_index_gt[next_state_index_gt == self.exit_state] = self.valid_state_type.index('exit')

        # build occ gt
        if self.predict_occ:

            grid_occ_eval_mask_seed = torch.ones_like(grid_agent_occ_seed).bool()
            grid_occ_eval_mask_seed[:, 0] = False
            grid_occ_eval_mask_seed[..., self.grid_size // 2] = False
            grid_agent_occ_eval_mask_seed = grid_pt_occ_eval_mask_seed = grid_occ_eval_mask_seed            

            # ! sanity check
            # s = random.randint(0, self.num_seed_feature - 1)
            # t = random.randint(0, num_step - 1)
            # grid_index = grid_agent_occ_gt_seed[s, t].nonzero()[..., 0]
            # check_mask = torch.zeros_like(pad_mask)
            # check_mask[av_index + s + 1, t] = 1
            # check_index = check_mask.transpose(0, 1).reshape(-1).nonzero()[..., 0]
            # check_agent_index = edge_index_a2a[0, edge_index_a2a[1] == check_index[0]] % (num_agent + self.num_seed_feature)
            # if not torch.all(grid_index == next_pos_rel_index_gt[check_agent_index, t].unique().sort()[0]):
            #     raise RuntimeError(f"Grid index not consistent s={s} t={t} scenario_id={data['scenario_id'][0]}")

        target_indices = pred_indices.clone()
        target_indices[~next_attr_eval_mask_seed] = -1

        return {'x_a': feat_a,
                'ego_pos': ego_pos,
                # motion token
                'next_token_idx': next_token_idx,
                'next_token_prob': next_token_prob,
                'next_token_idx_gt': next_token_index_gt,
                'next_token_eval_mask': next_token_eval_mask.bool(),
                # state token
                'next_state_idx': next_state_idx,
                'next_state_prob': next_state_prob,
                'next_state_idx_gt': next_state_index_gt,
                'next_state_eval_mask': next_state_eval_mask.bool(),
                # seed agent
                'next_state_idx_seed': next_state_idx_seed,
                'next_state_prob_seed': next_state_prob_seed,
                'next_state_idx_gt_seed': next_state_index_gt_seed,

                'next_type_idx_seed': next_type_idx_seed,
                'next_type_prob_seed': next_type_prob_seed,
                'next_type_idx_gt_seed': next_type_index_gt_seed,

                'next_pos_rel_prob_seed': next_pos_rel_prob_seed,
                'next_pos_rel_index_gt_seed': next_pos_rel_index_gt_seed,
                'next_pos_rel_xy_seed': next_pos_rel_xy_seed,
                'next_pos_rel_xy_gt_seed': next_pos_rel_xy_gt_seed,
                'next_head_rel_prob_seed': next_head_rel_prob_seed,
                'next_head_rel_index_gt_seed': next_head_rel_index_gt_seed,
                'next_head_rel_theta_seed': next_head_rel_theta_seed,
                'next_head_rel_theta_gt_seed': next_head_rel_theta_gt_seed,
                'next_offset_xy_seed': next_offset_xy_seed,
                'next_offset_xy_gt_seed': next_offset_xy_gt_seed,
                'next_shape_seed': next_shape_seed,
                'next_shape_gt_seed': next_shape_gt_seed,

                'grid_agent_occ_seed': grid_agent_occ_seed,
                'grid_pt_occ_seed': grid_pt_occ_seed,
                'grid_agent_occ_gt_seed': grid_agent_occ_gt_seed,
                'grid_pt_occ_gt_seed': grid_pt_occ_gt_seed,
                'neighbor_agent_grid_idx': neighbor_agent_grid_idx
                                            if self.use_grid_token else None,
                'neighbor_pt_grid_idx': neighbor_pt_grid_idx    
                                            if self.use_grid_token else None,
                'neighbor_agent_grid_index_gt': neighbor_agent_grid_index_gt
                                            if self.use_grid_token else None,
                'neighbor_pt_grid_index_gt': neighbor_pt_grid_index_gt
                                            if self.use_grid_token else None,

                'target_indices': target_indices[..., 0],
                'raw_next_state_prob_seed': raw_next_state_prob_seed,

                'next_state_eval_mask_seed': next_state_eval_mask_seed.bool(),
                'next_attr_eval_mask_seed': next_attr_eval_mask_seed.bool(),
                'next_head_eval_mask_seed': mask_sa.bool(),
                'grid_agent_occ_eval_mask_seed': grid_agent_occ_eval_mask_seed
                                            if self.use_grid_token else None,
                'grid_pt_occ_eval_mask_seed': grid_pt_occ_eval_mask_seed
                                            if self.use_grid_token else None,
                'neighbor_agent_grid_index_eval_mask': neighbor_agent_grid_index_eval_mask.bool()
                                                        if self.use_grid_token else None,
                'neighbor_pt_grid_index_eval_mask': neighbor_pt_grid_index_eval_mask.bool()
                                                        if self.use_grid_token else None,
                }

    def inference(self,
                  data: HeteroData,
                  map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        filter_mask = data['agent']['state_idx'][:, (self.num_historical_steps - 1) // self.shift - 1] != self.invalid_state

        seed_step_mask = data['agent']['state_idx'][:, (self.num_historical_steps - 1) // self.shift:] == self.enter_state
        # seed_agent_index_per_step = [torch.nonzero(seed_step_mask[:, t]).squeeze(dim=-1) for t in range(seed_step_mask.shape[1])]

        # num_historical_steps=11
        eval_mask = data['agent']['valid_mask'][filter_mask, self.num_historical_steps - 1]

        # agent attributes
        agent_id = data['agent']['id'][filter_mask].clone()
        agent_valid_mask = data['agent']['raw_agent_valid_mask'][filter_mask].clone() # token_valid_mask
        pos_a = data['agent']['token_pos'][filter_mask].clone() # (a, t, 2)
        token_a = data['agent']['token_idx'][filter_mask].clone() # (a, t)
        state_a = data['agent']['state_idx'][filter_mask].clone()
        head_a = data['agent']['token_heading'][filter_mask].clone()
        shape_a = data['agent']['shape'][filter_mask].clone()
        type_a = data['agent']['type'][filter_mask].clone()
        grid_a = data['agent']['grid_token_idx'][filter_mask].clone()
        gt_traj = data['agent']['position'][filter_mask, self.num_historical_steps:, :self.input_dim].contiguous()
        agent_token_traj_all = data['agent']['token_traj_all'][filter_mask]

        device = pos_a.device
        max_agent_id = agent_id.max()  # TODO: bs=1

        if self.num_recurrent_steps_val == -1:
            # self.num_recurrent_steps_val = 91 - 11 = 80
            self.num_recurrent_steps_val = data["agent"]['position'].shape[1] - self.num_historical_steps
        num_agent, num_ori_step, traj_dim = pos_a.shape
        num_infer_step = (self.num_recurrent_steps_val + self.num_historical_steps) // self.shift
        if num_infer_step > num_ori_step:
            pad_shape = num_agent, num_infer_step - num_ori_step
            agent_valid_mask = torch.cat([agent_valid_mask, torch.full(pad_shape, True, device=device)], dim=1)
            pos_a = torch.cat([pos_a, torch.zeros((*pad_shape, pos_a.shape[-1]), device=device)], dim=1)
            token_a = torch.cat([token_a, torch.full(pad_shape, -1, device=device)], dim=1)
            state_a = torch.cat([state_a, torch.full(pad_shape, self.invalid_state, device=device)], dim=1)
            head_a = torch.cat([head_a, torch.zeros(pad_shape, device=device)], dim=1)
            grid_a = torch.cat([grid_a, torch.full(pad_shape, -1, device=device)], dim=1)

        # TODO: support bs > 1 in inference !!!
        num_removed_agent = int((~filter_mask[:data['agent']['av_index']]).sum())
        data['batch_size_a'] -= num_removed_agent
        av_index = data['agent']['av_index'] - num_removed_agent

        # make future steps to zero
        pos_a[:, (self.num_historical_steps - 1) // self.shift:] = 0
        head_a[:, (self.num_historical_steps - 1) // self.shift:] = 0
        token_a[:, (self.num_historical_steps - 1) // self.shift:] = -1
        state_a[:, (self.num_historical_steps - 1) // self.shift:] = 0
        grid_a[:, (self.num_historical_steps - 1) // self.shift:] = -1

        motion_vector_a, head_vector_a = self._build_vector_a(pos_a, head_a, state_a)

        agent_valid_mask[:, (self.num_historical_steps - 1) // self.shift:] = True
        agent_valid_mask[~eval_mask] = False
        agent_token_index = data['agent']['token_idx'][filter_mask]
        agent_state_index = data['agent']['state_idx'][filter_mask]

        (feat_a, agent_token_emb, agent_token_emb_veh, agent_token_emb_ped, agent_token_emb_cyc, categorical_embs,
         trajectory_token_veh, trajectory_token_ped, trajectory_token_cyc) = self._agent_token_embedding(
                                                                                            data,
                                                                                            token_a,
                                                                                            state_a,
                                                                                            grid_a,
                                                                                            pos_a,
                                                                                            head_a,
                                                                                            inference=True,
                                                                                            filter_mask=filter_mask,
                                                                                            av_index=av_index,
                                                                                    )
        raw_feat_a = feat_a.clone()

        veh_mask = type_a == 0
        cyc_mask = type_a == 2
        ped_mask = type_a == 1

        pred_traj = torch.zeros(pos_a.shape[0], self.num_recurrent_steps_val, 2, device=device)  # (a, val_t, 2)
        pred_head = torch.zeros(pos_a.shape[0], self.num_recurrent_steps_val, device=device)
        pred_type = type_a.clone()
        pred_shape = shape_a[:, (self.num_historical_steps - 1) // self.shift - 1]  # (a, 3)
        pred_state = torch.zeros(pos_a.shape[0], self.num_recurrent_steps_val, device=device)
        pred_prob = torch.zeros(pos_a.shape[0], self.num_recurrent_steps_val // self.shift, device=device)  # (a, val_t)

        feat_a_t_dict = {}
        feat_sa_t_dict = {}

        # build masks (init)
        mask = agent_valid_mask.clone()
        temporal_mask = mask.clone()
        interact_mask = mask.clone()

        # find bos and eos index
        is_bos = state_a == self.enter_state
        is_eos = state_a == self.exit_state
        bos_index = torch.where(is_bos.any(dim=1), torch.argmax(is_bos.long(), dim=1), torch.tensor(0))
        eos_index = torch.where(is_eos.any(dim=1), torch.argmax(is_eos.long(), dim=1), torch.tensor(num_infer_step - 1))

        temporal_mask = torch.ones_like(mask)
        motion_mask = torch.arange(mask.shape[1]).expand(mask.shape[0], mask.shape[1]).to(mask.device)
        motion_mask = (motion_mask > bos_index[:, None]) & (motion_mask <= eos_index[:, None])
        motion_mask[:, self.num_historical_steps // self.shift:] = False
        temporal_mask[motion_mask] = mask[motion_mask]

        interact_mask = torch.ones_like(mask)
        non_motion_mask = ~motion_mask
        non_motion_mask[:, self.num_historical_steps // self.shift:] = False
        interact_mask[non_motion_mask] = 0
        interact_mask[state_a == self.enter_state] = 1
        interact_mask[av_index] = 1

        temporal_mask[:, (self.num_historical_steps - 1) // self.shift:] = 1
        interact_mask[:, (self.num_historical_steps - 1) // self.shift:] = 1

        self.log_message = ""
        num_inserted_agents_total = num_inserted_agents = 0
        next_token_idx_list = []
        next_state_idx_list = []
        grid_agent_occ_list = []
        grid_pt_occ_list = []
        grid_agent_occ_gt_list = []
        next_state_prob_seed_list = []
        next_pos_rel_prob_seed_list = []
        agent_labels = [[None] * num_infer_step for _ in range(pos_a.shape[0])]

        # append history motion/state tokens
        for i in range((self.num_historical_steps - 1) // self.shift):
            next_token_idx_list.append(agent_token_index[:, i : i + 1])
            next_state_idx_list.append(agent_state_index[:, i : i + 1])

        num_seed_feature = 1
        insert_limit = 10

        for t in (
            pbar := tqdm(range(self.num_recurrent_steps_val // self.shift), leave=False, desc='Timestep ...')
        ):

            # 1. insert agents
            num_new_agents = 0
            next_state_prob_seeds   = torch.zeros(10 + 1, 1, device=device)
            next_pos_rel_prob_seeds = torch.zeros(10 + 1, 1, self.attr_tokenizer.grid_size, device=device)
            grid_agent_occ_seeds    = torch.zeros(10 + 1, 1, self.attr_tokenizer.grid_size, device=device)
            grid_pt_occ_seeds       = torch.zeros(10 + 1, 1, self.attr_tokenizer.grid_size, device=device)
            grid_agent_occ_gt_seeds = torch.zeros(10 + 1, 1, self.attr_tokenizer.grid_size, device=device)

            valid_state_mask = state_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t] != self.invalid_state  # TODO: only support bs=1
            distance = ((pos_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t, :2] - pos_a[av_index, (self.num_historical_steps - 1) // self.shift - 1 + t, :2]) ** 2).sum(-1).sqrt()
            inrange_mask = distance <= self.pl2seed_radius
            seq_valid_mask = valid_state_mask & inrange_mask
            seq_valid_mask[av_index] = False
            res_seq_index = torch.zeros_like(state_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t])
            res_seq_index[seq_valid_mask] = torch.randperm(seq_valid_mask.sum(), device=device) + 1

            if t == 0:
                inference_mask = temporal_mask.clone()
                inference_mask = torch.cat([inference_mask, torch.ones_like(inference_mask[-1:]).repeat(
                    num_seed_feature, *([1] * (inference_mask.dim() - 1)))])
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t:] = False
            else:
                inference_mask = torch.zeros_like(temporal_mask)
                inference_mask = torch.cat([inference_mask, torch.zeros_like(inference_mask[-1:]).repeat(
                    num_seed_feature, *([1] * (inference_mask.dim() - 1)))])
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t - 1] = True

            plot_kwargs = dict()
            p = 0
            while True:

                p += 1
                if t == 0 or p - 1 >= insert_limit or self.disable_insertion: break

                # rebuild inference mask since number of agents have changed
                inference_mask = torch.zeros_like(temporal_mask)
                inference_mask = torch.cat([inference_mask, torch.zeros_like(inference_mask[-1:]).repeat(
                    num_seed_feature, *([1] * (inference_mask.dim() - 1)))])
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t - 1] = True

                # sanity check: make sure seed agents will interact with **all** non-invalid agents
                assert torch.all(state_a[:, :(self.num_historical_steps - 1) // self.shift + t][
                                ~interact_mask[:, :(self.num_historical_steps - 1) // self.shift + t]] == self.invalid_state) and \
                    torch.all(state_a[:, :(self.num_historical_steps - 1) // self.shift + t][
                                interact_mask[:, :(self.num_historical_steps - 1) // self.shift + t]] != self.invalid_state), \
                    f"Got wrong with interact mask at scenario {data['scenario_id'][0]} t={t}!"

                temporal_mask = torch.cat([temporal_mask, torch.ones_like(temporal_mask[:1]).repeat(
                                        num_seed_feature, *([1] * (temporal_mask.dim() - 1)))]).bool()
                interact_mask = torch.cat([interact_mask, torch.ones_like(interact_mask[:1]).repeat(
                                        num_seed_feature, *([1] * (interact_mask.dim() - 1)))]).bool() # placeholder

                pos_a_p, head_a_p, state_a_p, head_vector_a_p, grid_index_a_p, pad_mask = \
                    self._pad_feat(data.num_graphs, av_index, pos_a, head_a, state_a, head_vector_a, grid_a, num_seed_feature=num_seed_feature)
                # sanity check
                assert torch.all(~pad_mask[-num_seed_feature:]), "Got wrong with pad mask!"

                batch_s = torch.arange(num_infer_step, device=device).repeat_interleave(num_agent + num_seed_feature)
                batch_pl = torch.arange(num_infer_step, device=device).repeat_interleave(data['pt_token']['num_nodes'])

                inference_mask_sa = torch.zeros_like(inference_mask).bool()
                inference_mask_sa[:, (self.num_historical_steps - 1) // self.shift - 1 + t] = True

                # 1.1 build seed agent features
                if self.seed_use_ego_motion:
                    motion_vector_seed = motion_vector_a[av_index]
                    head_vector_seed = head_vector_a[av_index]
                else:
                    motion_vector_seed = head_vector_seed = None

                feat_seed, _ = self._build_agent_feature(num_infer_step, device,
                                                         motion_vector_seed,
                                                         head_vector_seed,
                                                         state_index=self.invalid_state,
                                                         n=num_seed_feature)

                if feat_a.shape[1] != feat_seed.shape[1]:
                    assert t == 0, f"Unmatched timestep {feat_a.shape[1]} and {feat_seed.shape[1]}."
                    feat_a = torch.cat([feat_a, feat_a[:, -1:].repeat(1, feat_seed.shape[1] - feat_a.shape[1], 1)], dim=1)

                raw_feat_a = feat_a.clone()
                feat_a = torch.cat([feat_a, feat_seed], dim=0)

                # 1.2 global feature aggregation
                plot_kwargs.update(t=t, n=num_new_agents, tag='global_feature')
                # 0, 0, 0, ..., N+1, N+2, ...
                seq_index = torch.cat([torch.zeros(pos_a.shape[0] - num_new_agents), torch.arange(num_new_agents + 1) + 1]).to(device)
                # 0, 2, 1, ..., N+1, N+2, ...
                # seq_index = torch.cat([res_seq_index, torch.arange(num_new_agents + 1, device=device) + 1 + seq_valid_mask.sum()])
                edge_index_a2seed, r_seed2a = self._build_a2sa_edge(data, pos_a_p, head_a_p, head_vector_a_p, batch_s,
                                                                    interact_mask.clone(),
                                                                    mask_sa=~pad_mask.clone(),
                                                                    inference_mask=inference_mask_sa,
                                                                    r=self.pl2seed_radius,
                                                                    max_num_neighbors=300,
                                                                    seq_index=seq_index,
                                                                    grid_index_a=grid_index_a_p,
                                                                    mode='insert', **plot_kwargs)
                edge_index_pl2seed, r_pl2seed = self._build_map2sa_edge(data, pos_a_p, head_a_p, head_vector_a_p, batch_s, batch_pl,
                                                                        mask_sa=~pad_mask.clone(),
                                                                        inference_mask=inference_mask_sa,
                                                                        r=self.pl2seed_radius,
                                                                        max_num_neighbors=2048,
                                                                        mode='insert')
                temporal_mask = temporal_mask[:-num_seed_feature]
                interact_mask = interact_mask[:-num_seed_feature]

                if self.use_grid_token:
                    grid_agent_occ_gt_t_1 = torch.zeros((self.grid_size,), device=device).long()
                    grid_t_1 = grid_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]
                    grid_agent_occ_gt_t_1[grid_t_1[grid_t_1 != -1]] = 1
                    occ_embed_a = self.seed_agent_occ_embed(grid_agent_occ_gt_t_1.reshape(1, self.grid_size).float()).repeat(num_seed_feature, 1)
                    edge_index_occ2sa_src = torch.arange(feat_a.shape[0] * feat_a.shape[1], device=device).long()
                    edge_index_occ2sa_src = edge_index_occ2sa_src[(~pad_mask.transpose(0, 1).reshape(-1)) & (inference_mask_sa.transpose(0, 1).reshape(-1))]
                    edge_index_occ2sa_tgt = torch.arange(occ_embed_a.shape[0], device=device).long()
                    edge_index_occ2sa = torch.stack([edge_index_occ2sa_tgt, edge_index_occ2sa_src], dim=0)

                for i in range(self.seed_layers):

                    feat_a = feat_a.transpose(0, 1).reshape(-1, self.hidden_dim)
                    if self.use_grid_token:
                        feat_a = self.occ2sa_attn_layers[i]((occ_embed_a, feat_a), None, edge_index_occ2sa)
                    feat_a = self.pt2sa_attn_layers[i]((
                        map_enc['x_pt'].repeat_interleave(repeats=num_infer_step, dim=0).reshape(-1, num_infer_step, self.hidden_dim).transpose(0, 1).reshape(
                        -1, self.hidden_dim), feat_a), r_pl2seed, edge_index_pl2seed)

                    feat_a = self.a2sa_attn_layers[i](feat_a, r_seed2a, edge_index_a2seed)
                    feat_a = feat_a.reshape(num_infer_step, -1, self.hidden_dim).transpose(0, 1)

                feat_seed = feat_a[-num_seed_feature:] # (s, t, d)
                
                ego_pos_t_1 = pos_a[av_index, (self.num_historical_steps - 1) // self.shift - 1 + t]
                ego_head_t_1 = head_a[av_index, (self.num_historical_steps - 1) // self.shift - 1 + t]

                # occupancy
                if self.predict_occ:
                    grid_agent_occ_seed = self.grid_agent_occ_head(feat_seed[:, (self.num_historical_steps - 1) // self.shift - 1 + t]) # (num_seed, grid_size)
                    grid_pt_occ_seed = self.grid_pt_occ_head(feat_seed[:, (self.num_historical_steps - 1) // self.shift - 1 + t])

                # insert prob
                next_state_prob_seed = self.seed_state_predict_head(feat_seed[:, (self.num_historical_steps - 1) // self.shift - 1 + t])
                next_state_idx_seed = next_state_prob_seed.softmax(dim=-1).argmax(dim=-1, keepdim=True)
                next_state_idx_seed[next_state_idx_seed == self.seed_state_type.index('invalid')] = self.invalid_state
                next_state_idx_seed[next_state_idx_seed == self.seed_state_type.index('enter')] = self.enter_state
                if int(os.getenv('DEBUG', 0)):
                    next_state_idx_seed = torch.full(next_state_idx_seed.shape, self.enter_state, device=device).long()

                # type and shape
                next_type_prob_seed = self.seed_type_predict_head(feat_seed[:, (self.num_historical_steps - 1) // self.shift - 1 + t])
                next_type_idx_seed = next_type_prob_seed.softmax(dim=-1).argmax(dim=-1, keepdim=True)
                next_shape_seed = self.seed_shape_predict_head(feat_seed[:, (self.num_historical_steps - 1) // self.shift - 1 + t])

                # position
                if self.use_grid_token:
                    next_pos_rel_prob_seed = self.seed_pos_rel_token_predict_head(feat_seed[:, (self.num_historical_steps - 1) // self.shift - 1 + t])
                    next_pos_rel_prob_softmax = torch.softmax(next_pos_rel_prob_seed, dim=-1)
                    # if self.inference_filter_overlap:
                        # next_pos_rel_prob_softmax[..., grid_agent_occ_gt_t_1.bool()] = 1e-6  # diffuse!
                    topk_pos_rel_prob, next_pos_rel_idx_seed = torch.topk(next_pos_rel_prob_softmax, k=self.insert_beam_size, dim=-1)
                    sample_pos_rel_index = torch.multinomial(topk_pos_rel_prob, 1).to(device)
                    next_pos_rel_idx_seed = next_pos_rel_idx_seed.gather(dim=1, index=sample_pos_rel_index)
                    next_pos_seed = self.attr_tokenizer.decode_pos(next_pos_rel_idx_seed[..., 0], y=ego_pos_t_1, theta_y=ego_head_t_1)
                    if self.inference_filter_overlap:
                        if grid_agent_occ_gt_t_1[next_pos_rel_idx_seed[..., 0]]:  # TODO: only support insert num=1 for each iter!!!
                            feat_a = raw_feat_a.clone()
                            continue
                else:
                    next_pos_rel_xy_seed = self.seed_pos_rel_xy_predict_head(feat_seed[:, (self.num_historical_steps - 1) // self.shift - 1 + t])
                    next_pos_seed = torch.tanh(next_pos_rel_xy_seed) * self.pl2seed_radius + ego_pos_t_1

                if torch.all(next_state_idx_seed == self.invalid_state) or num_new_agents + 1 > insert_limit:
                    break

                num_new_agent = 1 # TODO: fix this term
                num_new_agents += 1

                # ! 1.5. insert new agents and update attributes

                # append new agent id
                agent_id = torch.cat([agent_id, torch.tensor([max_agent_id + 1], device=device, dtype=agent_id.dtype)])
                max_agent_id += 1

                mask = torch.cat([mask, torch.ones(num_new_agent, num_infer_step, device=mask.device)], dim=0).bool()
                temporal_mask = torch.cat([temporal_mask, torch.ones(num_new_agent, num_infer_step, device=temporal_mask.device)], dim=0).bool()
                interact_mask = torch.cat([interact_mask, torch.ones(num_new_agent, num_infer_step, device=interact_mask.device)], dim=0).bool()

                # initialize new attributes
                new_pos_a = torch.zeros(num_new_agent, num_infer_step, 2, device=device)
                new_head_a = torch.zeros(num_new_agent, num_infer_step, device=device)
                new_grid_a = torch.full((num_new_agent, num_infer_step), -1, device=device)
                new_state_a = torch.full((num_new_agent, num_infer_step), self.invalid_state, device=state_a.device)
                new_shape_a = torch.full((num_new_agent, num_infer_step, 3), self.invalid_shape_value, device=device)
                new_type_a = torch.full((num_new_agent, num_infer_step), AGENT_TYPE.index('seed'), device=device)

                # add new attributes
                new_pos_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t] = next_pos_seed
                pos_a = torch.cat([pos_a, new_pos_a], dim=0)

                new_head_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t] = ego_head_t_1 # dummy values
                head_a = torch.cat([head_a, new_head_a], dim=0)

                if self.use_grid_token:
                    new_grid_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t] = next_pos_rel_idx_seed
                    grid_a = torch.cat([grid_a, new_grid_a], dim=0)

                new_type_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t:] = next_type_idx_seed
                new_shape_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t:] = next_shape_seed[:, None]
                pred_type = torch.cat([pred_type, new_type_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]])
                pred_shape = torch.cat([pred_shape, new_shape_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]])

                new_state_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t] = next_state_idx_seed # all enter state
                state_a = torch.cat([state_a, new_state_a], dim=0)

                mask[-num_new_agent:, :(self.num_historical_steps - 1) // self.shift + t] = 0
                interact_mask[-num_new_agent:, :(self.num_historical_steps - 1) // self.shift - 1 + t] = 0

                # placeholdersin pred_traj, pred_head, pred_state
                new_pred_traj = torch.zeros(num_new_agent, self.num_recurrent_steps_val, 2, device=device)
                new_pred_head = torch.zeros(num_new_agent, self.num_recurrent_steps_val, device=device)
                new_pred_state = torch.zeros(num_new_agent, self.num_recurrent_steps_val, device=device)

                if t > 0:
                    new_pred_traj[:, (t - 1) * 5 : t * 5] = new_pos_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t, None].repeat(1, 5, 1)
                    new_pred_head[:, (t - 1) * 5 : t * 5] = new_head_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t, None].repeat(1, 5)
                    new_pred_state[:, (t - 1) * 5 : t * 5] = next_state_idx_seed.repeat(1, 5)

                pred_traj = torch.cat([pred_traj, new_pred_traj], dim=0)
                pred_head = torch.cat([pred_head, new_pred_head], dim=0)
                pred_state = torch.cat([pred_state, new_pred_state], dim=0)

                # add new agents token embeddings
                new_agent_token_emb = self.no_token_emb(torch.zeros(1, device=device).long())[None, :].repeat(num_new_agent, num_infer_step, 1)
                new_agent_token_emb[:, (self.num_historical_steps - 1) // self.shift - 1 + t] = self.bos_token_emb(torch.zeros(1, device=device).long())
                agent_token_emb = torch.cat([agent_token_emb, new_agent_token_emb])
                next_veh_mask = next_type_idx_seed[..., 0] == AGENT_TYPE.index('veh')
                next_ped_mask = next_type_idx_seed[..., 0] == AGENT_TYPE.index('ped')
                next_cyc_mask = next_type_idx_seed[..., 0] == AGENT_TYPE.index('cyc')
                veh_mask = torch.cat([veh_mask, next_veh_mask])
                ped_mask = torch.cat([ped_mask, next_ped_mask])
                cyc_mask = torch.cat([cyc_mask, next_cyc_mask])

                # add new agents trajectory embeddings
                new_agent_token_traj_all = torch.zeros((num_new_agent, self.token_size, self.shift + 1, 4, 2), device=device)
                new_agent_token_traj_all[next_veh_mask] = trajectory_token_veh[None, ...]
                new_agent_token_traj_all[next_ped_mask] = trajectory_token_ped[None, ...]
                new_agent_token_traj_all[next_cyc_mask] = trajectory_token_cyc[None, ...]

                agent_token_traj_all = torch.cat([agent_token_traj_all, new_agent_token_traj_all], dim=0)

                new_categorical_embs = [self.type_a_emb(new_type_a.reshape(-1).long()), self.shape_emb(new_shape_a.reshape(-1, 3))]
                categorical_embs = [torch.cat([categorical_embs[0], new_categorical_embs[0]], dim=0),
                                    torch.cat([categorical_embs[1], new_categorical_embs[1]], dim=0)]

                new_labels = [None] * num_infer_step
                new_labels[(self.num_historical_steps - 1) // self.shift + t] = f'A{num_new_agents}' # the first step after bos step!
                agent_labels.append(new_labels)

                # 2. predict headings for seed agents
                motion_vector_sa, head_vector_sa = self._build_vector_a(pos_a[-num_new_agent:],
                                                                        head_a[-num_new_agent:],
                                                                        state_a[-num_new_agent:])
                # sanity check
                assert torch.all(motion_vector_sa[:, :(self.num_historical_steps - 1) // self.shift - 1 + t] == self.invalid_motion_value) and \
                       torch.all(motion_vector_sa[:,  (self.num_historical_steps - 1) // self.shift - 1 + t] == self.motion_gap), \
                       f"Found invalid values in motion_vectect_a at scenario {data['scenario_id'][0]} t={t}!"

                motion_vector_sa[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0.
                head_vector_sa[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0.
                motion_vector_a = torch.cat([motion_vector_a, motion_vector_sa])
                head_vector_a = torch.cat([head_vector_a, head_vector_sa])

                new_offset_pos = pos_a[-num_new_agent:] - pos_a[av_index]
                new_agent_grid_emb = self.grid_token_emb[new_grid_a] if self.use_grid_token else None

                feat_sa, _ = self._build_agent_feature(num_infer_step, device,
                                                       motion_vector_sa,
                                                       head_vector_sa,
                                                       agent_token_emb=new_agent_token_emb,
                                                       agent_grid_emb=new_agent_grid_emb,
                                                       offset_pos=new_offset_pos,
                                                       categorical_embs_a=new_categorical_embs,
                                                       state=new_state_a)

                feat_a = torch.cat([raw_feat_a, feat_sa])

                batch_s = torch.arange(num_infer_step, device=device).repeat_interleave(num_agent + num_new_agent)
                batch_pl = torch.arange(num_infer_step, device=device).repeat_interleave(data['pt_token']['num_nodes'])

                # sanity check
                assert pos_a.shape[0] == head_a.shape[0] == head_vector_a.shape[0] == interact_mask.shape[0] == \
                       pad_mask.shape[0] == inference_mask_sa.shape[0] == (num_agent + num_new_agent), f"Inconsistent shapes!"

                plot_kwargs.update(tag='heading')
                edge_index_a2sa, r_a2sa = self._build_a2sa_edge(data, pos_a, head_a, head_vector_a, batch_s,
                                                                interact_mask.clone(),
                                                                mask_sa=~pad_mask.clone(),
                                                                inference_mask=inference_mask_sa,
                                                                r=self.a2sa_radius,
                                                                max_num_neighbors=24,
                                                                **plot_kwargs)
                edge_index_pl2sa, r_pl2sa = self._build_map2sa_edge(data, pos_a, head_a, head_vector_a, batch_s, batch_pl,
                                                                    mask_sa=~pad_mask.clone(),
                                                                    inference_mask=inference_mask_sa,
                                                                    r=self.pl2sa_radius,
                                                                    max_num_neighbors=128)

                for i in range(self.seed_layers):

                    feat_a = feat_a.transpose(0, 1).reshape(-1, self.hidden_dim)
                    feat_a = self.pt2a_attn_layers[i]((
                        map_enc['x_pt'].repeat_interleave(repeats=num_infer_step, dim=0).reshape(-1, num_infer_step, self.hidden_dim).transpose(0, 1).reshape(
                        -1, self.hidden_dim), feat_a), r_pl2sa, edge_index_pl2sa)

                    feat_a = self.a2a_attn_layers[i](feat_a, r_a2sa, edge_index_a2sa)
                    feat_a = feat_a.reshape(num_infer_step, -1, self.hidden_dim).transpose(0, 1)

                if self.use_head_token:
                    next_head_rel_prob_seed = self.seed_heading_rel_token_predict_head(feat_a[-num_new_agent:, (self.num_historical_steps - 1) // self.shift - 1 + t])
                    next_head_rel_idx_seed = next_head_rel_prob_seed.softmax(dim=-1).argmax(dim=-1, keepdim=True)
                    next_head_seed = wrap_angle(self.attr_tokenizer.decode_heading(next_head_rel_idx_seed) + ego_head_t_1)
                else:
                    next_head_rel_theta_seed = self.seed_heading_rel_theta_predict_head(feat_a[-num_new_agent:, (self.num_historical_steps - 1) // self.shift - 1 + t])
                    next_head_seed = torch.tanh(next_head_rel_theta_seed) * torch.pi + ego_head_t_1

                head_a[-num_new_agent:, (self.num_historical_steps - 1) // self.shift - 1 + t] = next_head_seed

                if self.use_grid_token:
                    next_offset_xy_seed = self.seed_offset_xy_predict_head(feat_a[-num_new_agent:, (self.num_historical_steps - 1) // self.shift - 1 + t])
                    next_offset_xy_seed = torch.tanh(next_offset_xy_seed) * 2

                    pos_a[-num_new_agent:, (self.num_historical_steps - 1) // self.shift - 1 + t] += next_offset_xy_seed

                # ! finalize new features
                motion_vector_sa, head_vector_sa = self._build_vector_a(pos_a[-num_new_agent:],
                                                                        head_a[-num_new_agent:],
                                                                        state_a[-num_new_agent:])
                motion_vector_sa[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0.
                head_vector_sa[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0.
                motion_vector_a[-num_new_agent:] = motion_vector_sa
                head_vector_a[-num_new_agents:] = head_vector_sa

                feat_sa, _ = self._build_agent_feature(num_infer_step, device,
                                                       motion_vector_sa,
                                                       head_vector_sa,
                                                       agent_token_emb=new_agent_token_emb,
                                                       agent_grid_emb=new_agent_grid_emb,
                                                       offset_pos=new_offset_pos,
                                                       categorical_embs_a=new_categorical_embs,
                                                       state=state_a[-num_new_agent:],
                                                       n=num_new_agent)

                feat_a = torch.cat([raw_feat_a, feat_sa])
                raw_feat_a = feat_a.clone()

                num_agent = pos_a.shape[0]

                if self.use_grid_token:
                    grid_agent_occ_gt_seeds[num_new_agents] = grid_agent_occ_gt_t_1
                    grid_agent_occ_seeds[num_new_agents] = grid_agent_occ_seed
                    grid_pt_occ_seeds[num_new_agents] = grid_pt_occ_seed
                    next_pos_rel_prob_seeds[num_new_agents] = next_pos_rel_prob_softmax
                next_state_prob_seeds[num_new_agents] = next_state_prob_seed.softmax(dim=-1)[:, -1]

            inference_mask = inference_mask[:-num_seed_feature]
            next_state_prob_seed_list.append(next_state_prob_seeds)
            if self.use_grid_token:
                next_pos_rel_prob_seed_list.append(next_pos_rel_prob_seeds)
                grid_agent_occ_list.append(grid_agent_occ_seeds)
                grid_pt_occ_list.append(grid_pt_occ_seeds)
                grid_agent_occ_gt_list.append(grid_agent_occ_gt_seeds)
            next_state_idx_list[-1] = torch.cat([next_state_idx_list[-1], torch.full((num_new_agents, 1), self.enter_state, device=device).long()])

            # 3. predict motions for all agents
            feat_a = raw_feat_a

            # rebuild inference mask since number of agents have changed
            inference_mask = torch.zeros_like(temporal_mask)
            inference_mask[:, (self.num_historical_steps - 1) // self.shift + t - 1] = True

            edge_index_t, r_t = self._build_temporal_edge(data, pos_a, head_a, state_a, head_vector_a, temporal_mask, inference_mask.clone())

            batch_s = torch.arange(num_infer_step, device=device).repeat_interleave(num_agent)
            batch_pl = torch.arange(num_infer_step, device=device).repeat_interleave(data['pt_token']['num_nodes'])

            edge_index_a2a, r_a2a = self._build_interaction_edge(data, pos_a, head_a, state_a, head_vector_a, batch_s,
                                                                 interact_mask, inference_mask=inference_mask, av_index=av_index, **plot_kwargs)
            edge_index_pl2a, r_pl2a = self._build_map2agent_edge(data, pos_a, head_a, state_a, head_vector_a, batch_s, batch_pl,
                                                                 interact_mask, inference_mask=inference_mask, av_index=av_index, **plot_kwargs)

            for i in range(self.num_layers):

                if i in feat_a_t_dict:
                    feat_a = feat_a_t_dict[i]

                feat_a = feat_a.reshape(-1, self.hidden_dim)
                feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)

                feat_a = feat_a.reshape(-1, num_infer_step, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                feat_a = self.pt2a_attn_layers[i]((
                    map_enc['x_pt'].repeat_interleave(repeats=num_infer_step, dim=0).reshape(-1, num_infer_step, self.hidden_dim).transpose(0, 1).reshape(
                    -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)

                feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
                feat_a = feat_a.reshape(num_infer_step, -1, self.hidden_dim).transpose(0, 1)

                if t == 0:
                    feat_a_t_dict[i + 1] = feat_a
                else:
                    # update agent features at current step
                    n = feat_a_t_dict[i + 1].shape[0]
                    feat_a_t_dict[i + 1][:n, (self.num_historical_steps - 1) // self.shift - 1 + t] = feat_a[:n, (self.num_historical_steps - 1) // self.shift - 1 + t]
                    # add newly inserted agent features (only when t changed)
                    if feat_a.shape[0] > n:
                        m = feat_a.shape[0] - n
                        feat_a_t_dict[i + 1] = torch.cat([feat_a_t_dict[i + 1], feat_a[-m:]])

            # next motion token
            next_token_prob = self.token_predict_head(feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t])
            next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
            topk_token_prob, next_token_idx = torch.topk(next_token_prob_softmax, k=self.motion_beam_size, dim=-1) # both (num_agent, beam_size) e.g. (31, 5)

            # next state token
            next_state_prob = self.state_predict_head(feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t])
            next_state_idx = next_state_prob.softmax(dim=-1).argmax(dim=-1)
            next_state_idx[next_state_idx == self.valid_state_type.index('exit')] = self.exit_state
            next_state_idx[av_index] = self.valid_state # force ego_agent to be valid
            if not self.use_state_token:
                next_state_idx[next_state_idx == self.exit_state] = self.valid_state
            if self.disable_insertion:
                next_state_idx[:] = self.valid_state

            # convert the predicted token to a 0.5s (6 timesteps) trajectory
            expanded_token_index = next_token_idx[..., None, None, None].expand(-1, -1, 6, 4, 2)
            next_token_traj = torch.gather(agent_token_traj_all, 1, expanded_token_index) # (num_agent, beam_size, 6, 4, 2)

            # apply rotation and translation on 'next_token_traj'
            theta = head_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = torch.zeros((num_agent, 2, 2), device=theta.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_diff_rel = torch.bmm(next_token_traj.view(-1, 4, 2),
                                       rot_mat[:, None, None, ...].repeat(1, self.motion_beam_size, self.shift + 1, 1, 1).view(
                                           -1, 2, 2)).view(num_agent, self.motion_beam_size, self.shift + 1, 4, 2)
            agent_pred_rel = agent_diff_rel + pos_a[:, None, None, None, (self.num_historical_steps - 1) // self.shift - 1 + t, :]

            # sample 1 most probable index of top beam_size tokens, (num_agent, beam_size) -> (num_agent, 1)
            # then sample the agent_pred_rel, (num_agent, beam_size, 6, 4, 2) -> (num_agent, 6, 4, 2)
            sample_token_index = torch.multinomial(topk_token_prob, 1).to(agent_pred_rel.device)
            next_token_idx = next_token_idx.gather(dim=1, index=sample_token_index).squeeze(-1)
            agent_pred_rel = agent_pred_rel.gather(dim=1,
                                                   index=sample_token_index[..., None, None, None].expand(-1, -1, 6, 4,
                                                                                                    2))[:, 0, ...]

            # get predicted position and heading of current shifted timesteps
            diff_xy = agent_pred_rel[:, 1:, 0, :] - agent_pred_rel[:, 1:, 3, :]
            pred_traj[:num_agent, t * 5 : (t + 1) * 5] = agent_pred_rel[:, 1:, ...].clone().mean(dim=2)
            pred_head[:num_agent, t * 5 : (t + 1) * 5] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
            pred_state[:num_agent, t * 5 : (t + 1) * 5] = next_state_idx[:, None].repeat(1, 5)
            # pred_prob[:num_agent, t] = topk_token_prob.gather(dim=-1, index=sample_token_index)[:, 0] # (num_agent, beam_size) -> (num_agent,)

            # update pos/head/state of current step
            pos_a[:, (self.num_historical_steps - 1) // self.shift + t] = agent_pred_rel[:, -1, ...].clone().mean(dim=1)
            diff_xy = agent_pred_rel[:, -1, 0, :] - agent_pred_rel[:, -1, 3, :]
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            head_a[:, (self.num_historical_steps - 1) // self.shift + t] = theta
            state_a[:, (self.num_historical_steps - 1) // self.shift + t] = next_state_idx
            if self.use_grid_token:
                grid_a[:, (self.num_historical_steps - 1) // self.shift + t] = self.attr_tokenizer.encode_pos(
                    x=pos_a[:, (self.num_historical_steps - 1) // self.shift + t],
                    y=pos_a[av_index, (self.num_historical_steps - 1) // self.shift + t],
                    theta_y=theta[av_index],
                )[0]

            # the case that the current predicted state token is invalid/exit
            is_eos = next_state_idx == self.exit_state
            is_invalid = next_state_idx == self.invalid_state

            next_token_idx[is_invalid] = -1
            pos_a[is_invalid, (self.num_historical_steps - 1) // self.shift + t] = 0.
            head_a[is_invalid, (self.num_historical_steps - 1) // self.shift + t] = 0.
            if self.use_grid_token:
                grid_a[is_invalid, (self.num_historical_steps - 1) // self.shift + t] = -1

            mask[is_invalid, (self.num_historical_steps - 1) // self.shift + t] = False # to handle those newly-added agents
            interact_mask[is_invalid, (self.num_historical_steps - 1) // self.shift + t] = False

            agent_token_emb[is_invalid, (self.num_historical_steps - 1) // self.shift + t] = self.no_token_emb(torch.zeros(1, device=device).long())

            type_emb = categorical_embs[0].reshape(num_agent, num_infer_step, -1)
            shape_emb = categorical_embs[1].reshape(num_agent, num_infer_step, -1)
            type_emb[is_invalid, (self.num_historical_steps - 1) // self.shift + t] = self.type_a_emb(torch.tensor(AGENT_TYPE.index('seed'), device=device).long())
            shape_emb[is_invalid, (self.num_historical_steps - 1) // self.shift + t] = self.shape_emb(torch.full((1, 3), self.invalid_shape_value, device=device))
            categorical_embs = [type_emb.reshape(-1, self.hidden_dim), shape_emb.reshape(-1, self.hidden_dim)]

            # if is_eos.any():

            #     pos_a[is_eos, (self.num_historical_steps - 1) // self.shift + t + 1:] = 0.
            #     head_a[is_eos, (self.num_historical_steps - 1) // self.shift + t + 1:] = 0.
            #     mask[is_eos, (self.num_historical_steps - 1) // self.shift + t + 1:] = False # to handle those newly-added agents
            #     interact_mask[torch.cat([is_eos, torch.zeros(1, device=is_eos.device).bool()]), (self.num_historical_steps - 1) // self.shift + t + 1:] = False

            #     agent_token_emb[is_eos, (self.num_historical_steps - 1) // self.shift + t + 1:] = self.no_token_emb(torch.zeros(1, device=device).long())

            #     type_emb = categorical_embs[0].reshape(num_agent, num_infer_step, -1)
            #     shape_emb = categorical_embs[1].reshape(num_agent, num_infer_step, -1)
            #     type_emb[is_eos, (self.num_historical_steps - 1) // self.shift + t + 1:] = self.type_a_emb(torch.tensor(AGENT_TYPE.index('seed'), device=device).long())
            #     shape_emb[is_eos, (self.num_historical_steps - 1) // self.shift + t + 1:] = self.shape_emb(torch.full((1, 3), self.invalid_shape_value, device=device))
            #     categorical_embs = [type_emb.reshape(-1, self.hidden_dim), shape_emb.reshape(-1, self.hidden_dim)]

            # update token embeddings of current step
            agent_token_emb[veh_mask, (self.num_historical_steps - 1) // self.shift + t] = agent_token_emb_veh[
                next_token_idx[veh_mask]]
            agent_token_emb[ped_mask, (self.num_historical_steps - 1) // self.shift + t] = agent_token_emb_ped[
                next_token_idx[ped_mask]]
            agent_token_emb[cyc_mask, (self.num_historical_steps - 1) // self.shift + t] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]]

            # 4. update feat_a (t-1)
            motion_vector_a, head_vector_a = self._build_vector_a(pos_a, head_a, state_a)
            motion_vector_a[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0.
            head_vector_a[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0.

            offset_pos = pos_a - pos_a[av_index]

            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 # torch.norm(offset_pos[:, :, :2], p=2, dim=-1),
                 ], dim=-1)

            x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)),
                               categorical_embs=categorical_embs)
            x_a = x_a.view(-1, num_infer_step, self.hidden_dim)

            s_a = self.state_a_emb(state_a.reshape(-1).long()).reshape(-1, num_infer_step, self.hidden_dim)
            feat_a = torch.cat((agent_token_emb, x_a, s_a), dim=-1)
            if self.use_grid_token:
                agent_grid_emb = self.grid_token_emb[grid_a]
                feat_a = torch.cat([feat_a, agent_grid_emb], dim=-1)
            feat_a = self.fusion_emb(feat_a)
            raw_feat_a = feat_a.clone()  # ! IMPORANT: need to update `raw_feat_a`

            next_token_idx_list.append(next_token_idx[:, None])
            next_state_idx_list.append(next_state_idx[:, None])

            # get log message
            num_inserted_agents_total += num_new_agents
            num_inserted_agents += num_new_agents
            if num_new_agents > 0:
                self.log(t, next_pos_seed, ego_pos_t_1, next_head_seed, ego_head_t_1, next_shape_seed, next_type_idx_seed)

            # pbar
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            pbar.set_postfix(memory=f'{allocated_memory:.2f}GB',
                            insert=f'{num_inserted_agents_total}/{seed_step_mask.sum()}')

        for i in range(len(next_token_idx_list)):
            next_token_idx_list[i] = torch.cat([next_token_idx_list[i], torch.zeros(num_agent - next_token_idx_list[i].shape[0], 1, device=device) - 1], dim=0).long()  # -1: invalid motion token
            next_state_idx_list[i] = torch.cat([next_state_idx_list[i], torch.zeros(num_agent - next_state_idx_list[i].shape[0], 1, device=device)], dim=0).long()      # 0: invalid state token

        # add history attributes
        num_agent = pred_traj.shape[0]
        num_init_agent = filter_mask.sum()

        pred_traj = torch.cat([torch.zeros(num_agent, self.num_historical_steps, *(pred_traj.shape[2:]), device=pred_traj.device), pred_traj], dim=1)
        pred_head = torch.cat([torch.zeros(num_agent, self.num_historical_steps, *(pred_head.shape[2:]), device=pred_head.device), pred_head], dim=1)
        pred_state = torch.cat([torch.zeros(num_agent, self.num_historical_steps, *(pred_state.shape[2:]), device=pred_state.device), pred_state], dim=1)

        pred_traj[:num_init_agent, 0] = data['agent']['position'][filter_mask, 0, :2]
        pred_head[:num_init_agent, 0] = data['agent']['heading'][filter_mask, 0]
        pred_state[:num_init_agent, 1 : self.num_historical_steps] = data['agent']['state_idx'][filter_mask, :(self.num_historical_steps - 1) // self.shift].repeat_interleave(repeats=self.shift, dim=1)

        historical_token_idx = data['agent']['token_idx'][filter_mask, :(self.num_historical_steps - 1) // self.shift]
        historical_token_idx[historical_token_idx < 0] = 0
        historical_token_traj_all = torch.gather(agent_token_traj_all, 1, historical_token_idx[..., None, None, None].expand(-1, -1, 6, 4, 2))
        init_theta = head_a[:num_init_agent, 0]
        cos, sin = init_theta.cos(), init_theta.sin()
        rot_mat = torch.zeros((num_init_agent, 2, 2), device=init_theta.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        historical_token_traj_all = torch.bmm(historical_token_traj_all.view(-1, 4, 2),
                                              rot_mat[:, None, None, ...].repeat(1, (self.num_historical_steps - 1) // self.shift, self.shift + 1, 1, 1).view(
                                              -1, 2, 2)).view(num_init_agent, (self.num_historical_steps - 1) // self.shift, self.shift + 1, 4, 2)
        historical_token_traj_all = historical_token_traj_all + pos_a[:num_init_agent, 0, :][:, None, None, None, ...]
        pred_traj[:num_init_agent, 1 : self.num_historical_steps] = historical_token_traj_all[:, :, 1:, ...].clone().mean(dim=3).reshape(num_init_agent, -1, 2)
        diff_xy = historical_token_traj_all[..., 1:, 0, :] - historical_token_traj_all[..., 1:, 3, :]
        pred_head[:num_init_agent, 1 : self.num_historical_steps] = torch.arctan2(diff_xy[..., 1], diff_xy[..., 0]).reshape(num_init_agent, -1)

        # ! build z and valid
        pred_z = torch.zeros_like(pred_traj[..., 0])  # hard code
        pred_valid = (pred_state != self.invalid_state) & (pred_state != self.enter_state)

        # ! predefined agent shape
        eval_shape = torch.zeros_like(pred_shape)
        eval_shape[veh_mask] = torch.tensor(AGENT_SHAPE['vehicle'], device=device)[None, ...]
        eval_shape[ped_mask] = torch.tensor(AGENT_SHAPE['pedstrain'], device=device)[None, ...]
        eval_shape[cyc_mask] = torch.tensor(AGENT_SHAPE['cyclist'], device=device)[None, ...]

        next_token_idx = torch.cat(next_token_idx_list, dim=-1)
        next_state_idx = torch.cat(next_state_idx_list, dim=-1) if len(next_state_idx_list) > 0 else None

        # sanity check
        assert torch.all(pos_a[next_state_idx == self.invalid_state] == 0), f'Invalid step should have all zeros position!'

        if self.log_message == "":
            self.log_message = "No agents inserted!"
        else:
            self.log_message += f"\nNumber of total inserted agents: {num_inserted_agents_total}/{seed_step_mask.sum()}"

        return {
            'ego_index': int(av_index),
            'agent_id': agent_id,
            # 'valid_mask': agent_valid_mask[:, self.num_historical_steps:],
            # 'pos_a': pos_a[:, (self.num_historical_steps - 1) // self.shift:],
            # 'head_a': head_a[:, (self.num_historical_steps - 1) // self.shift:],
            'valid_mask': agent_valid_mask,  # [n_agent, n_infer_step // shift]
            'pos_a': pos_a,     # [n_agent, n_infer_step // shift, 2]
            'head_a': head_a,   # [n_agent, n_infer_step // shift]
            'gt_traj': gt_traj,
            'pred_traj': pred_traj,  # [n_agent, n_infer_step, 2]
            'pred_head': pred_head,  # [n_agent, n_infer_step]
            'pred_type': pred_type,
            'pred_state': pred_state,
            'pred_z': pred_z,
            'pred_shape': pred_shape,
            'eval_shape': eval_shape,
            'pred_valid': pred_valid,
            'next_state_prob_seed': torch.cat(next_state_prob_seed_list, dim=1),
            'next_pos_rel_prob_seed': torch.cat(next_pos_rel_prob_seed_list, dim=1)
                                            if self.use_grid_token else None,
            'next_token_idx': next_token_idx,  # [n_agent, n_infer_step // shift]
            'next_state_idx': next_state_idx,  # [n_agent, n_infer_step // shift]
            'grid_agent_occ_seed': torch.cat(grid_agent_occ_list, dim=1)
                                            if self.use_grid_token else None,
            'grid_pt_occ_seed': torch.cat(grid_pt_occ_list, dim=1)
                                            if self.use_grid_token else None,
            'grid_agent_occ_gt_seed': torch.cat(grid_agent_occ_gt_list, dim=1)
                                            if self.use_grid_token else None,
            'agent_labels': agent_labels,
            'log_message': self.log_message,
        }

    def log(self, t, next_pos_seed, ego_pos, next_head_seed, ego_head, next_shape_seed, next_type_idx_seed):
        i = 0
        _repr_indent = 4
        for sa in range(next_pos_seed.shape[0]):
            head = f"\n{i} agent {sa} is entering at step {t}"
            body = [
                f"rel pos {(next_pos_seed[sa] - ego_pos).tolist()}, pos {next_pos_seed[sa].tolist()}",
                f"rel head {wrap_angle(next_head_seed[sa] - ego_head).item()}, head {next_head_seed[sa].item()}",
                f"shape {next_shape_seed[sa].tolist()}, type {next_type_idx_seed[sa].item()}",
            ]
            self.log_message += "\n".join([head] + [" " * _repr_indent + line for line in body])
            i += 1
