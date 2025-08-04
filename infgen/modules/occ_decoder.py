import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Mapping, Optional, Literal
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import dense_to_sparse, subgraph
from scipy.optimize import linear_sum_assignment

from infgen.modules.attr_tokenizer import Attr_Tokenizer
from infgen.modules.layers import *
from infgen.utils.visualization import *
from infgen.utils.func import angle_between_2d_vectors, wrap_angle, weight_init


class InfGenOccDecoder(nn.Module):

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
                 token_data: Dict,
                 token_size: int,
                 special_token_index: list=[],
                 attr_tokenizer: Attr_Tokenizer=None,
                 predict_motion: bool=False,
                 predict_state: bool=False,
                 predict_map: bool=False,
                 predict_occ: bool=False,
                 state_token: Dict[str, int]=None,
                 seed_size: int=5,
                 buffer_size: int=32,
                 loss_weight: dict=None,
                 logger=None) -> None:

        super(InfGenOccDecoder, self).__init__()
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
        self.special_token_index = special_token_index
        self.predict_motion = predict_motion
        self.predict_state = predict_state
        self.predict_map = predict_map
        self.predict_occ = predict_occ
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

        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3

        self.seed_size = seed_size
        self.buffer_size = buffer_size

        self.agent_type = ['veh', 'ped', 'cyc', 'seed']
        self.type_a_emb = nn.Embedding(len(self.agent_type), hidden_dim)
        self.shape_emb = MLPEmbedding(input_dim=3, hidden_dim=hidden_dim)
        self.state_a_emb = nn.Embedding(len(self.state_type), hidden_dim)
        self.motion_gap = 1.
        self.heading_gap = 1.
        self.invalid_shape_value = .1
        self.invalid_motion_value = -2.
        self.invalid_head_value = -2.

        self.r_pt2a_emb = FourierEmbedding(input_dim=input_dim_r_pt2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)

        self.token_size = token_size # 2048
        self.grid_size = self.attr_tokenizer.grid_size
        self.angle_size = self.attr_tokenizer.angle_size
        self.agent_limit = 3
        self.pt_limit = 10
        self.grid_agent_occ_head = MLPLayer(input_dim=hidden_dim, hidden_dim=self.grid_size,
                                            output_dim=self.agent_limit * self.grid_size)
        self.grid_pt_occ_head = MLPLayer(input_dim=hidden_dim, hidden_dim=self.grid_size,
                                         output_dim=self.pt_limit * self.grid_size)

        # self.num_seed_feature = 1
        # self.num_seed_feature = self.seed_size
        self.num_seed_feature = 10

        self.trajectory_token = token_data['token'] # dict('veh', 'ped', 'cyc') (2048, 4, 2)
        self.trajectory_token_traj = token_data['traj'] # (2048, 6, 3)
        self.trajectory_token_all = token_data['token_all'] # (2048, 6, 4, 2)
        self.apply(weight_init)

        self.shift = 5
        self.beam_size = 5
        self.hist_mask = True
        self.temporal_attn_to_invalid = False
        self.use_rel = False

        # seed agent
        self.temporal_attn_seed = False
        self.seed_attn_to_av = True
        self.seed_use_ego_motion = False

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
        veh_mask = (agent_type == 0)
        ped_mask = (agent_type == 1)
        cyc_mask = (agent_type == 2)

        motion_vector_a, head_vector_a = self._build_vector_a(pos_a, head_a, agent_state)

        trajectory_token_veh = torch.from_numpy(self.trajectory_token['veh']).clone().to(pos_a.device).to(torch.float)
        trajectory_token_ped = torch.from_numpy(self.trajectory_token['ped']).clone().to(pos_a.device).to(torch.float)
        trajectory_token_cyc = torch.from_numpy(self.trajectory_token['cyc']).clone().to(pos_a.device).to(torch.float)
        self.agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh.view(trajectory_token_veh.shape[0], -1)) # (token_size, 8)
        self.agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped.view(trajectory_token_ped.shape[0], -1))
        self.agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc.view(trajectory_token_cyc.shape[0], -1))

        # add bos token embedding
        self.agent_token_emb_veh = torch.cat([self.agent_token_emb_veh, self.bos_token_emb(torch.zeros(1, device=pos_a.device).long())])
        self.agent_token_emb_ped = torch.cat([self.agent_token_emb_ped, self.bos_token_emb(torch.zeros(1, device=pos_a.device).long())])
        self.agent_token_emb_cyc = torch.cat([self.agent_token_emb_cyc, self.bos_token_emb(torch.zeros(1, device=pos_a.device).long())])

        # add invalid token embedding
        self.agent_token_emb_veh = torch.cat([self.agent_token_emb_veh, self.no_token_emb(torch.zeros(1, device=pos_a.device).long())])
        self.agent_token_emb_ped = torch.cat([self.agent_token_emb_ped, self.no_token_emb(torch.zeros(1, device=pos_a.device).long())])
        self.agent_token_emb_cyc = torch.cat([self.agent_token_emb_cyc, self.no_token_emb(torch.zeros(1, device=pos_a.device).long())])

        # self.grid_token_emb = self.token_emb_grid(torch.stack([self.attr_tokenizer.dist,
        #                                                        self.attr_tokenizer.dir], dim=-1).to(pos_a.device))
        self.grid_token_emb = self.token_emb_grid(self.attr_tokenizer.grid)
        self.grid_token_emb = torch.cat([self.grid_token_emb, self.invalid_offset_token_emb(torch.zeros(1, device=pos_a.device).long())])

        if inference:
            agent_token_traj_all = torch.zeros((num_agent, self.token_size, self.shift + 1, 4, 2), device=pos_a.device)
            trajectory_token_all_veh = torch.from_numpy(self.trajectory_token_all['veh']).clone().to(pos_a.device).to(torch.float)
            trajectory_token_all_ped = torch.from_numpy(self.trajectory_token_all['ped']).clone().to(pos_a.device).to(torch.float)
            trajectory_token_all_cyc = torch.from_numpy(self.trajectory_token_all['cyc']).clone().to(pos_a.device).to(torch.float)
            agent_token_traj_all[veh_mask] = torch.cat(
                [trajectory_token_all_veh[:, :self.shift], trajectory_token_veh[:, None, ...]], dim=1)
            agent_token_traj_all[ped_mask] = torch.cat(
                [trajectory_token_all_ped[:, :self.shift], trajectory_token_ped[:, None, ...]], dim=1)
            agent_token_traj_all[cyc_mask] = torch.cat(
                [trajectory_token_all_cyc[:, :self.shift], trajectory_token_cyc[:, None, ...]], dim=1)

        # additional token embeddings are already added -> -1: invalid, -2: bos
        agent_token_emb = torch.zeros((num_agent, num_step, self.hidden_dim), device=pos_a.device)
        agent_token_emb[veh_mask] = self.agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = self.agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = self.agent_token_emb_cyc[agent_token_index[cyc_mask]]

        offset_token_emb = self.grid_token_emb[agent_offset_token_idx]

        # 'vehicle', 'pedestrian', 'cyclist', 'background'
        is_invalid = agent_state == self.invalid_state
        agent_types = data['agent']['type'].clone()[filter_mask].long().repeat_interleave(repeats=num_step, dim=0)
        agent_types[is_invalid.reshape(-1)] = self.agent_type.index('seed')
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
            return feat_a, agent_token_traj_all, agent_token_emb, categorical_embs
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

            feat_a = torch.cat([feat_a, feat_seed], dim=0) # (a + n, t, d)

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
                type = torch.tensor([self.agent_type.index('seed')], device=device)
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

        feat_a = torch.cat((agent_token_emb, x_a, s_a, agent_grid_emb), dim=-1)
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

    def _build_seed_feat(self, data, pos_a, head_a, state_a, head_vector_a, mask, sort_indices, av_index):
        seed_mask = sort_indices != av_index.repeat_interleave(repeats=data['batch_size_a'], dim=0)[:, None]
        # TODO: fix batch_size!!!
        print(mask.shape, sort_indices.shape, seed_mask.shape)
        mask[-data.num_graphs * self.num_seed_feature:] = seed_mask[:self.num_seed_feature]

        insert_pos_a = torch.gather(pos_a, dim=0, index=sort_indices[:self.num_seed_feature, :, None].expand(-1, -1, pos_a.shape[-1]))
        pos_a[mask] = insert_pos_a[mask[-self.num_seed_feature:]]

        state_a[-data.num_graphs * self.num_seed_feature:] = self.enter_state

        return pos_a, head_a, state_a, head_vector_a, mask

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

        if os.getenv('PLOT_EDGE', False):
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
             rel_head_a2a,
             torch.zeros_like(edge_index_a2a[0])], dim=-1)
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
                                                                seq_mask=seq_mask, seq_index=seq_index, grid_index_a=grid_index_a, mode='grid')

            if os.getenv('PLOT_EDGE', False):
                plot_interact_edge(edge_index_seed2a, data['scenario_id'], data['batch_size_a'].cpu(), self.num_seed_feature, num_step,
                                   'interact_edge_map_seed', av_index=av_index, **plot_kwargs)

            edge_index_a2a = torch.cat([edge_index_a2a, edge_index_seed2a], dim=-1)
            r_a2a = torch.cat([r_a2a, r_seed2a])

            return edge_index_a2a, r_a2a, (edge_index_a2a.shape[1], edge_index_seed2a.shape[1]) #, nearest_dict

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
                                                                    r=self.pl2seed_radius, max_num_neighbors=2048, mode='grid')

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
                         grid_index_a=None, mode: Literal['grid', 'heading']='heading', **plot_kwargs):

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

        r_a2sa = torch.stack(
            [torch.norm(rel_pos_a2sa[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2sa[1]], nbr_vector=rel_pos_a2sa[:, :2]),
            rel_head_a2sa,
            seq_index[edge_index_a2sa[0]] - seq_index[edge_index_a2sa[1]]], dim=-1)
        r_a2sa = self.r_a2sa_emb(continuous_inputs=r_a2sa, categorical_embs=None)

        return edge_index_a2sa, r_a2sa

    def _build_map2sa_edge(self, data, pos_a, head_a, head_vector_a, batch_s, batch_pl,
                           mask_sa, inference_mask=None, r=None, max_num_neighbors=32, mode: Literal['grid', 'heading']='heading'):

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

        # plot edge map
        # if os.getenv('PLOT_EDGE', False):
        #     plot_map_edge(edge_index_pl2sa, pos_s[:, :2], data, save_path='map2sa_edge_map')

        rel_pos_pl2sa = pos_pl[edge_index_pl2sa[0]] - pos_s[edge_index_pl2sa[1]]
        rel_orient_pl2sa = wrap_angle(orient_pl[edge_index_pl2sa[0]] - head_s[edge_index_pl2sa[1]])

        r_pl2sa = torch.stack(
            [torch.norm(rel_pos_pl2sa[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2sa[1]], nbr_vector=rel_pos_pl2sa[:, :2]),
            rel_orient_pl2sa], dim=-1)
        r_pl2sa = self.r_pt2sa_emb(continuous_inputs=r_pl2sa, categorical_embs=None)

        return edge_index_pl2sa, r_pl2sa

    def _build_sa2sa_edge(self, data, pos_a, head_a, state_a, head_vector_a, batch_s, mask, inference_mask=None, **plot_kwargs):

        num_agent = pos_a.shape[0]

        pos_t = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        edge_index_sa2sa = dense_to_sparse(mask_t)[0]
        edge_index_sa2sa = edge_index_sa2sa[:, edge_index_sa2sa[1] - edge_index_sa2sa[0] > 0]
        rel_pos_t = pos_t[edge_index_sa2sa[0]] - pos_t[edge_index_sa2sa[1]]
        rel_head_t = wrap_angle(head_t[edge_index_sa2sa[0]] - head_t[edge_index_sa2sa[1]])

        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_sa2sa[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_sa2sa[0] - edge_index_sa2sa[1]], dim=-1)
        r_sa2sa = self.r_sa2sa_emb(continuous_inputs=r_t, categorical_embs=None)

        return edge_index_sa2sa, r_sa2sa

    def _build_seq(self, device, num_agent, num_step, av_index, sort_indices):
        """
        Args:
            sort_indices (torch.Tensor): shape (num_agent, num_atep)
        """
        # sort_indices = sort_indices[:self.num_seed_feature]
        seq_mask = torch.ones(self.num_seed_feature, num_step, num_agent + self.num_seed_feature, device=device).bool()
        seq_mask[..., -self.num_seed_feature:] = False
        for t in range(num_step):
            for s in range(self.num_seed_feature):
                seq_mask[s, t, sort_indices[s:, t].flatten().long()] = False
        if self.seed_attn_to_av:
            seq_mask[..., av_index] = True
        seq_mask = seq_mask.transpose(0, 1).reshape(-1, num_agent + self.num_seed_feature)

        seq_index = torch.cat([torch.zeros(num_agent), torch.arange(self.num_seed_feature) + 1]).to(device)
        seq_index = seq_index[:, None].repeat(1, num_step)
        for t in range(num_step):
            for s in range(self.num_seed_feature):
                seq_index[sort_indices[s : s + 1, t].flatten().long(), t] = s + 1
        seq_index[av_index] = 0

        return seq_mask, seq_index

    def _build_occ_gt(self, data, seq_mask, pos_rel_index_gt, pos_rel_index_gt_seed, mask_seed,
                      edge_index=None, mode='edge_index'):
        """
        Args:
            seq_mask (torch.Tensor): shape (num_step * num_seed_feature, num_agent + self.num_seed_feature)
            pos_rel_index_gt (torch.Tensor): shape (num_agent, num_step)
            pos_rel_index_gt_seed (torch.Tensor): shape (num_seed, num_step)
        """
        num_agent = data['agent']['state_idx'].shape[0] + self.num_seed_feature
        num_step = data['agent']['state_idx'].shape[1]
        data['agent']['agent_occ'] = torch.zeros(data.num_graphs * self.num_seed_feature, num_step, self.attr_tokenizer.grid_size,
                                                            device=data['agent']['state_idx'].device).long()
        data['agent']['map_occ'] = torch.zeros(data.num_graphs, num_step, self.attr_tokenizer.grid_size,
                                                            device=data['agent']['state_idx'].device).long()

        if mode == 'edge_index':

            assert edge_index is not None, f"Need edge_index input!"
            for src_index in torch.unique(edge_index[1]):
                # decode src
                src_row = src_index % num_agent - (num_agent - self.num_seed_feature)
                src_col = src_index // num_agent
                # decode tgt
                tgt_indexes = edge_index[0, edge_index[1] == src_index]
                tgt_rows = tgt_indexes % num_agent
                tgt_cols = tgt_indexes // num_agent
                assert tgt_rows.max() < num_agent - self.num_seed_feature, f"Invalid {tgt_rows}"
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

        # TODO: fix batch_size!!!
        pt_grid_token_idx = data['agent']['pt_grid_token_idx'] # (t, num_pt)
        for t in range(num_step):
            data['agent']['map_occ'][:, t, pt_grid_token_idx[t][pt_grid_token_idx[t] != -1]] = 1
        data['agent']['map_occ'] = data['agent']['map_occ'].repeat_interleave(repeats=self.num_seed_feature, dim=0)

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        pos_a = data['agent']['token_pos'].clone() # (a, t, 2)
        head_a = data['agent']['token_heading'].clone() # (a, t)
        num_agent, num_step, traj_dim = pos_a.shape # e.g. (50, 18, 2)
        num_pt = data['pt_token']['position'].shape[0]
        agent_category = data['agent']['category'].clone() # (a,)
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
        sort_indices = data['agent']['sort_indices']
        pt_grid_token_idx = data['agent']['pt_grid_token_idx']

        ori_pos_pl = data['pt_token']['position'][:, :self.input_dim].contiguous()
        ori_orient_pl = data['pt_token']['orientation'].contiguous()
        pos_pl = ori_pos_pl.repeat(num_step, 1)
        orient_pl = ori_orient_pl.repeat(num_step)

        # build relative 3d descriptors
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)

        ego_pos_a = ego_pos.repeat_interleave(repeats=data['batch_size_a'], dim=0)
        ego_head_a = ego_head.repeat_interleave(repeats=data['batch_size_a'], dim=0)
        ego_pos_s = ego_pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        ego_head_s = ego_head_a.transpose(0, 1).reshape(-1)
        rel_pos_a2a = pos_s - ego_pos_s
        rel_head_a2a = head_s - ego_head_s

        ego_pos_pl = ego_pos.repeat_interleave(repeats=data['batch_size_pl'], dim=0)
        ego_head_pl = ego_head.repeat_interleave(repeats=data['batch_size_pl'], dim=0)
        ego_pos_s = ego_pos_pl.transpose(0, 1).reshape(-1, self.input_dim)
        ego_head_s = ego_head_pl.transpose(0, 1).reshape(-1)
        rel_pos_pl2a = pos_pl - ego_pos_s
        rel_head_pl2a = orient_pl - ego_head_s

        # releative encodings
        ego_head_vector_a = head_vector_a[av_index].repeat_interleave(repeats=data['batch_size_a'], dim=0)
        ego_head_vector_s = ego_head_vector_a.transpose(0, 1).reshape(-1, 2)
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=ego_head_vector_s, nbr_vector=rel_pos_a2a[:, :2]),
            rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)  # [N, hidden_dim]

        ego_head_vector_a = head_vector_a[av_index].repeat_interleave(repeats=data['batch_size_pl'], dim=0)
        ego_head_vector_s = ego_head_vector_a.transpose(0, 1).reshape(-1, 2)
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=ego_head_vector_s, nbr_vector=rel_pos_pl2a[:, :2]),
            rel_head_pl2a], dim=-1)
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)  # [M, d]

        r_a2a = r_a2a.reshape(num_step, num_agent, -1).transpose(0, 1)
        r_pl2a = r_pl2a.reshape(num_step, num_pt, -1).transpose(0, 1)
        select_agent = torch.randperm(num_agent)[:self.agent_limit]
        select_pt = torch.randperm(num_pt)[:self.pt_limit]
        r_a2a = r_a2a[select_agent]
        r_pl2a = r_pl2a[select_pt]

        # aggregate to global feature
        r_a2a = r_a2a.mean(dim=0)  # [t, d]
        r_pl2a = r_pl2a.mean(dim=0)

        # decode grid index of neighbor agents
        agent_occ = self.grid_agent_occ_head(r_a2a)  # [t, grid_size]
        pt_occ = self.grid_pt_occ_head(r_pl2a)

        # 1.
        # agent_occ_gt = torch.zeros_like(agent_occ).long()
        # pt_occ_gt = torch.zeros_like(pt_occ).long()

        # for t in range(num_step):
        #     agent_occ_gt[t, agent_grid_token_idx[:, t][agent_grid_token_idx[:, t] != -1]] = 1
        #     pt_occ_gt[t, pt_grid_token_idx[t][pt_grid_token_idx[t] != -1]] = 1

        # agent_occ_gt[:, self.grid_size // 2] = 0
        # pt_occ_gt[:, self.grid_size // 2] = 0

        # agent_occ_eval_mask = torch.ones_like(agent_occ_gt)
        # agent_occ_eval_mask[0] = 0
        # agent_occ_eval_mask[:, self.grid_size // 2] = 0
        # pt_occ_eval_mask = torch.ones_like(pt_occ_gt)
        # pt_occ_eval_mask[0] = 0
        # pt_occ_eval_mask[:, self.grid_size // 2] = 0

        # 2.
        # agent_occ_gt = agent_grid_token_idx.transpose(0, 1).reshape(-1)
        # pt_occ_gt = pt_grid_token_idx.reshape(-1)

        # agent_occ_eval_mask = torch.zeros_like(agent_occ_gt)
        # agent_occ_eval_mask[torch.randperm(agent_occ_gt.shape[0])[:(num_step * 10)]] = 1
        # agent_occ_eval_mask[agent_occ_gt == -1] = 0

        # pt_occ_eval_mask = torch.zeros_like(pt_occ_gt)
        # pt_occ_eval_mask[torch.randperm(pt_occ_gt.shape[0])[:(num_step * 300)]] = 1
        # pt_occ_eval_mask[pt_occ_gt == -1] = 0

        # 3.
        agent_occ = agent_occ.reshape(num_step, self.agent_limit, -1)
        pt_occ = pt_occ.reshape(num_step, self.pt_limit, -1)
        agent_occ_gt = agent_grid_token_idx[select_agent].transpose(0, 1)
        pt_occ_gt = pt_grid_token_idx[:, select_pt]
        agent_occ_eval_mask = agent_occ_gt != -1
        pt_occ_eval_mask = pt_occ_gt != -1

        agent_occ = agent_occ[:, :agent_occ_gt.shape[1]]
        pt_occ = pt_occ[:, :pt_occ_gt.shape[1]]

        return {'occ_decoder': True,
                'num_step': num_step,
                'num_agent': self.agent_limit, # num_agent
                'num_pt': self.pt_limit, # num_pt
                'agent_occ': agent_occ,
                'agent_occ_gt': agent_occ_gt,
                'agent_occ_eval_mask': agent_occ_eval_mask.bool(),
                'pt_occ': pt_occ,
                'pt_occ_gt': pt_occ_gt,
                'pt_occ_eval_mask': pt_occ_eval_mask.bool(),
                }

    def inference(self, *args, **kwargs):
        return self(*args, **kwargs)

