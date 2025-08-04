# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import signal
import multiprocessing
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import pickle
import easydict
from functools import partial
from scipy.interpolate import interp1d
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from waymo_open_dataset.protos import scenario_pb2


MIN_VALID_STEPS = 15


_polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
_polygon_light_type = ['LANE_STATE_STOP', 'LANE_STATE_GO', 'LANE_STATE_CAUTION', 'LANE_STATE_UNKNOWN']
_point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'EDGE',
                'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
_polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']


Lane_type_hash = {
    4: "BIKE",
    3: "VEHICLE",
    2: "VEHICLE",
    1: "BUS"
}

boundary_type_hash = {
        5: "UNKNOWN",
        6: "DASHED_WHITE",
        7: "SOLID_WHITE",
        8: "DOUBLE_DASH_WHITE",
        9: "DASHED_YELLOW",
        10: "DOUBLE_DASH_YELLOW",
        11: "SOLID_YELLOW",
        12: "DOUBLE_SOLID_YELLOW",
        13: "DASH_SOLID_YELLOW",
        14: "UNKNOWN",
        15: "EDGE",
        16: "EDGE"
}


def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None


# def get_agent_features(df: pd.DataFrame, av_id, num_historical_steps=11, dim=3, num_steps=91) -> Dict[str, Any]:
#     if args.disable_invalid:  # filter out agents that are unseen during the historical time steps
#         historical_df = df[df['timestep'] == num_historical_steps-1] # extract the timestep==10 (current)
#         agent_ids = list(historical_df['track_id'].unique()) # these agents are seen at timestep==10 (current)
#         df = df[df['track_id'].isin(agent_ids)] # remove other agents
#     else:
#         agent_ids = list(df['track_id'].unique())

#     num_agents = len(agent_ids)
#     # initialization
#     valid_mask              = torch.zeros(num_agents, num_steps, dtype=torch.bool)
#     current_valid_mask      = torch.zeros(num_agents, dtype=torch.bool)
#     predict_mask            = torch.zeros(num_agents, num_steps, dtype=torch.bool)
#     agent_id: List[Optional[str]] = [None] * num_agents
#     agent_type              = torch.zeros(num_agents, dtype=torch.uint8)
#     agent_category          = torch.zeros(num_agents, dtype=torch.uint8)
#     position                = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
#     heading                 = torch.zeros(num_agents, num_steps, dtype=torch.float)
#     velocity                = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
#     shape                   = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)

#     for track_id, track_df in df.groupby('track_id'):
#         agent_idx = agent_ids.index(track_id)
#         all_agent_steps = track_df['timestep'].values
#         valid_agent_steps = all_agent_steps[track_df['validity'].astype(np.bool_)].astype(np.int32)
#         valid_mask[agent_idx, valid_agent_steps] = True
#         current_valid_mask[agent_idx] = valid_mask[agent_idx, num_historical_steps - 1] # current timestep 10
#         if args.disable_invalid:
#             predict_mask[agent_idx, valid_agent_steps] = True
#         else:
#             predict_mask[agent_idx] = True
#         predict_mask[agent_idx, :num_historical_steps] = False
#         if not current_valid_mask[agent_idx]:
#             predict_mask[agent_idx, num_historical_steps:] = False

#         # TODO: why using vector_repr?
#         if vector_repr:  # a time step t is valid only when both t and t-1 are valid
#             valid_mask[agent_idx, 1 : num_historical_steps] = (
#                 valid_mask[agent_idx, : num_historical_steps - 1] &
#                 valid_mask[agent_idx, 1 : num_historical_steps])
#             valid_mask[agent_idx, 0] = False

#         agent_id[agent_idx] = track_id
#         agent_type[agent_idx] = _agent_types.index(track_df['object_type'].values[0])
#         agent_category[agent_idx] = track_df['object_category'].values[0]
#         position[agent_idx, valid_agent_steps, :3] = torch.from_numpy(np.stack([track_df['position_x'].values[valid_agent_steps],
#                                                                                 track_df['position_y'].values[valid_agent_steps],
#                                                                                 track_df['position_z'].values[valid_agent_steps]],
#                                                                         axis=-1)).float()
#         heading[agent_idx, valid_agent_steps] = torch.from_numpy(track_df['heading'].values[valid_agent_steps]).float()
#         velocity[agent_idx, valid_agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values[valid_agent_steps],
#                                                                                 track_df['velocity_y'].values[valid_agent_steps]],
#                                                                         axis=-1)).float()
#         shape[agent_idx, valid_agent_steps, :3] = torch.from_numpy(np.stack([track_df['length'].values[valid_agent_steps],
#                                                                              track_df['width'].values[valid_agent_steps],
#                                                                              track_df["height"].values[valid_agent_steps]],
#                                                                         axis=-1)).float()
#     av_idx = agent_id.index(av_id)
#     if split == 'test':
#         predict_mask[current_valid_mask
#                      | (agent_category == 2)
#                      | (agent_category == 3), num_historical_steps:] = True

#     return {
#         'num_nodes': num_agents,
#         'av_index': av_idx,
#         'valid_mask': valid_mask,
#         'predict_mask': predict_mask,
#         'id': agent_id,
#         'type': agent_type,
#         'category': agent_category,
#         'position': position,
#         'heading': heading,
#         'velocity': velocity,
#         'shape': shape
#     }


def get_agent_features(track_infos: Dict[str, np.ndarray], av_id: int, num_historical_steps: int, num_steps: int) -> Dict[str, Any]:

    agent_idx_to_add = []
    for i in range(len(track_infos['object_id'])):
        is_visible = track_infos['valid'][i, num_historical_steps - 1]

        if split == 'testing':
            assert args.disable_invalid, f'Need to disable invalid for testing!'
            is_valid = True
        else:
            valid_steps = np.where(track_infos['valid'][i])[0]
            valid_start, valid_end = valid_steps[0], valid_steps[-1]
            is_valid = (valid_end - valid_start + 1) >= MIN_VALID_STEPS

        if (is_visible or not args.disable_invalid) and is_valid:
            agent_idx_to_add.append(i)

    num_agents = len(agent_idx_to_add)
    out_dict = {
        'num_nodes':  num_agents,
        'valid_mask': torch.zeros(num_agents, num_steps, dtype=torch.bool),
        'role':       torch.zeros(num_agents, 3, dtype=torch.bool),
        'id':         torch.zeros(num_agents, dtype=torch.int64) - 1,
        'type':       torch.zeros(num_agents, dtype=torch.uint8),
        'category':   torch.zeros(num_agents, dtype=torch.uint8),
        'position':   torch.zeros(num_agents, num_steps, 3, dtype=torch.float),
        'heading':    torch.zeros(num_agents, num_steps, dtype=torch.float),
        'velocity':   torch.zeros(num_agents, num_steps, 2, dtype=torch.float),
        'shape':      torch.zeros(num_agents, num_steps, 3, dtype=torch.float),
    }

    for i, idx in enumerate(agent_idx_to_add):

        out_dict['role'][i] = torch.from_numpy(track_infos['role'][idx])
        out_dict['id'][i] = track_infos['object_id'][idx]
        out_dict['type'][i] = track_infos['object_type'][idx]
        out_dict['category'][i] = idx in track_infos['tracks_to_predict']

        valid = track_infos["valid"][idx]  # [n_step]
        states = track_infos["states"][idx]

        object_shape = states[:, 3:6]  # [n_step, 3], length, width, height
        object_shape = object_shape[valid].mean(axis=0)  # [3]
        out_dict["shape"][i] = torch.from_numpy(object_shape)

        valid_steps = np.where(valid)[0]
        position = states[:, :3]  # [n_step, dim], x, y, z
        velocity = states[:, 7:9]  # [n_step, 2], vx, vy
        heading = states[:, 6]  # [n_step], heading

        if split == 'testing':

            out_dict['valid_mask'][i, valid_steps] = True
            out_dict['position'][i, valid_steps] = torch.from_numpy(position[valid_steps])
            out_dict['velocity'][i, valid_steps] = torch.from_numpy(velocity[valid_steps])
            out_dict['heading'][i, valid_steps] = torch.from_numpy(heading[valid_steps])

        else:

            # valid.sum() should > 1:
            t_start, t_end = valid_steps[0], valid_steps[-1]
            f_pos = interp1d(valid_steps, position[valid], axis=0)
            f_vel = interp1d(valid_steps, velocity[valid], axis=0)
            f_yaw = interp1d(valid_steps, np.unwrap(heading[valid], axis=0), axis=0)
            t_in = np.arange(t_start, t_end + 1)
            out_dict["valid_mask"][i, t_start : t_end + 1] = True
            out_dict["position"][i, t_start : t_end + 1] = torch.from_numpy(f_pos(t_in))
            out_dict["velocity"][i, t_start : t_end + 1] = torch.from_numpy(f_vel(t_in))
            out_dict["heading"][i, t_start : t_end + 1] = torch.from_numpy(f_yaw(t_in))

    out_dict['av_idx'] = out_dict['id'].tolist().index(av_id)

    return out_dict


def get_map_features(map_infos, tf_current_light, dim=3):
    lane_segments = map_infos['lane']
    all_polylines = map_infos["all_polylines"]
    crosswalks = map_infos['crosswalk']
    road_edges = map_infos['road_edge']
    road_lines = map_infos['road_line']
    lane_segment_ids = [info["id"] for info in lane_segments]
    cross_walk_ids = [info["id"] for info in crosswalks]
    road_edge_ids = [info["id"] for info in road_edges]
    road_line_ids = [info["id"] for info in road_lines]
    polygon_ids = lane_segment_ids + road_edge_ids + road_line_ids + cross_walk_ids
    num_polygons = len(lane_segment_ids) + len(road_edge_ids) + len(road_line_ids) + len(cross_walk_ids)

    # initialization
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_light_type = torch.ones(num_polygons, dtype=torch.uint8) * 3

    # list of (num_of_segments,), each element has shape of (num_of_points_of_current_segment - 1, dim)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type: List[Optional[torch.Tensor]] = [None] * num_polygons

    for lane_segment in lane_segments:
        lane_segment = easydict.EasyDict(lane_segment)
        lane_segment_idx = polygon_ids.index(lane_segment.id)
        polyline_index = lane_segment.polyline_index # (start index of point in current scenario, end index of point in current scenario)
        centerline = all_polylines[polyline_index[0] : polyline_index[1], :] # （num_of_points_of_current_segment, 5)
        centerline = torch.from_numpy(centerline).float()
        polygon_type[lane_segment_idx] = _polygon_types.index(Lane_type_hash[lane_segment.type])

        res = tf_current_light[tf_current_light["lane_id"] == str(lane_segment.id)]
        if len(res) != 0:
            polygon_light_type[lane_segment_idx] = _polygon_light_type.index(res["state"].item())

        point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0) # (num_of_points_of_current_segment - 1, 3)
        center_vectors = centerline[1:] - centerline[:-1] # (num_of_points_of_current_segment - 1, 5)
        point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0) # (num_of_points_of_current_segment - 1,)
        point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1) # (num_of_points_of_current_segment - 1,)
        point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0) # (num_of_points_of_current_segment - 1,)
        center_type = _point_types.index('CENTERLINE')
        point_type[lane_segment_idx] = torch.cat(
            [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)

    for lane_segment in road_edges:
        lane_segment = easydict.EasyDict(lane_segment)
        lane_segment_idx = polygon_ids.index(lane_segment.id)
        polyline_index = lane_segment.polyline_index
        centerline = all_polylines[polyline_index[0] : polyline_index[1], :]
        centerline = torch.from_numpy(centerline).float()
        polygon_type[lane_segment_idx] = _polygon_types.index("VEHICLE")

        point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
        center_vectors = centerline[1:] - centerline[:-1]
        point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
        point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
        point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0)
        center_type = _point_types.index('EDGE')
        point_type[lane_segment_idx] = torch.cat(
            [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)

    for lane_segment in road_lines:
        lane_segment = easydict.EasyDict(lane_segment)
        lane_segment_idx = polygon_ids.index(lane_segment.id)
        polyline_index = lane_segment.polyline_index
        centerline = all_polylines[polyline_index[0] : polyline_index[1], :]
        centerline = torch.from_numpy(centerline).float()

        polygon_type[lane_segment_idx] = _polygon_types.index("VEHICLE")

        point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
        center_vectors = centerline[1:] - centerline[:-1]
        point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
        point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
        point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0)
        center_type = _point_types.index(boundary_type_hash[lane_segment.type])
        point_type[lane_segment_idx] = torch.cat(
            [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)

    for crosswalk in crosswalks:
        crosswalk = easydict.EasyDict(crosswalk)
        lane_segment_idx = polygon_ids.index(crosswalk.id)
        polyline_index = crosswalk.polyline_index
        centerline = all_polylines[polyline_index[0] : polyline_index[1], :]
        centerline = torch.from_numpy(centerline).float()

        polygon_type[lane_segment_idx] = _polygon_types.index("PEDESTRIAN")

        point_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
        center_vectors = centerline[1:] - centerline[:-1]
        point_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
        point_magnitude[lane_segment_idx] = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
        point_height[lane_segment_idx] = torch.cat([center_vectors[:, 2]], dim=0)
        center_type = _point_types.index("CROSSWALK")
        point_type[lane_segment_idx] = torch.cat(
            [torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)

    # (num_of_segments,), each element represents the number of points of the segment
    num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
    # (2, total_num_of_points_of_all_segments), store the point index of segment and its corresponding segment index
    # e.g. a scenario has 203 segments, and totally 14039 points:
    # tensor([[    0,     1,     2,  ..., 14927, 14928, 14929],
    #         [    0,     0,     0,  ...,   202,   202,   202]]) => polygon_ids.index(lane_segment.id)
    point_to_polygon_edge_index = torch.stack(
        [torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
    # list of (num_of_lane_segments,)
    polygon_to_polygon_edge_index = []
    # list of (num_of_lane_segments,)
    polygon_to_polygon_type = []
    for lane_segment in lane_segments:
        lane_segment = easydict.EasyDict(lane_segment)
        lane_segment_idx = polygon_ids.index(lane_segment.id)
        pred_inds = []
        for pred in lane_segment.entry_lanes:
            pred_idx = safe_list_index(polygon_ids, pred)
            if pred_idx is not None:
                pred_inds.append(pred_idx)
        if len(pred_inds) != 0:
            polygon_to_polygon_edge_index.append(
                torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                             torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
            polygon_to_polygon_type.append(
                torch.full((len(pred_inds),), _polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
        succ_inds = []
        for succ in lane_segment.exit_lanes:
            succ_idx = safe_list_index(polygon_ids, succ)
            if succ_idx is not None:
                succ_inds.append(succ_idx)
        if len(succ_inds) != 0:
            polygon_to_polygon_edge_index.append(
                torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                             torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
            polygon_to_polygon_type.append(
                torch.full((len(succ_inds),), _polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
        if len(lane_segment.left_neighbors) != 0:
            left_neighbor_ids = lane_segment.left_neighbors
            for left_neighbor_id in left_neighbor_ids:
                left_idx = safe_list_index(polygon_ids, left_neighbor_id)
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([_polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
        if len(lane_segment.right_neighbors) != 0:
            right_neighbor_ids = lane_segment.right_neighbors
            for right_neighbor_id in right_neighbor_ids:
                right_idx = safe_list_index(polygon_ids, right_neighbor_id)
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([_polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))
    if len(polygon_to_polygon_edge_index) != 0:
        polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
        polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
    else:
        polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
        polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

    map_data = {
        'map_polygon': {},
        'map_point': {},
        ('map_point', 'to', 'map_polygon'): {},
        ('map_polygon', 'to', 'map_polygon'): {},
    }
    map_data['map_polygon']['num_nodes'] = num_polygons # int, number of map segments in the scenario
    map_data['map_polygon']['type'] = polygon_type # (num_polygons,) type of each polygon
    map_data['map_polygon']['light_type'] = polygon_light_type # (num_polygons,) light type of each polygon, 3 means unknown
    if len(num_points) == 0:
        map_data['map_point']['num_nodes'] = 0
        map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
        if dim == 3:
            map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
        map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
    else:
        map_data['map_point']['num_nodes'] = num_points.sum().item() # int, number of total points of all segments in the scenario
        map_data['map_point']['position'] = torch.cat(point_position, dim=0) # (num_of_total_points_of_all_segments, 3)
        map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0) # (num_of_total_points_of_all_segments,)
        map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0) # (num_of_total_points_of_all_segments,)
        if dim == 3:
            map_data['map_point']['height'] = torch.cat(point_height, dim=0) # (num_of_total_points_of_all_segments,)
        map_data['map_point']['type'] = torch.cat(point_type, dim=0) # (num_of_total_points_of_all_segments,) type of point => `_point_types`
    map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index # (2, num_of_total_points_of_all_segments)
    map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

    if int(os.getenv('DEBUG_MAP', 1)):
        import matplotlib.pyplot as plt
        plt.axis('equal')
        plt.scatter(map_data['map_point']['position'][:, 0],
                    map_data['map_point']['position'][:, 1], s=0.2, c='black', edgecolors='none')
        plt.savefig("debug.png", dpi=600)

    return map_data


# def process_agent(track_info, tracks_to_predict, scenario_id, start_timestamp, end_timestamp):

#     agents_array = track_info["states"].transpose(1, 0, 2) # (num_timesteps, num_agents, 10) e.g. (91, 15, 10)
#     object_id = np.array(track_info["object_id"]) # (num_agents,) global id of each agent
#     object_type = track_info["object_type"] # (num_agents,) type of each agent, e.g. 'TYPE_VEHICLE'
#     id_hash = {object_id[o_idx]: object_type[o_idx] for o_idx in range(len(object_id))}

#     def type_hash(x):
#         tp = id_hash[x]
#         type_re_hash = {
#             "TYPE_VEHICLE": "vehicle",
#             "TYPE_PEDESTRIAN": "pedestrian",
#             "TYPE_CYCLIST": "cyclist",
#             "TYPE_OTHER": "background",
#             "TYPE_UNSET": "background"
#         }
#         return type_re_hash[tp]

#     columns = ['observed', 'track_id', 'object_type', 'object_category', 'timestep',
#                'position_x', 'position_y', 'position_z', 'length', 'width', 'height', 'heading', 'velocity_x', 'velocity_y',
#                'scenario_id', 'start_timestamp', 'end_timestamp', 'num_timestamps',
#                'focal_track_id', 'city', 'validity']

#     # (num_timesteps, num_agents, 10) e.g. (91, 15, 10)
#     new_columns = np.ones((agents_array.shape[0], agents_array.shape[1], 11))
#     new_columns[:11, :, 0] = True # observed, 10 timesteps
#     new_columns[11:, :, 0] = False # not observed (current + future) 
#     for index in range(new_columns.shape[0]):
#         new_columns[index, :, 4] = int(index) # timestep (0 ~ 90)
#     new_columns[..., 1] = object_id
#     new_columns[..., 2] = object_id
#     new_columns[:, tracks_to_predict['track_index'], 3] = 3
#     new_columns[..., 5] = 11
#     new_columns[..., 6] = int(start_timestamp) # 0
#     new_columns[..., 7] = int(end_timestamp)   # 91
#     new_columns[..., 8] = int(91)              # 91
#     new_columns[..., 9] = object_id
#     new_columns[..., 10] = 10086
#     new_columns = new_columns
#     new_agents_array = np.concatenate([new_columns, agents_array], axis=-1) # (num_timesteps, num_agents, 21) e.g. (91, 15, 21)
#     # filter out the invalid timestep of agents, reshape to (num_valid_of_timesteps_of_all_agents, 21) e.g. (91, 15, 21) -> (1137, 21)
#     if args.disable_invalid:
#         new_agents_array = new_agents_array[new_agents_array[..., -1] == 1.0].reshape(-1, new_agents_array.shape[-1])
#     else:
#         agent_valid_mask = new_agents_array[..., -1] # (num_timesteps, num_agents)
#         agent_mask = np.sum(agent_valid_mask, axis=0) > MIN_VALID_STEPS # NOTE: 10 is a empirical parameter
#         new_agents_array = new_agents_array[:, agent_mask]
#         new_agents_array = new_agents_array.reshape(-1, new_agents_array.shape[-1]) # (91, 15, 21) -> (1365, 21)
#     new_agents_array = new_agents_array[..., [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 5, 6, 7, 8, 9, 10, 20]]
#     new_agents_array = pd.DataFrame(data=new_agents_array, columns=columns)
#     new_agents_array["object_type"] = new_agents_array["object_type"].apply(func=type_hash)
#     new_agents_array["start_timestamp"] = new_agents_array["start_timestamp"].astype(int)
#     new_agents_array["end_timestamp"] = new_agents_array["end_timestamp"].astype(int)
#     new_agents_array["num_timestamps"] = new_agents_array["num_timestamps"].astype(int)
#     new_agents_array["scenario_id"] = scenario_id

#     return new_agents_array


def process_dynamic_map(dynamic_map_infos):
    lane_ids = dynamic_map_infos["lane_id"]
    tf_lights = []
    for t in range(len(lane_ids)):
        lane_id = lane_ids[t]
        time = np.ones_like(lane_id) * t
        state = dynamic_map_infos["state"][t]
        tf_light = np.concatenate([lane_id, time, state], axis=0)
        tf_lights.append(tf_light)
    tf_lights = np.concatenate(tf_lights, axis=1).transpose(1, 0)
    tf_lights = pd.DataFrame(data=tf_lights, columns=["lane_id", "time_step", "state"])
    tf_lights["time_step"] = tf_lights["time_step"].astype("str")
    tf_lights["lane_id"] = tf_lights["lane_id"].astype("str")
    tf_lights["state"] = tf_lights["state"].astype("str")
    tf_lights.loc[tf_lights["state"].str.contains("STOP"), ["state"]] = (
        "LANE_STATE_STOP"
    )
    tf_lights.loc[tf_lights["state"].str.contains("GO"), ["state"]] = "LANE_STATE_GO"
    tf_lights.loc[tf_lights["state"].str.contains("CAUTION"), ["state"]] = (
        "LANE_STATE_CAUTION"
    )
    tf_lights.loc[tf_lights["state"].str.contains("UNKNOWN"), ["state"]] = (
        "LANE_STATE_UNKNOWN"
    )

    return tf_lights


polyline_type = {
    # for lane
    'TYPE_UNDEFINED': -1,
    'TYPE_FREEWAY': 1,
    'TYPE_SURFACE_STREET': 2,
    'TYPE_BIKE_LANE': 3,

    # for roadline
    'TYPE_UNKNOWN': -1,
    'TYPE_BROKEN_SINGLE_WHITE': 6,
    'TYPE_SOLID_SINGLE_WHITE': 7,
    'TYPE_SOLID_DOUBLE_WHITE': 8,
    'TYPE_BROKEN_SINGLE_YELLOW': 9,
    'TYPE_BROKEN_DOUBLE_YELLOW': 10,
    'TYPE_SOLID_SINGLE_YELLOW': 11,
    'TYPE_SOLID_DOUBLE_YELLOW': 12,
    'TYPE_PASSING_DOUBLE_YELLOW': 13,

    # for roadedge
    'TYPE_ROAD_EDGE_BOUNDARY': 15,
    'TYPE_ROAD_EDGE_MEDIAN': 16,

    # for stopsign
    'TYPE_STOP_SIGN': 17,

    # for crosswalk
    'TYPE_CROSSWALK': 18,

    # for speed bump
    'TYPE_SPEED_BUMP': 19
}

object_type = {
    0: 'TYPE_UNSET',
    1: 'TYPE_VEHICLE',
    2: 'TYPE_PEDESTRIAN',
    3: 'TYPE_CYCLIST',
    4: 'TYPE_OTHER'
}


def decode_tracks_from_proto(scenario):
    sdc_track_index = scenario.sdc_track_index
    track_index_predict = [i.track_index for i in scenario.tracks_to_predict]
    object_id_interest = [i for i in scenario.objects_of_interest]

    track_infos = {
        'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        'object_type': [],
        'states': [],
        'valid': [],
        'role': [],
    }

    # tracks mean N number of objects, e.g. len(tracks) = 55
    # each track has 91 states, e.g. len(tracks[0].states) == 91
    # each state has 10 attributes: center_x, center_y, center_z, length, ..., velocity_y, valid
    for i, cur_data in enumerate(scenario.tracks):

        step_state = []
        step_valid = []

        for s in cur_data.states:  # n_steps
            step_state.append(
                [
                    s.center_x,
                    s.center_y,
                    s.center_z,
                    s.length,
                    s.width,
                    s.height,
                    s.heading,
                    s.velocity_x,
                    s.velocity_y,
                ]
            )
            step_valid.append(s.valid)
            # This angle is normalized to [-pi, pi). The velocity vector in m/s

        track_infos['object_id'].append(cur_data.id) # id of object in this track
        track_infos['object_type'].append(cur_data.object_type - 1)
        track_infos['states'].append(np.array(step_state, dtype=np.float32))
        track_infos['valid'].append(np.array(step_valid))

        track_infos['role'].append([False, False, False])
        if i in track_index_predict:
            track_infos['role'][-1][2] = True  # predict=2
        if cur_data.id in object_id_interest:
            track_infos['role'][-1][1] = True  # interest=1
        if i == sdc_track_index:
            track_infos['role'][-1][0] = True  # ego_vehicle=0

    track_infos['states']   = np.array(track_infos['states'], dtype=np.float32)  # (n_agent, n_step, 9)
    track_infos['valid']    = np.array(track_infos['valid'], dtype=np.bool_)
    track_infos['role']     = np.array(track_infos['role'], dtype=np.bool_)
    track_infos['object_id'] = np.array(track_infos['object_id'], dtype=np.int64)
    track_infos['object_type'] = np.array(track_infos['object_type'], dtype=np.uint8)
    track_infos['tracks_to_predict'] = np.array(track_index_predict, dtype=np.int64)

    return track_infos


from collections import defaultdict

def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': [],
        'lane_dict': {},
        'lane2other_dict': {}
    }
    polylines = []

    point_cnt = 0
    lane2other_dict = defaultdict(list)

    for cur_data in map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = cur_data.lane.type + 1  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane
            cur_info['left_neighbors'] = [lane.feature_id for lane in cur_data.lane.left_neighbors]

            cur_info['right_neighbors'] = [lane.feature_id for lane in cur_data.lane.right_neighbors]

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary_type'] = [x.boundary_type + 5 for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary_type'] = [x.boundary_type + 5 for x in cur_data.lane.right_boundaries]

            cur_info['left_boundary'] = [x.boundary_feature_id for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary'] = [x.boundary_feature_id for x in cur_data.lane.right_boundaries]
            cur_info['left_boundary_start_index'] = [lane.lane_start_index for lane in cur_data.lane.left_boundaries]
            cur_info['left_boundary_end_index'] = [lane.lane_end_index for lane in cur_data.lane.left_boundaries]
            cur_info['right_boundary_start_index'] = [lane.lane_start_index for lane in cur_data.lane.right_boundaries]
            cur_info['right_boundary_end_index'] = [lane.lane_end_index for lane in cur_data.lane.right_boundaries]

            lane2other_dict[cur_data.id].extend(cur_info['left_boundary'])
            lane2other_dict[cur_data.id].extend(cur_info['right_boundary'])

            global_type = cur_info['type']
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in cur_data.lane.polyline],
                axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['lane'].append(cur_info)
            map_infos['lane_dict'][cur_data.id] = cur_info

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = cur_data.road_line.type + 5

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.road_line.polyline], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['road_line'].append(cur_info) # (num_points, 5)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = cur_data.road_edge.type + 14

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.road_edge.polyline], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            for i in cur_info['lane_ids']:
                lane2other_dict[i].append(cur_data.id)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, global_type, cur_data.id]).reshape(1, 5)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.crosswalk.polygon], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.speed_bump.polygon], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['speed_bump'].append(cur_info)

        else:
            continue

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline)) # (start index of point in current scenario, end index of point in current scenario)
        point_cnt += len(cur_polyline)

    polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    map_infos['all_polylines'] = polylines # (num_of_total_points_in_current_scenario, 5)
    map_infos['lane2other_dict'] = lane2other_dict
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):

    signal_state = {
        0: 'LANE_STATE_UNKNOWN',
        # States for traffic signals with arrows.
        1: 'LANE_STATE_ARROW_STOP',
        2: 'LANE_STATE_ARROW_CAUTION',
        3: 'LANE_STATE_ARROW_GO',
        # Standard round traffic signals.
        4: 'LANE_STATE_STOP',
        5: 'LANE_STATE_CAUTION',
        6: 'LANE_STATE_GO',
        # Flashing light signals.
        7: 'LANE_STATE_FLASHING_STOP',
        8: 'LANE_STATE_FLASHING_CAUTION'
    }

    dynamic_map_infos = {
        'lane_id': [],
        'state': [],
        'stop_point': []
    }
    for cur_data in dynamic_map_states:  # len(dynamic_map_states) = num_timestamp
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos['lane_id'].append(np.array([lane_id]))
        dynamic_map_infos['state'].append(np.array([state]))
        dynamic_map_infos['stop_point'].append(np.array([stop_point]))

    return dynamic_map_infos


# def process_single_data(scenario):
#     info = {}
#     info['scenario_id'] = scenario.scenario_id
#     info['timestamps_seconds'] = list(scenario.timestamps_seconds)  # list of int of shape (91)
#     info['current_time_index'] = scenario.current_time_index  # int, 10
#     info['sdc_track_index'] = scenario.sdc_track_index  # int
#     info['objects_of_interest'] = list(scenario.objects_of_interest)  # list, could be empty list

#     info['tracks_to_predict'] = {
#         'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
#         'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
#     }  # for training: suggestion of objects to train on, for val/test: need to be predicted

#     # decode tracks data
#     track_infos = decode_tracks_from_proto(scenario.tracks)
#     info['tracks_to_predict']['object_type'] = [track_infos['object_type'][cur_idx] for cur_idx in
#                                                 info['tracks_to_predict']['track_index']]
#     # decode map related data
#     map_infos = decode_map_features_from_proto(scenario.map_features)
#     dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

#     save_infos = {
#         'track_infos': track_infos,
#         'map_infos': map_infos,
#         'dynamic_map_infos': dynamic_map_infos,
#     }
#     save_infos.update(info)
#     return save_infos


def wm2argo(file, input_dir, output_dir, existing_files=[], output_dir_tfrecords_splitted=None):
    file_path = os.path.join(input_dir, file)
    dataset = tf.data.TFRecordDataset(file_path, compression_type='', num_parallel_reads=3)

    for cnt, tf_data in tqdm(enumerate(dataset), leave=False, desc=f'Process {file}...'):

        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(tf_data.numpy()))
        scenario_id = scenario.scenario_id
        tqdm.write(f"idx: {cnt}, scenario_id: {scenario_id} of {file}")

        if f'{scenario_id}.pkl' not in existing_files:

            map_infos = decode_map_features_from_proto(scenario.map_features)
            track_infos = decode_tracks_from_proto(scenario)
            dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)
            sdc_track_index = scenario.sdc_track_index  # int
            av_id = track_infos['object_id'][sdc_track_index]
            # if len(track_infos['tracks_to_predict']) < 1:
            #     return

            current_time_index = scenario.current_time_index
            tf_lights = process_dynamic_map(dynamic_map_infos)
            tf_current_light = tf_lights.loc[tf_lights["time_step"] == current_time_index] # 10 (history) + 1 (current) + 80 (future)
            map_data = get_map_features(map_infos, tf_current_light)

            # new_agents_array = process_agent(track_infos, tracks_to_predict, scenario_id, 0, 91) # mtr2argo
            data = dict()
            data.update(map_data)
            data['scenario_id'] = scenario_id
            data['agent'] = get_agent_features(track_infos, av_id, num_historical_steps=current_time_index + 1, num_steps=91)

            with open(os.path.join(output_dir, f'{scenario_id}.pkl'), "wb+") as f:
                pickle.dump(data, f)

        if output_dir_tfrecords_splitted is not None:
            tf_file = os.path.join(output_dir_tfrecords_splitted, f'{scenario_id}.tfrecords')
            if not os.path.exists(tf_file):
                with tf.io.TFRecordWriter(tf_file) as file_writer:
                    file_writer.write(tf_data.numpy())


def batch_process9s_transformer(input_dir, output_dir, num_workers=2):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    output_dir_tfrecords_splitted = None
    if split == "validation":
        output_dir_tfrecords_splitted = os.path.join(output_dir, 'validation_tfrecords_splitted')
        os.makedirs(output_dir_tfrecords_splitted, exist_ok=True)

    input_dir = os.path.join(input_dir, split)
    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    packages = sorted(os.listdir(input_dir))
    existing_files = sorted(os.listdir(output_dir))
    func = partial(
        wm2argo,
        output_dir=output_dir,
        input_dir=input_dir,
        existing_files=existing_files,
        output_dir_tfrecords_splitted=output_dir_tfrecords_splitted
    )
    try:
        with multiprocessing.Pool(num_workers, maxtasksperchild=10) as p:
            r = list(tqdm(p.imap_unordered(func, packages), total=len(packages)))
    except KeyboardInterrupt:
        p.terminate()
        p.join()


def generate_meta_infos(data_dir):
    import json

    meta_infos = dict()

    for split in tqdm(['training', 'validation', 'test'], leave=False):
        if not os.path.exists(os.path.join(data_dir, split)):
            continue

        split_infos = dict()
        files = os.listdir(os.path.join(data_dir, split))
        for file in tqdm(files, leave=False):
            try:
                data = pickle.load(open(os.path.join(data_dir, split, file), 'rb'))
            except Exception as e:
                tqdm.write(f'Failed to load scenario {file} due to {e}')
                continue
            scenario_infos = dict(num_agents=data['agent']['num_nodes'])
            scenario_id = data['scenario_id']
            split_infos[scenario_id] = scenario_infos

        meta_infos[split] = split_infos

    with open(os.path.join(data_dir, 'meta_infos.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_infos, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/waymo/')
    parser.add_argument('--output_dir', type=str, default='data/waymo_processed/')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--no_batch', action='store_true')
    parser.add_argument('--disable_invalid', action="store_true")
    parser.add_argument('--generate_meta_infos', action="store_true")
    args = parser.parse_args()

    split = args.split

    if args.generate_meta_infos:
        generate_meta_infos(args.output_dir)

    elif args.no_batch:

        output_dir_tfrecords_splitted = None
        if args.split == "validation":
            output_dir_tfrecords_splitted = os.path.join(args.output_dir, 'validation_tfrecords_splitted')
            os.makedirs(output_dir_tfrecords_splitted, exist_ok=True)

        input_dir = os.path.join(args.input_dir, args.split)
        output_dir = os.path.join(args.output_dir, args.split)
        os.makedirs(output_dir, exist_ok=True)

        files = sorted(os.listdir(input_dir))
        os.makedirs(args.output_dir, exist_ok=True)
        for file in tqdm(files, leave=False, desc=f'Process {args.split}...'):
            wm2argo(file, input_dir, output_dir, output_dir_tfrecords_splitted)

    else:

        batch_process9s_transformer(args.input_dir, args.output_dir, num_workers=96)
