# ! Metrics Calculation
import concurrent.futures
import os
import torch
import tensorflow as tf
import collections
import dataclasses
import fnmatch
import json
import pandas as pd
import pickle
import copy
import concurrent
import multiprocessing
from torch_geometric.utils import degree
from functools import partial
from dataclasses import dataclass, field
from tqdm import tqdm
from argparse import ArgumentParser
from torch import Tensor
from google.protobuf import text_format
from torchmetrics import Metric
from typing import Optional, Sequence, List, Dict, Tuple

from waymo_open_dataset.utils.sim_agents import submission_specs

from infgen.utils.visualization import safe_run
from infgen.utils.func import CONSOLE
from infgen.datasets.scalable_dataset import WaymoTargetBuilder
from infgen.datasets.preprocess import TokenProcessor, SHIFT, AGENT_STATE
from infgen.metrics import trajectory_features, interact_features, map_features, placement_features
from infgen.metrics.protos import scenario_pb2, long_metrics_pb2


_METRIC_FIELD_NAMES_BY_BUCKET = {
    'kinematic': [
        'linear_speed', 'linear_acceleration',
        'angular_speed', 'angular_acceleration',
    ],
    'interactive': [
        'distance_to_nearest_object', 'collision_indication',
        'time_to_collision',
    ],
    'map_based': [
        # 'distance_to_road_edge', 'offroad_indication'
    ],
    'placement_based': [
        'num_placement', 'num_removement',
        'distance_placement', 'distance_removement',
    ]
}
_METRIC_FIELD_NAMES = (
    _METRIC_FIELD_NAMES_BY_BUCKET['kinematic'] +
    _METRIC_FIELD_NAMES_BY_BUCKET['interactive'] +
    _METRIC_FIELD_NAMES_BY_BUCKET['map_based'] +
    _METRIC_FIELD_NAMES_BY_BUCKET['placement_based']
)


""" Help Functions """

def _arg_gather(tensor: Tensor, reference_tensor: Tensor) -> Tensor:
    """Finds corresponding indices in `tensor` for each element in `reference_tensor`.

    Args:
        tensor: A 1D tensor without repetitions.
        reference_tensor: A 1D tensor containing items from `tensor`.

    Returns:
        A tensor of indices such that `tensor[indices] == reference_tensor`.
    """
    assert tensor.ndim == 1, "tensor must be 1D"
    assert reference_tensor.ndim == 1, "reference_tensor must be 1D"
    
    # Create the comparison matrix
    bit_mask = tensor[None, :] == reference_tensor[:, None]  # Shape: [len(reference_tensor), len(tensor)]
    
    # Count the matches along `tensor` dimension
    bit_mask_sum = bit_mask.int().sum(dim=1)
    
    if (bit_mask_sum < 1).any():
        raise ValueError(
            'Some items in `reference_tensor` are missing from `tensor`: '
            f'\n{reference_tensor} \nvs. \n{tensor}.'
        )
    
    if (bit_mask_sum > 1).any():
        raise ValueError('Some items in `tensor` are repeated.')
    
    # Compute indices
    indices = torch.matmul(bit_mask.int(), torch.arange(tensor.shape[0], dtype=torch.int32))
    return indices


def is_valid_sim_agent(track: scenario_pb2.Track) -> bool: # type: ignore
    """Checks if the object needs to be resimulated as a sim agent.

    For the Sim Agents challenge, every object that is valid at the
    `current_time_index` step (here hardcoded to 10) needs to be resimulated.

    Args:
    track: A track proto for a single object.

    Returns:
    A boolean flag, True if the object needs to be resimulated, False otherwise.
    """
    return track.states[submission_specs.CURRENT_TIME_INDEX].valid


def get_sim_agent_ids(
    scenario: scenario_pb2.Scenario) -> Sequence[int]: # type: ignore
    """Returns the list of object IDs that needs to be resimulated.

    Internally calls `is_valid_sim_agent` to verify the simulation criteria,
    i.e. is the object valid at `current_time_index`.

    Args:
    scenario: The Scenario proto containing the data.

    Returns:
    A list of int IDs, containing all the objects that need to be simulated.
    """
    object_ids = []
    for track in scenario.tracks:
        if is_valid_sim_agent(track):
            object_ids.append(track.id)
    return object_ids


def get_evaluation_agent_ids(
    scenario: scenario_pb2.Scenario) -> Sequence[int]: # type: ignore
    # Start with the AV object.
    object_ids = {scenario.tracks[scenario.sdc_track_index].id}
    # Add the `tracks_to_predict` objects.
    for required_prediction in scenario.tracks_to_predict:
        object_ids.add(scenario.tracks[required_prediction.track_index].id)
    return sorted(object_ids)


""" Base Data Classes s"""

@dataclass(frozen=True)
class ObjectTrajectories:

    x: Tensor
    y: Tensor
    z: Tensor
    heading: Tensor
    length: Tensor
    width: Tensor
    height: Tensor
    valid: Tensor
    object_id: Tensor
    object_type: Tensor

    state: Optional[Tensor] = None
    token_pos: Optional[Tensor] = None
    token_heading: Optional[Tensor] = None
    token_valid: Optional[Tensor] = None
    processed_object_id: Optional[Tensor] = None
    av_id: Optional[int] = None
    processed_av_id: Optional[int] = None

    def slice_time(self, start_index: int = 0, end_index: Optional[int] = None):
        return ObjectTrajectories(
            x=self.x[..., start_index:end_index],
            y=self.y[..., start_index:end_index],
            z=self.z[..., start_index:end_index],
            heading=self.heading[..., start_index:end_index],
            length=self.length[..., start_index:end_index],
            width=self.width[..., start_index:end_index],
            height=self.height[..., start_index:end_index],
            valid=self.valid[..., start_index:end_index],
            object_id=self.object_id,
            object_type=self.object_type,

            # these properties can only come from processed file
            state=self.state,
            token_pos=self.token_pos,
            token_heading=self.token_heading,
            token_valid=self.token_valid,
            processed_object_id=self.processed_object_id,
            av_id=self.av_id,
            processed_av_id=self.processed_av_id,
        )

    def gather_objects(self, object_indices: Tensor):
        assert object_indices.ndim == 1, "object_indices must be 1D"
        return ObjectTrajectories(
            x=torch.index_select(self.x, dim=-2, index=object_indices),
            y=torch.index_select(self.y, dim=-2, index=object_indices),
            z=torch.index_select(self.z, dim=-2, index=object_indices),
            heading=torch.index_select(self.heading, dim=-2, index=object_indices),
            length=torch.index_select(self.length, dim=-2, index=object_indices),
            width=torch.index_select(self.width, dim=-2, index=object_indices),
            height=torch.index_select(self.height, dim=-2, index=object_indices),
            valid=torch.index_select(self.valid, dim=-2, index=object_indices),
            object_id=torch.index_select(self.object_id, dim=-1, index=object_indices),
            object_type=torch.index_select(self.object_type, dim=-1, index=object_indices),

            # these properties can only come from processed file
            state=self.state,
            token_pos=self.token_pos,
            token_heading=self.token_heading,
            token_valid=self.token_valid,
            processed_object_id=self.processed_object_id,
            av_id=self.av_id,
            processed_av_id=self.processed_av_id,
        )

    def gather_objects_by_id(self, object_ids: tf.Tensor):
        indices = _arg_gather(self.object_id, object_ids)
        return self.gather_objects(indices)

    @classmethod
    def _get_init_dict_from_processed(cls, scenario: dict):
        """Load from processed pkl data"""
        position = scenario['agent']['position']
        heading = scenario['agent']['heading']
        shape = scenario['agent']['shape']
        object_ids = scenario['agent']['id']
        object_types = scenario['agent']['type']
        valid = scenario['agent']['valid_mask']

        init_dict = dict(x=position[..., 0],
                         y=position[..., 1],
                         z=position[..., 2],
                         heading=heading,
                         length=shape[..., 0],
                         width=shape[..., 1],
                         height=shape[..., 2],
                         valid=valid,
                         object_ids=object_ids,
                         object_types=object_types)

        return init_dict

    @classmethod
    def _get_init_dict_from_raw(cls,
                                scenario: scenario_pb2.Scenario): # type: ignore

        """Load from tfrecords data"""
        states, dimensions, objects = [], [], []
        for track in scenario.tracks:  # n_object
            # Iterate over a single object's states.
            track_states, track_dimensions = [], []
            for state in track.states:  # n_timestep
                track_states.append((state.center_x, state.center_y, state.center_z,
                                     state.heading, state.valid))
                track_dimensions.append((state.length, state.width, state.height))
            # Adds to the global states.
            states.append(list(zip(*track_states)))
            dimensions.append(list(zip(*track_dimensions)))
            objects.append((track.id, track.object_type))

        # Unpack and convert to tf tensors.
        x, y, z, heading, valid = [torch.tensor(s) for s in zip(*states)]
        length, width, height = [torch.tensor(s) for s in zip(*dimensions)]
        object_ids, object_types = [torch.tensor(s) for s in zip(*objects)]

        av_id = object_ids[scenario.sdc_track_index]

        init_dict = dict(x=x, y=y, z=z,
                         heading=heading,
                         length=length,
                         width=width,
                         height=height,
                         valid=valid,
                         object_id=object_ids,
                         object_type=object_types,
                         av_id=int(av_id))

        return init_dict

    @classmethod
    def from_scenario(cls,
                        scenario: scenario_pb2.Scenario, # type: ignore
                        processed_scenario: Optional[dict]=None,
                        from_where: str='raw'):

        if from_where == 'raw':
            init_dict = cls._get_init_dict_from_raw(scenario)
        elif from_where == 'processed':
            assert processed_scenario is not None, f'`processed_scenario` should be given!'
            init_dict = cls._get_init_dict_from_processed(processed_scenario)
        else:
            raise RuntimeError(f'Invalid from {from_where}')

        if processed_scenario is not None:
            init_dict.update(state=processed_scenario['agent']['state_idx'],
                             token_pos=processed_scenario['agent']['token_pos'],
                             token_heading=processed_scenario['agent']['token_heading'],
                             token_valid=processed_scenario['agent']['raw_agent_valid_mask'],
                             processed_object_id=processed_scenario['agent']['id'],
                             processed_av_id=int(processed_scenario['agent']['id'][
                                 processed_scenario['agent']['av_idx']
                             ]),
            )

        return cls(**init_dict)


@dataclass
class ScenarioRollouts:
    scenario_id: Optional[str] = None
    joint_scenes: List[ObjectTrajectories] = field(default_factory=list)


""" Conversion Methods """

def scenario_to_trajectories(
    scenario: scenario_pb2.Scenario, # type: ignore
    processed_scenario: Optional[dict]=None,
    from_where: Optional[str]='raw',
    remove_history: Optional[bool]=False
) -> ObjectTrajectories:
    """Converts a WOMD Scenario proto into the `ObjectTrajectories`.

    Returns:
    A `ObjectTrajectories` with trajectories copied from data.
    """
    trajectories = ObjectTrajectories.from_scenario(scenario,
                                                    processed_scenario,
                                                    from_where,
                    )
    # Slice by the required sim agents.
    sim_agent_ids = get_sim_agent_ids(scenario)
    # CONSOLE.log(f'sim_agent_ids of log scenario: {sim_agent_ids} total: {len(sim_agent_ids)}')
    trajectories = trajectories.gather_objects_by_id(torch.tensor(sim_agent_ids))

    if remove_history:
        # Slice in time to only include steps after `current_time_index`.
        trajectories = trajectories.slice_time(submission_specs.CURRENT_TIME_INDEX + 1)  # 10 + 1
        if trajectories.valid.shape[-1] != submission_specs.N_SIMULATION_STEPS:  # 80 simulated steps
            raise ValueError(
                'The Scenario used does not include the right number of time steps. '
                f'Expected: {submission_specs.N_SIMULATION_STEPS}, '
                f'Actual: {trajectories.valid.shape[-1]}.')

    return trajectories


def _unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def get_scenario_id_int_tensor(scenario_id: List[str], device: torch.device=torch.device('cpu')) -> torch.Tensor:
    scenario_id_int_tensor = []
    for str_id in scenario_id:
        int_id = [-1] * 16  # max_len of scenario_id string is 16
        for i, c in enumerate(str_id):
            int_id[i] = ord(c)
        scenario_id_int_tensor.append(
            torch.tensor(int_id, dtype=torch.int32, device=device)
        )
    return torch.stack(scenario_id_int_tensor, dim=0)  # [n_scenario, 16]


def output_to_rollouts(scenario: dict) -> List[ScenarioRollouts]:  # n_scenario
    # scenario_id: Tensor,    # [n_scenario, n_str_length]
    # agent_id: Tensor,       # [n_agent, n_rollout]
    # agent_batch: Tensor,    # [n_agent]
    # pred_traj: Tensor,      # [n_agent, n_rollout, n_step, 2]
    # pred_z: Tensor,         # [n_agent, n_rollout, n_step]
    # pred_head: Tensor,      # [n_agent, n_rollout, n_step]
    # pred_shape: Tensor,     # [n_agent, n_rollout, 3]
    # pred_type: Tensor,      # [n_agent, n_rollout]
    # pred_state: Tensor,     # [n_agent, n_rollout, n_step]
    scenario_id = scenario['scenario_id']
    av_id = (
        scenario['av_id'] if 'av_id' in scenario else -1
    )
    agent_id = scenario['agent_id']
    agent_batch = scenario['agent_batch']
    pred_traj = scenario['pred_traj']
    pred_z = scenario['pred_z']
    pred_head = scenario['pred_head']
    pred_shape = scenario['pred_shape']
    pred_type = scenario['pred_type']
    pred_state = (
        scenario['pred_state'] if 'pred_state' in scenario else
        torch.zeros_like(pred_z).long()
    )
    pred_valid = scenario['pred_valid']
    token_pos = scenario['token_pos']
    token_head = scenario['token_head']

    # CONSOLE.log("Generate scenario rollouts ...")
    # CONSOLE.log(f'scenario_id: {scenario_id}')
    # CONSOLE.log(f'agent_id: {agent_id.flatten()} total: {agent_id.shape}')
    # CONSOLE.log(f'av_id: {av_id}')
    # CONSOLE.log(f'agent_batch: {agent_batch} total: {agent_batch.shape}')
    # CONSOLE.log(f'pred_traj: {pred_traj.shape}')
    # CONSOLE.log(f'pred_z: {pred_z.shape}')
    # CONSOLE.log(f'pred_head: {pred_head.shape}')
    # CONSOLE.log(f'pred_shape: {pred_shape.shape}')
    # CONSOLE.log(f'pred_type: {pred_type.shape}')
    # CONSOLE.log(f'pred_state: {pred_state.shape}')
    # CONSOLE.log(f'token_pos: {token_pos.shape}')
    # CONSOLE.log(f'token_head: {token_head.shape}')

    scenario_id = scenario_id.cpu().numpy()
    n_agent, n_rollout, n_step, _ = pred_traj.shape
    agent_id = _unbatch(agent_id, agent_batch)
    pred_traj = _unbatch(pred_traj, agent_batch)
    pred_z = _unbatch(pred_z, agent_batch)
    pred_head = _unbatch(pred_head, agent_batch)
    pred_shape = _unbatch(pred_shape, agent_batch)
    pred_type = _unbatch(pred_type, agent_batch)
    pred_state = _unbatch(pred_state, agent_batch)
    pred_valid = _unbatch(pred_valid, agent_batch)
    token_pos = _unbatch(token_pos, agent_batch)
    token_head = _unbatch(token_head, agent_batch)

    agent_id = [x.cpu() for x in agent_id]
    pred_traj = [x.cpu() for x in pred_traj]
    pred_z = [x.cpu() for x in pred_z]
    pred_head = [x.cpu() for x in pred_head]
    pred_shape = [x[:, :, None].repeat(1, 1, n_step, 1).cpu() for x in pred_shape]
    pred_type = [x[:, :, None].repeat(1, 1, n_step, 1).cpu() for x in pred_type]
    pred_state = [x.cpu() for x in pred_state]
    pred_valid = [x.cpu() for x in pred_valid]
    token_pos = [x.cpu() for x in token_pos]
    token_head = [x.cpu() for x in token_head]

    n_scenario = scenario_id.shape[0]
    scenario_rollouts = []
    for i_scenario in range(n_scenario):
        joint_scenes = []
        for i_rollout in range(n_rollout):  # 1
            joint_scenes.append(
                ObjectTrajectories(
                    x=pred_traj[i_scenario][:, i_rollout, :, 0],
                    y=pred_traj[i_scenario][:, i_rollout, :, 1],
                    z=pred_z[i_scenario][:, i_rollout],
                    heading=pred_head[i_scenario][:, i_rollout],
                    length=pred_shape[i_scenario][:, i_rollout, :, 0],
                    width=pred_shape[i_scenario][:, i_rollout, :, 1],
                    height=pred_shape[i_scenario][:, i_rollout, :, 2],
                    valid=pred_valid[i_scenario][:, i_rollout],
                    state=pred_state[i_scenario][:, i_rollout],
                    object_id=agent_id[i_scenario][:, i_rollout],
                    processed_object_id=agent_id[i_scenario][:, i_rollout],
                    object_type=pred_type[i_scenario][:, i_rollout],
                    token_pos=token_pos[i_scenario][:, i_rollout, :, :2],
                    token_heading=token_head[i_scenario][:, i_rollout],
                    av_id=av_id,
                    processed_av_id=av_id,
                )
            )

        _str_scenario_id = "".join([chr(x) for x in scenario_id[i_scenario] if x > 0])
        scenario_rollouts.append(
            ScenarioRollouts(
                joint_scenes=joint_scenes, scenario_id=_str_scenario_id
            )
        )

    # CONSOLE.log(f'n_scenario: {len(scenario_rollouts)}')
    # CONSOLE.log(f'n_rollout: {len(scenario_rollouts[0].joint_scenes)}')
    # CONSOLE.log(f'x shape: {scenario_rollouts[0].joint_scenes[0].x.shape}')

    return scenario_rollouts


""" Compute Metric Features """

def _compute_metametric(
    config: long_metrics_pb2.SimAgentMetricsConfig, # type: ignore
    metrics: long_metrics_pb2.SimAgentMetrics, # type: ignore
):
    """Computes the meta-metric aggregation."""
    metametric = 0.0
    for field_name in _METRIC_FIELD_NAMES:
        likelihood_field_name = field_name + '_likelihood'
        weight = getattr(config, field_name).metametric_weight
        metric_score = getattr(metrics, likelihood_field_name)
        metametric += weight * metric_score
    return metametric


def _compute_metametric_long(
    config: long_metrics_pb2.SimAgentMetricsConfig, # type: ignore
    metrics: Dict[str, Tensor],
):
    """Computes the meta-metric aggregation."""
    metametric = torch.zeros((metrics['linear_speed_likelihood'].shape[1]))
    for field_name in _METRIC_FIELD_NAMES:
        likelihood_field_name = field_name + '_likelihood'
        weight = getattr(config, field_name).metametric_weight
        metric_score = metrics[likelihood_field_name][0]
        metametric += weight * metric_score
    for field_name in _METRIC_FIELD_NAMES:
        likelihood_field_name = field_name + '_likelihood'
        metric_score = metrics[likelihood_field_name][0]
        metametric[metric_score == 0] = 0.
    return metametric


@dataclasses.dataclass(frozen=True)
class MetricFeatures:

    object_id: Tensor
    valid: Tensor
    linear_speed: Tensor
    linear_acceleration: Tensor
    angular_speed: Tensor
    angular_acceleration: Tensor
    distance_to_nearest_object: Tensor
    collision_per_step: Tensor
    time_to_collision: Tensor
    distance_to_road_edge: Tensor
    offroad_per_step: Tensor
    num_placement: Tensor
    num_removement: Tensor
    distance_placement: Tensor
    distance_removement: Tensor

    @classmethod
    def from_file(cls, file_path: str):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Not found file {file_path}')

        with open(file_path, 'rb') as f:
            feat_dict = pickle.load(f)

        fields = [field.name for field in dataclasses.fields(cls)]
        init_dict = dict()
        
        for field in fields:
            if field in feat_dict:
                init_dict[field] = feat_dict[field]
            else:
                init_dict[field] = None

        return cls(**init_dict)

    def unfold(self, size: int, step: int):
        return MetricFeatures(
            object_id=self.object_id,
            valid=self.valid.unfold(1, size, step),
            linear_speed=self.linear_speed.unfold(1, size, step),
            linear_acceleration=self.linear_acceleration.unfold(1, size, step),
            angular_speed=self.angular_speed.unfold(1, size, step),
            angular_acceleration=self.angular_acceleration.unfold(1, size, step),
            distance_to_nearest_object=self.distance_to_nearest_object.unfold(1, size, step),
            collision_per_step=self.collision_per_step.unfold(1, size, step),
            time_to_collision=self.time_to_collision.unfold(1, size, step),
            distance_to_road_edge=self.distance_to_road_edge.unfold(1, size, step),
            offroad_per_step=self.offroad_per_step.unfold(1, size, step),
            num_placement=self.num_placement.unfold(1, size // SHIFT, step // SHIFT),
            num_removement=self.num_removement.unfold(1, size // SHIFT, step // SHIFT),
            distance_placement=self.distance_placement.unfold(1, size // SHIFT, step // SHIFT),
            distance_removement=self.distance_removement.unfold(1, size // SHIFT, step // SHIFT),
        )


def compute_metric_features(
    simulate_trajectories: ObjectTrajectories,
    evaluate_agent_ids: Optional[Tensor]=None,
    scenario_log: Optional[scenario_pb2.Scenario]=None, # type: ignore
) -> MetricFeatures:

    if evaluate_agent_ids is not None:
        evaluate_trajectories = simulate_trajectories.gather_objects_by_id(
            evaluate_agent_ids
        )
    else:
        evaluate_trajectories = simulate_trajectories

    # valid mask
    validity_mask = evaluate_trajectories.valid
    validity_mask = validity_mask[:, submission_specs.CURRENT_TIME_INDEX + 1:]

    # ! Kinematics-related features, i.e. speed and acceleration, this needs
    # history steps to be prepended to make the first evaluate step valid.
    # Resulted `lienar_speed` and others: (n_object_to_evaluate, n_future_step)
    linear_speed, linear_accel, angular_speed, angular_accel = (
        trajectory_features.compute_kinematic_features(
            evaluate_trajectories.x,
            evaluate_trajectories.y,
            evaluate_trajectories.z,
            evaluate_trajectories.heading,
            seconds_per_step=submission_specs.STEP_DURATION_SECONDS))
    # Removes the data corresponding to the history time interval.
    linear_speed, linear_accel, angular_speed, angular_accel = (
        map(lambda t: t[:, submission_specs.CURRENT_TIME_INDEX + 1:],
            [linear_speed, linear_accel, angular_speed, angular_accel])
    )

    # ! Distances to nearest objects.
    evaluate_object_mask = torch.ones(len(simulate_trajectories.object_id)).bool()
    distances_to_objects = interact_features.compute_distance_to_nearest_object(
        center_x=simulate_trajectories.x,
        center_y=simulate_trajectories.y,
        center_z=simulate_trajectories.z,
        length=simulate_trajectories.length,
        width=simulate_trajectories.width,
        height=simulate_trajectories.height,
        heading=simulate_trajectories.heading,
        valid=simulate_trajectories.valid,
        evaluated_object_mask=evaluate_object_mask,
    )
    distances_to_objects = (
        distances_to_objects[:, submission_specs.CURRENT_TIME_INDEX + 1:])
    is_colliding_per_step = torch.lt(
        distances_to_objects, interact_features.COLLISION_DISTANCE_THRESHOLD)

    # ! Time to collision
    times_to_collision = (
        interact_features.compute_time_to_collision_with_object_in_front(
            center_x=simulate_trajectories.x,
            center_y=simulate_trajectories.y,
            length=simulate_trajectories.length,
            width=simulate_trajectories.width,
            heading=simulate_trajectories.heading,
            valid=simulate_trajectories.valid,
            evaluated_object_mask=evaluate_object_mask,
            seconds_per_step=submission_specs.STEP_DURATION_SECONDS,
        )
    )
    times_to_collision = times_to_collision[:, submission_specs.CURRENT_TIME_INDEX + 1:]

    # ! Distance to road edge
    distances_to_road_edge = torch.empty_like(distances_to_objects)
    is_offroad_per_step = torch.empty_like(is_colliding_per_step)
    if scenario_log is not None:
        road_edges = []
        for map_feature in scenario_log.map_features:
            if map_feature.HasField('road_edge'):
                road_edges.append(map_feature.road_edge.polyline)
        distances_to_road_edge = map_features.compute_distance_to_road_edge(
            center_x=simulate_trajectories.x,
            center_y=simulate_trajectories.y,
            center_z=simulate_trajectories.z,
            length=simulate_trajectories.length,
            width=simulate_trajectories.width,
            height=simulate_trajectories.height,
            heading=simulate_trajectories.heading,
            valid=simulate_trajectories.valid,
            evaluated_object_mask=evaluate_object_mask,
            road_edge_polylines=road_edges,
        )
        distances_to_road_edge = distances_to_road_edge[:, submission_specs.CURRENT_TIME_INDEX + 1:]
        is_offroad_per_step = torch.gt(
            distances_to_road_edge, map_features.OFFROAD_DISTANCE_THRESHOLD
        )

    # ! Placement
    if simulate_trajectories.av_id == simulate_trajectories.processed_av_id == -1:
        n_agent, n_step_10hz = linear_speed.shape
        num_placement = torch.zeros((n_step_10hz // SHIFT,))
        num_removement = torch.zeros((n_step_10hz // SHIFT,))
        distance_placement = torch.zeros((n_agent, n_step_10hz // SHIFT))
        distance_removement = torch.zeros((n_agent, n_step_10hz // SHIFT))

    else:
        assert simulate_trajectories.av_id == simulate_trajectories.processed_av_id, \
                f"Got duplicated av_id: {simulate_trajectories.av_id} and {simulate_trajectories.processed_av_id}"
        num_placement, num_removement = (
            placement_features.compute_num_placement(
                state=simulate_trajectories.state,
                valid=simulate_trajectories.token_valid,
                av_id=simulate_trajectories.processed_av_id,
                object_id=simulate_trajectories.processed_object_id,
                agent_state=AGENT_STATE,
            )
        )
        num_placement = num_placement[submission_specs.CURRENT_TIME_INDEX // SHIFT:]
        num_removement = num_removement[submission_specs.CURRENT_TIME_INDEX // SHIFT:]
        distance_placement, distance_removement = (
            placement_features.compute_distance_placement(
                position=simulate_trajectories.token_pos,
                state=simulate_trajectories.state,
                valid=simulate_trajectories.valid,
                av_id=simulate_trajectories.processed_av_id,
                object_id=simulate_trajectories.processed_object_id,
                agent_state=AGENT_STATE,
            )
        )
        distance_placement = distance_placement[:, submission_specs.CURRENT_TIME_INDEX // SHIFT:]
        distance_removement = distance_removement[:, submission_specs.CURRENT_TIME_INDEX // SHIFT:]

    return MetricFeatures(
            object_id=simulate_trajectories.object_id,
            valid=validity_mask,
            # kinematic
            linear_speed=linear_speed,
            linear_acceleration=linear_accel,
            angular_speed=angular_speed,
            angular_acceleration=angular_accel,
            # interact
            distance_to_nearest_object=distances_to_objects,
            collision_per_step=is_colliding_per_step,
            time_to_collision=times_to_collision,
            # map
            distance_to_road_edge=distances_to_road_edge,
            offroad_per_step=is_offroad_per_step,
            # placement
            num_placement=num_placement[None, ...],
            num_removement=num_removement[None, ...],
            distance_placement=distance_placement,
            distance_removement=distance_removement,
    )


@dataclass(frozen=True)
class LogDistributions:

    linear_speed: Tensor
    linear_acceleration: Tensor
    angular_speed: Tensor
    angular_acceleration: Tensor
    distance_to_nearest_object: Tensor
    collision_indication: Tensor
    time_to_collision: Tensor
    num_placement: Tensor
    num_removement: Tensor
    distance_placement: Tensor
    distance_removement: Tensor
    distance_to_road_edge: Optional[Tensor] = None
    offroad_indication: Optional[Tensor] = None


""" Compute Metrics """

def _assert_and_return_batch_size(
    log_samples: Tensor, 
    sim_samples: Tensor
) -> int:
    """Asserts consistency in the tensor shapes and returns batch size.

    Args:
        log_samples: A tensor of shape (batch_size, log_sample_size).
        sim_samples: A tensor of shape (batch_size, sim_sample_size).

    Returns:
        The `batch_size`.
    """
    assert log_samples.shape[0] == sim_samples.shape[0], "Log and Sim batch sizes must be equal."
    return log_samples.shape[0]


def _reduce_average_with_validity(
    tensor: Tensor, validity: Tensor) -> Tensor:
    """Returns the tensor's average, only selecting valid items.

    Args:
    tensor: A float tensor of any shape.
    validity: A boolean tensor of the same shape as `tensor`.

    Returns:
    A float tensor of shape (1,), containing the average of the valid elements
    of `tensor`.
    """
    if tensor.shape != validity.shape:
        raise ValueError('Shapes of `tensor` and `validity` must be the same.'
                            f'(Actual: {tensor.shape}, {validity.shape}).')
    cond_sum = torch.sum(torch.where(validity, tensor, torch.zeros_like(tensor)), dim=-1)
    valid_sum = torch.sum(validity, dim=-1)
    if valid_sum.sum() == 0:
        return torch.full(valid_sum.shape[:2], -torch.inf)
    return cond_sum / valid_sum


def _reduce_mean(tensor: Tensor, dim: Optional[int] = None) -> Tensor:
    validity = (tensor > 0.) & (tensor <= 1.)
    if dim is None:
        sum = torch.sum(torch.where(validity, tensor, torch.zeros_like(tensor)))
        count = validity.sum().clamp(min=1)
        return sum / count
    else:
        sum = torch.sum(torch.where(validity, tensor, torch.zeros_like(tensor)), dim=0)
        count = validity.sum(dim=0).clamp(min=1)
        return sum / count


def histogram_estimate(
    config: long_metrics_pb2.SimAgentMetricsConfig.HistogramEstimate, # type: ignore
    log_samples: Tensor,
    sim_samples: Tensor,
) -> Tensor:
    """Computes log-likelihoods of samples based on histograms.

    Args:
        config: A configuration dictionary, similar to the one in TensorFlow.
        log_samples: A tensor of shape (batch_size, log_sample_size),
            containing `log_sample_size` samples from `batch_size` independent
            populations.
        sim_samples: A tensor of shape (batch_size, sim_sample_size),
            containing `sim_sample_size` samples from `batch_size` independent
            populations.

    Returns:
        A tensor of shape (batch_size, log_sample_size), where each element (i, k)
        is the log likelihood of the log sample (i, k) under the sim distribution
        (i).
    """
    batch_size = _assert_and_return_batch_size(log_samples, sim_samples)

    # We generate `num_bins`+1 edges for the histogram buckets.
    edges = torch.linspace(
        config.min_val, config.max_val, config.num_bins + 1
    ).float()

    # Clip the samples to avoid errors with histograms.
    log_samples = torch.clamp(log_samples, config.min_val, config.max_val)
    sim_samples = torch.clamp(sim_samples, config.min_val, config.max_val)

    # Create the categorical distribution for simulation. `tfp.histogram` returns
    # a tensor of shape (num_bins, batch_size), so we need to transpose to conform
    # to `tfp.distribution.Categorical`, which requires `probs` to be
    # (batch_size, num_bins).
    sim_counts = torch.vmap(lambda x: torch.histogram(x, bins=edges).hist)(sim_samples)
    sim_counts += config.additive_smoothing_pseudocount
    distributions = torch.distributions.Categorical(probs=sim_counts)

    # Generate the counts for the log distribution. We reshape the log samples to
    # (batch_size * log_sample_size, 1), so every log sample is independently
    # scored.
    log_values_flat = log_samples.reshape(-1, 1)
    # Shape of log_counts: (batch_size * log_sample_size, num_bins).
    log_counts = torch.vmap(lambda x: torch.histogram(x, bins=edges).hist)(log_values_flat)
    # Identify which bin each sample belongs to and get the log probability of
    # that bin under the sim distribution.
    max_log_bin = log_counts.argmax(dim=-1)
    batched_max_log_bin = max_log_bin.reshape(batch_size, -1)

    # Since we have defined the categorical distribution to have `batch_size`
    # independent populations, tfp expects this `batch_size` to be in the last
    # dimension of the tensor, so transpose the log bins to
    # (log_sample_size, batch_size).
    log_likelihood = distributions.log_prob(batched_max_log_bin.transpose(0, 1))

    # Return log likelihood in the shape (batch_size, log_sample_size)
    return log_likelihood.transpose(0, 1)


def log_likelihood_estimate_timeseries(
    field: str,
    feature_config: long_metrics_pb2.SimAgentMetricsConfig.FeatureConfig, # type: ignore
    sim_values: Tensor,
    log_distributions: torch.distributions.Categorical,
    estimate_method: str='histogram',
) -> Tensor:
    """Computes the log-likelihood estimates for a time-series simulated feature.

    Args:
    feature_config: A time-series compatible `FeatureConfig`.
    log_distributions: A float Tensor with shape (batch_sizie, n_bins).
    sim_values: A float Tensor with shape (n_objects / n_scenarios, n_segments, n_steps).

    Returns:
    A tensor of shape (n_objects, n_steps) containing the simulation probability
    estimates of the simulation features under the logged distribution of the same
    feature.
    """
    assert sim_values.ndim == 3, f'Expect sim_values.ndim==3, got {sim_values.ndim}, shape {sim_values.shape} for {field}' 

    sim_values_flat = sim_values.reshape(-1, 1)  # [n_objects * n_segments * n_steps]

    # ! calculate distributions for simulate features
    if estimate_method == 'histogram':
        config = feature_config.histogram
    elif estimate_method == 'bernoulli':
        config = (
            long_metrics_pb2.SimAgentMetricsConfig.HistogramEstimate(
                min_val=-0.5, max_val=0.5, num_bins=2,
                additive_smoothing_pseudocount=feature_config.bernoulli.additive_smoothing_pseudocount
            )
        )
        sim_values_flat = sim_values_flat.float()  # cast torch.bool to torch.float32

    # We generate `num_bins`+1 edges for the histogram buckets.
    edges = torch.linspace(
        config.min_val, config.max_val, config.num_bins + 1
    ).float()

    sim_counts = torch.vmap(lambda x: torch.histogram(x, bins=edges).hist)(sim_values_flat)  # [batch_size, num_bins]
    # Identify which bin each sample belongs to and get the log probability of
    # that bin under the sim distribution.
    max_sim_bin = sim_counts.argmax(dim=-1)
    batched_max_sim_bin = max_sim_bin.reshape(1, -1)  # `batch_size` = 1, follows the log distributions

    sim_likelihood = log_distributions.log_prob(batched_max_sim_bin.transpose(0, 1)).flatten()
    return sim_likelihood.reshape(*sim_values.shape)  # [n_objects, n_segments, n_steps]


def compute_scenario_metrics_for_bundle(
    config: long_metrics_pb2.SimAgentMetricsConfig, # type: ignore
    log_distributions: LogDistributions,
    scenario_log: Optional[scenario_pb2.Scenario], # type: ignore
    scenario_rollouts: ScenarioRollouts,
) -> Tuple[long_metrics_pb2.SimAgentMetrics, dict]: # type: ignore

    features_fields = [field.name for field in dataclasses.fields(MetricFeatures)]
    features_fields.remove('object_id')

    # ! compute simluation features
    # CONSOLE.log('[on yellow] Compute sim features [/]')
    sim_features = collections.defaultdict(list)
    for simulate_trajectories in tqdm(scenario_rollouts.joint_scenes, leave=False, desc='rollouts ...'):  # n_rollout=1
        rollout_features = compute_metric_features(
            simulate_trajectories,
            evaluate_agent_ids=None,
            scenario_log=scenario_log
        )

        for field in features_fields:
            sim_features[field].append(getattr(rollout_features, field))

    for field in features_fields:
        if sim_features[field][0] is not None:
            sim_features[field] = torch.concat(sim_features[field], dim=0)  # n_rollout for dim=0

    sim_features = MetricFeatures(
        **sim_features, object_id=None,
    )
    # after unfold: linear_speed shape [n_agent, n_window, window_size],
    # num_placement shape [n_scenario=1, n_window, window_size]
    flattened_sim_features = copy.deepcopy(sim_features)
    sim_features = sim_features.unfold(size=submission_specs.N_SIMULATION_STEPS, step=SHIFT)

    ## ! compute metrics

    # ! kinematics-related metrics
    linear_speed_log_likelihood = log_likelihood_estimate_timeseries(
        field='linear_speed',
        feature_config=config.linear_speed,
        sim_values=sim_features.linear_speed,
        log_distributions=log_distributions.linear_speed,
    )
    angular_speed_log_likelihood = log_likelihood_estimate_timeseries(
        field='angular_speed',
        feature_config=config.angular_speed,
        sim_values=sim_features.angular_speed,
        log_distributions=log_distributions.angular_speed,
    )
    speed_validity, acceleration_validity = (
        trajectory_features.compute_kinematic_validity(flattened_sim_features.valid)
    )
    speed_validity = speed_validity.unfold(1, size=submission_specs.N_SIMULATION_STEPS, step=SHIFT)
    acceleration_validity = acceleration_validity.unfold(1, size=submission_specs.N_SIMULATION_STEPS, step=SHIFT)
    linear_speed_likelihood = torch.exp(_reduce_average_with_validity(
        linear_speed_log_likelihood, speed_validity))
    angular_speed_likelihood = torch.exp(_reduce_average_with_validity(
        angular_speed_log_likelihood, speed_validity))
    # CONSOLE.log(f'linear_speed_likelihood: {linear_speed_likelihood}')
    # CONSOLE.log(f'angular_speed_likelihood: {angular_speed_likelihood}')

    linear_accel_log_likelihood = log_likelihood_estimate_timeseries(
        field='linear_acceleration',
        feature_config=config.linear_acceleration,
        sim_values=sim_features.linear_acceleration,
        log_distributions=log_distributions.linear_acceleration,
    )
    angular_accel_log_likelihood = log_likelihood_estimate_timeseries(
        field='angular_acceleration',
        feature_config=config.angular_acceleration,
        sim_values=sim_features.angular_acceleration,
        log_distributions=log_distributions.angular_acceleration,
    )
    linear_accel_likelihood = torch.exp(_reduce_average_with_validity(
        linear_accel_log_likelihood, acceleration_validity))
    angular_accel_likelihood = torch.exp(_reduce_average_with_validity(
        angular_accel_log_likelihood, acceleration_validity))

    # ! collision and distance to other objects.

    sim_collision_indication = torch.any(
        torch.where(sim_features.valid, sim_features.collision_per_step, False),
        dim=2)[..., None]  # add a dummy time dimension
    collision_score = log_likelihood_estimate_timeseries(
        field='collision_indication',
        feature_config=config.collision_indication,
        sim_values=sim_collision_indication,
        log_distributions=log_distributions.collision_indication,
        estimate_method='bernoulli',
    )
    collision_likelihood = torch.exp(torch.mean(collision_score))

    distance_to_objects_log_likelihod = log_likelihood_estimate_timeseries(
        field='distance_to_nearest_object',
        feature_config=config.distance_to_nearest_object,
        sim_values=sim_features.distance_to_nearest_object,
        log_distributions=log_distributions.distance_to_nearest_object,
    )
    distance_to_objects_valid = sim_features.valid & (
        (sim_features.distance_to_nearest_object >= config.distance_to_nearest_object.histogram.min_val) &
        (sim_features.distance_to_nearest_object <= config.distance_to_nearest_object.histogram.max_val)
    )
    distance_to_objects_likelihod = torch.exp(_reduce_average_with_validity(
        distance_to_objects_log_likelihod, distance_to_objects_valid))

    ttc_log_likelihood = log_likelihood_estimate_timeseries(
        field='time_to_collision',
        feature_config=config.time_to_collision,
        sim_values=sim_features.time_to_collision,
        log_distributions=log_distributions.time_to_collision,
    )
    ttc_likelihood = torch.exp(_reduce_average_with_validity(
        ttc_log_likelihood, sim_features.valid))

    # ! placement

    num_placement_log_likelihood = log_likelihood_estimate_timeseries(
        field='num_placement',
        feature_config=config.num_placement,
        sim_values=sim_features.num_placement.float(),
        log_distributions=log_distributions.num_placement,
    )
    num_placement_likelihood = torch.exp(torch.mean(num_placement_log_likelihood))
    num_removement_log_likelihood = log_likelihood_estimate_timeseries(
        field='num_removement',
        feature_config=config.num_removement,
        sim_values=sim_features.num_removement.float(),
        log_distributions=log_distributions.num_removement,
    )
    num_removement_likelihood = torch.exp(torch.mean(num_removement_log_likelihood))

    distance_placement_log_likelihood = log_likelihood_estimate_timeseries(
        field='distance_placement',
        feature_config=config.distance_placement,
        sim_values=sim_features.distance_placement,
        log_distributions=log_distributions.distance_placement,
    )
    distance_placement_validity = sim_features.valid.unfold(-1, SHIFT, SHIFT)[..., 0] & (
        (sim_features.distance_placement > config.distance_placement.histogram.min_val) &
        (sim_features.distance_placement < config.distance_placement.histogram.max_val)
    )
    distance_placement_likelihood = torch.exp(_reduce_average_with_validity(
        distance_placement_log_likelihood, distance_placement_validity))
    distance_removement_log_likelihood = log_likelihood_estimate_timeseries(
        field='distance_removement',
        feature_config=config.distance_removement,
        sim_values=sim_features.distance_removement,
        log_distributions=log_distributions.distance_removement,
    )
    distance_removement_validity = sim_features.valid.unfold(-1, SHIFT, SHIFT)[..., 0] & (
        (sim_features.distance_removement > config.distance_removement.histogram.min_val) &
        (sim_features.distance_removement < config.distance_removement.histogram.max_val)
    )
    distance_removement_likelihood = torch.exp(_reduce_average_with_validity(
        distance_removement_log_likelihood, distance_removement_validity))

    # ==== Simulated collision and offroad rates ====
    simulated_collision_rate = torch.sum(
        sim_collision_indication.long()
    ) / torch.sum(torch.ones_like(sim_collision_indication).long())

    # ==== Meta metric ====
    likelihood_metrics = {
        'linear_speed_likelihood': float(_reduce_mean(linear_speed_likelihood).numpy()),
        'linear_acceleration_likelihood': float(_reduce_mean(linear_accel_likelihood).numpy()),
        'angular_speed_likelihood': float(_reduce_mean(angular_speed_likelihood).numpy()),
        'angular_acceleration_likelihood': float(_reduce_mean(angular_accel_likelihood).numpy()),
        'distance_to_nearest_object_likelihood': float(_reduce_mean(distance_to_objects_likelihod).numpy()),
        'collision_indication_likelihood': float(_reduce_mean(collision_likelihood).numpy()),
        'time_to_collision_likelihood': float(_reduce_mean(ttc_likelihood).numpy()),
        'num_placement_likelihood': float(_reduce_mean(num_placement_likelihood).numpy()),
        'num_removement_likelihood': float(_reduce_mean(num_removement_likelihood).numpy()),
        'distance_placement_likelihood': float(_reduce_mean(distance_placement_likelihood).numpy()),
        'distance_removement_likelihood': float(_reduce_mean(distance_removement_likelihood).numpy()),
    }

    likelihood_metrics_long = {
        'linear_speed_likelihood': _reduce_mean(linear_speed_likelihood, dim=0).unsqueeze(dim=0),
        'linear_acceleration_likelihood': _reduce_mean(linear_accel_likelihood, dim=0).unsqueeze(dim=0),
        'angular_speed_likelihood': _reduce_mean(angular_speed_likelihood, dim=0).unsqueeze(dim=0),
        'angular_acceleration_likelihood': _reduce_mean(angular_accel_likelihood, dim=0).unsqueeze(dim=0),
        'distance_to_nearest_object_likelihood': _reduce_mean(distance_to_objects_likelihod, dim=0).unsqueeze(dim=0),
        'collision_indication_likelihood': _reduce_mean(torch.exp(torch.mean(collision_score, dim=-1)), dim=0).unsqueeze(dim=0),
        'time_to_collision_likelihood': _reduce_mean(ttc_likelihood, dim=0).unsqueeze(dim=0),
        'num_placement_likelihood': torch.exp(torch.mean(num_placement_log_likelihood, dim=-1)),
        'num_removement_likelihood': torch.exp(torch.mean(num_removement_log_likelihood, dim=-1)),
        'distance_placement_likelihood': _reduce_mean(distance_placement_likelihood, dim=0).unsqueeze(dim=0),
        'distance_removement_likelihood': _reduce_mean(distance_removement_likelihood, dim=0).unsqueeze(dim=0),
    }

    metametric = _compute_metametric(
        config, long_metrics_pb2.SimAgentMetrics(**likelihood_metrics)
    )
    metametric_long = _compute_metametric_long(
        config, likelihood_metrics_long
    )
    # CONSOLE.log(f'metametric: {metametric}')

    return long_metrics_pb2.SimAgentMetrics(
        scenario_id=scenario_rollouts.scenario_id,
        metametric=metametric,
        simulated_collision_rate=float(simulated_collision_rate.numpy()),
        # simulated_offroad_rate=simulated_offroad_rate.numpy(),
        **likelihood_metrics,
    ), dict(
        scenario_id=scenario_rollouts.scenario_id,
        metametric=metametric_long.unsqueeze(dim=0),
        **likelihood_metrics_long,
    )


""" Log Features """

def _get_log_distributions(
    field: str,
    feature_config: long_metrics_pb2.SimAgentMetricsConfig.FeatureConfig, # type: ignore
    log_values: Tensor,
    estimate_method: str = 'histogram',
) -> Tensor:
    """Computes the log-likelihood estimates for a time-series simulated feature.

    Args:
    feature_config: A time-series compatible `FeatureConfig`.
    log_values: A float Tensor with shape (n_objects, n_steps).
    sim_values: A float Tensor with shape (n_rollouts, n_objects, n_steps).

    Returns:
    A tensor of shape (n_objects, n_steps) containing the log probability
    estimates of the log features under the simulated distribution of the same
    feature.
    """
    assert log_values.ndim == 2, f'Expect log_values.ndim==2, got {log_values.ndim}, shape {log_values.shape} for {field}' 

    # ! estimate
    if estimate_method == 'histogram':
        config = feature_config.histogram
    elif estimate_method == 'bernoulli':
        config = (
            long_metrics_pb2.SimAgentMetricsConfig.HistogramEstimate(
                min_val=-0.5, max_val=0.5, num_bins=2,
                additive_smoothing_pseudocount=feature_config.bernoulli.additive_smoothing_pseudocount
            )
        )
        log_values = log_values.float()  # cast torch.bool to torch.float32

    if 'distance_' in field:
        log_values = log_values[(log_values > config.min_val) & (log_values < config.max_val)]

    if field == 'num_placement':
        log_values = log_values[:, :-2]  # ignore the last two steps

    # [n_objects, n_steps] -> [n_objects * n_steps]
    log_samples = log_values.reshape(-1)

    # We generate `num_bins`+1 edges for the histogram buckets.
    edges = torch.linspace(
        config.min_val, config.max_val, config.num_bins + 1
    ).float()

    # Clip the samples to avoid errors with histograms. Nonetheless, the min/max
    # values should be configured to never hit this condition in practice.
    log_samples = torch.clamp(log_samples, config.min_val, config.max_val)

    # Create the categorical distribution for simulation. `tfp.histogram` returns
    # a tensor of shape (num_bins, batch_size), so we need to transpose to conform
    # to `tfp.distribution.Categorical`, which requires `probs` to be
    # (batch_size, num_bins).
    log_counts = torch.histogram(log_samples, bins=edges).hist.unsqueeze(dim=0)  # [1, n_samples]
    log_counts += config.additive_smoothing_pseudocount
    distributions = torch.distributions.Categorical(probs=log_counts)

    return distributions


class LongMetric(Metric):

    log_features: MetricFeatures

    def __init__(
            self,
            prefix: str='',
            log_features_dir: str='data/waymo_processed/log_features/',
            config_path: str='infgen/metrics/metric_config.textproto',
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.metrics_config = self.load_metrics_config(config_path)

        self.use_log = False

        self.field_names = [
            "metametric",
            "average_displacement_error",
            "min_average_displacement_error",
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "angular_speed_likelihood",
            "angular_acceleration_likelihood",
            'distance_to_nearest_object_likelihood',
            'collision_indication_likelihood',
            'time_to_collision_likelihood',
            'simulated_collision_rate',
            'num_placement_likelihood',
            'num_removement_likelihood',
            'distance_placement_likelihood',
            'distance_removement_likelihood',
        ]
        for k in self.field_names:
            self.add_state(k, default=torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state(f'{k}_long', default=[], dist_reduce_fx='cat')
        self.add_state('scenario_counter', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('placement_valid_scenario_counter', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('removement_valid_scenario_counter', default=torch.tensor(0.), dist_reduce_fx='sum')

        # get log features
        log_features_path = os.path.join(log_features_dir, 'total_features.pkl')
        if not os.path.exists(log_features_path):
            CONSOLE.log(f'[on yellow] Log features does not exist, loading now ... [/]')
            log_features = aggregate_log_metric_features(log_features_dir)
        else:
            log_features = MetricFeatures.from_file(log_features_path)
            CONSOLE.log(f'Loaded log features from {log_features_path}')
        self.log_features = log_features

        self._compute_distributions()
        CONSOLE.log(f"Calculated log distributions:\n{self.log_distributions}")

    def _compute_distributions(self):
        self.log_distributions = LogDistributions(
            linear_speed = _get_log_distributions('linear_speed',
                self.metrics_config.linear_speed, self.log_features.linear_speed,
            ),
            linear_acceleration = _get_log_distributions('linear_acceleration',
                self.metrics_config.linear_acceleration, self.log_features.linear_acceleration,
            ),
            angular_speed = _get_log_distributions('angular_speed',
                self.metrics_config.angular_speed, self.log_features.angular_speed,
            ),
            angular_acceleration = _get_log_distributions('angular_acceleration',
                self.metrics_config.angular_acceleration, self.log_features.angular_acceleration,
            ),
            distance_to_nearest_object = _get_log_distributions('distance_to_nearest_object',
                self.metrics_config.distance_to_nearest_object, self.log_features.distance_to_nearest_object,
            ),
            collision_indication = _get_log_distributions('collision_indication',
                self.metrics_config.collision_indication,
                log_collision_indication := torch.any(
                    torch.where(self.log_features.valid, self.log_features.collision_per_step, False), dim=1
                )[..., None],  # add a dummy time dimension
                estimate_method = 'bernoulli',
            ),
            time_to_collision = _get_log_distributions('time_to_collision',
                self.metrics_config.time_to_collision, self.log_features.time_to_collision,
            ),
            num_placement = _get_log_distributions('num_placement',
                self.metrics_config.num_placement, self.log_features.num_placement.float(),
            ),
            num_removement = _get_log_distributions('num_removement',
                self.metrics_config.num_removement, self.log_features.num_removement.float(),
            ),
            distance_placement = _get_log_distributions('distance_placement',
                self.metrics_config.distance_placement, self.log_features.distance_placement,
            ),
            distance_removement = _get_log_distributions('distance_removement',
                self.metrics_config.distance_removement, self.log_features.distance_removement,
            ),
        )

    def _compute_scenario_metrics(
        self,
        scenario_file: Optional[str],
        scenario_rollout: ScenarioRollouts,
    ) -> long_metrics_pb2.SimAgentMetrics: # type: ignore

        scenario_log = None
        if self.use_log and scenario_file is not None:
            if not os.path.exists(scenario_file):
                raise FileNotFoundError(f"Not found file {scenario_file}")
            scenario_log = scenario_pb2.Scenario()
            for data in tf.data.TFRecordDataset([scenario_file], compression_type=''):
                scenario_log.ParseFromString(bytes(data.numpy()))
                break

        return compute_scenario_metrics_for_bundle(
            self.metrics_config, self.log_distributions, scenario_log, scenario_rollout
        )

    def compute_metrics(self, outputs: dict) -> List[long_metrics_pb2.SimAgentMetrics]: # type: ignore
        """
        `outputs` is a dict directly generated by predict models:
            >>> outputs = dict(
            >>>     scenario_id=get_scenario_id_int_tensor(data['scenario_id'], device),
            >>>     agent_id=agent_id,
            >>>     agent_batch=agent_batch,
            >>>     pred_traj=pred_traj,
            >>>     pred_z=pred_z,
            >>>     pred_head=pred_head,
            >>>     pred_shape=pred_shape,
            >>>     pred_type=pred_type,
            >>>     pred_state=pred_state,
            >>> )
        """

        scenario_rollouts = output_to_rollouts(outputs)
        log_paths: List[str] = outputs['tfrecord_path']

        pool_scenario_metrics = []
        for _scenario_file, _scenario_rollout in tqdm(
            zip(log_paths, scenario_rollouts), leave=False, desc='scenarios ...'):  # n_scenarios
            pool_scenario_metrics.append(
                self._compute_scenario_metrics(
                    _scenario_file, _scenario_rollout,
                )
            )

        return pool_scenario_metrics

    def update(
            self,
            outputs: Optional[dict]=None,
            metrics: Optional[List[long_metrics_pb2.SimAgentMetrics]]=None # type: ignore
        ) -> None:

        if metrics is None:
            assert outputs is not None, f'`outputs` should not be None!'
            metrics = self.compute_metrics(outputs)

        for scenario_metrics in metrics:

            _scenario_metrics, _scenario_metrics_long = scenario_metrics

            self.scenario_counter += 1

            if _scenario_metrics.distance_placement_likelihood > 0:
                self.placement_valid_scenario_counter += 1

            if _scenario_metrics.distance_removement_likelihood > 0:
                self.removement_valid_scenario_counter += 1

            # float metrics
            self.metametric += _scenario_metrics.metametric
            self.average_displacement_error += (
                _scenario_metrics.average_displacement_error
            )
            self.min_average_displacement_error += (
                _scenario_metrics.min_average_displacement_error
            )
            self.linear_speed_likelihood += _scenario_metrics.linear_speed_likelihood
            self.linear_acceleration_likelihood += (
                _scenario_metrics.linear_acceleration_likelihood
            )
            self.angular_speed_likelihood += _scenario_metrics.angular_speed_likelihood
            self.angular_acceleration_likelihood += (
                _scenario_metrics.angular_acceleration_likelihood
            )
            self.distance_to_nearest_object_likelihood += (
                _scenario_metrics.distance_to_nearest_object_likelihood
            )
            self.collision_indication_likelihood += (
                _scenario_metrics.collision_indication_likelihood
            )
            self.time_to_collision_likelihood += (
                _scenario_metrics.time_to_collision_likelihood
            )
            self.simulated_collision_rate += _scenario_metrics.simulated_collision_rate

            self.num_placement_likelihood += (
                _scenario_metrics.num_placement_likelihood
            )
            self.num_removement_likelihood += (
                _scenario_metrics.num_removement_likelihood
            )
            self.distance_placement_likelihood += (
                _scenario_metrics.distance_placement_likelihood
            )
            self.distance_removement_likelihood += (
                _scenario_metrics.distance_removement_likelihood
            )

            # long metrics
            self.metametric_long.append(_scenario_metrics_long['metametric'])
            self.linear_speed_likelihood_long.append(_scenario_metrics_long['linear_speed_likelihood'])
            self.linear_acceleration_likelihood_long.append(
                _scenario_metrics_long['linear_acceleration_likelihood']
            )
            self.angular_speed_likelihood_long.append(_scenario_metrics_long['angular_speed_likelihood'])
            self.angular_acceleration_likelihood_long.append(
                _scenario_metrics_long['angular_acceleration_likelihood']
            )
            self.distance_to_nearest_object_likelihood_long.append(
                _scenario_metrics_long['distance_to_nearest_object_likelihood']
            )
            self.collision_indication_likelihood_long.append(
                _scenario_metrics_long['collision_indication_likelihood']
            )
            self.time_to_collision_likelihood_long.append(
                _scenario_metrics_long['time_to_collision_likelihood']
            )
            self.num_placement_likelihood_long.append(
                _scenario_metrics_long['num_placement_likelihood']
            )
            self.num_removement_likelihood_long.append(
                _scenario_metrics_long['num_removement_likelihood']
            )
            self.distance_placement_likelihood_long.append(
                _scenario_metrics_long['distance_placement_likelihood']
            )
            self.distance_removement_likelihood_long.append(
                _scenario_metrics_long['distance_removement_likelihood']
            )

    def compute(self) -> Dict[str, Tensor]:
        metrics_dict = {}
        metrics_long_dict = {}
        for k in self.field_names:
            # float metrics
            if k not in ('distance_placement_likelihood', 'distance_removement_likelihood'):
                metrics_dict[k] = getattr(self, k) / max(self.scenario_counter, 1)
            if k == 'distance_placement_likelihood':
                metrics_dict[k] = getattr(self, k) / max(self.placement_valid_scenario_counter, 1)
            if k == 'distance_removement_likelihood':
                metrics_dict[k] = getattr(self, k) / max(self.removement_valid_scenario_counter, 1)
            # long metrics
            if len(getattr(self, f'{k}_long')) > 0:
                metrics_long_dict[k] = _reduce_mean(torch.cat(getattr(self, f'{k}_long')), dim=0)

        mean_metrics = long_metrics_pb2.SimAgentMetrics(
            scenario_id='', **metrics_dict,
        )
        final_metrics = self.aggregate_metrics_to_buckets(
            self.metrics_config, mean_metrics
        )
        mean_long_metrics = metrics_long_dict
        final_long_metrics = self.aggregate_metrics_long_to_buckets(
            self.metrics_config, mean_long_metrics
        )

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric": final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics": final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics": final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics": final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/placement_based_metrics": final_metrics.placement_based_metrics,
            f"{self.prefix}/wosac/min_ade": final_metrics.min_ade,
            f"{self.prefix}/wosac/scenario_counter": int(self.scenario_counter),
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = float(metrics_dict[k])

        out_dict.update({
            f"{self.prefix}/wosac_long/realism_meta_metric": [round(x, 4) for x in final_long_metrics['realism_meta_metric'].tolist()],
            f"{self.prefix}/wosac_long/kinematic_metrics": [round(x, 4) for x in final_long_metrics['kinematic_metrics'].tolist()],
            f"{self.prefix}/wosac_long/interactive_metrics": [round(x, 4) for x in final_long_metrics['interactive_metrics'].tolist()],
            f"{self.prefix}/wosac_long/map_based_metrics": [round(x, 4) for x in final_long_metrics['map_based_metrics'].tolist()],
            f"{self.prefix}/wosac_long/placement_based_metrics": [round(x, 4) for x in final_long_metrics['placement_based_metrics'].tolist()],
        })
        for k in self.field_names:
            if k not in metrics_long_dict:
                continue
            out_dict[f"{self.prefix}/wosac_long_likelihood/{k}"] = [round(x, 4) for x in metrics_long_dict[k].tolist()]

        return out_dict

    @staticmethod
    def aggregate_metrics_to_buckets(
        config: long_metrics_pb2.SimAgentMetricsConfig, # type: ignore
        metrics: long_metrics_pb2.SimAgentMetrics # type: ignore
    ) -> long_metrics_pb2.SimAgentsBucketedMetrics: # type: ignore
        """Aggregates metrics into buckets for better readability."""
        bucketed_metrics = {}
        for bucket_name, fields_in_bucket in _METRIC_FIELD_NAMES_BY_BUCKET.items():
            weighted_metric, weights_sum = 0.0, 0.0
            for field_name in fields_in_bucket:
                likelihood_field_name = field_name + '_likelihood'
                weight = getattr(config, field_name).metametric_weight
                metric_score = getattr(metrics, likelihood_field_name)
                weighted_metric += weight * metric_score
                weights_sum += weight
            if weights_sum == 0:
                weights_sum = 1  # FIXME: hack!!!
                # raise ValueError('The bucket\'s weight sum is zero. Check your metrics'
                #                 ' config.')
            bucketed_metrics[bucket_name] = weighted_metric / weights_sum

        return long_metrics_pb2.SimAgentsBucketedMetrics(
                    realism_meta_metric=metrics.metametric,
                    kinematic_metrics=bucketed_metrics['kinematic'],
                    interactive_metrics=bucketed_metrics['interactive'],
                    map_based_metrics=bucketed_metrics['map_based'],
                    placement_based_metrics=bucketed_metrics['placement_based'],
                    min_ade=metrics.min_average_displacement_error,
                    simulated_collision_rate=metrics.simulated_collision_rate,
                    simulated_offroad_rate=metrics.simulated_offroad_rate,
                )

    @staticmethod
    def aggregate_metrics_long_to_buckets(
        config: long_metrics_pb2.SimAgentMetricsConfig, # type: ignore
        metrics: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Aggregates metrics into buckets for better readability."""
        bucketed_metrics = {}
        for bucket_name, fields_in_bucket in _METRIC_FIELD_NAMES_BY_BUCKET.items():
            weighted_metric, weights_sum = torch.zeros(metrics['linear_speed_likelihood'].shape[0]), 0.0
            for field_name in fields_in_bucket:
                likelihood_field_name = field_name + '_likelihood'
                weight = getattr(config, field_name).metametric_weight
                metric_score = metrics[likelihood_field_name]
                weighted_metric += weight * metric_score
                weights_sum += weight
            if weights_sum == 0:
                weights_sum = 1  # FIXME: hack!!!
            bucketed_metrics[bucket_name] = weighted_metric / weights_sum

        return dict(
                    realism_meta_metric=metrics['metametric'],
                    kinematic_metrics=bucketed_metrics['kinematic'],
                    interactive_metrics=bucketed_metrics['interactive'],
                    map_based_metrics=bucketed_metrics['map_based'],
                    placement_based_metrics=bucketed_metrics['placement_based'],
                )

    @staticmethod
    def load_metrics_config(config_path: str = 'infgen/metrics/metric_config.textproto',
                            ) -> long_metrics_pb2.SimAgentMetricsConfig: # type: ignore
        config = long_metrics_pb2.SimAgentMetricsConfig()
        with open(config_path, 'r') as f:
            text_format.Parse(f.read(), config)
        return config

    def dumps(self, dir):
        from datetime import datetime

        timestamp = datetime.now().strftime("%m_%d_%H%M%S")

        results = self.compute()
        path = os.path.join(dir, f'{self.prefix}_{timestamp}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        CONSOLE.log(f'Saved results to [bold][yellow]{path}')


""" Preprocess Methods """

def _dump_log_metric_features(
        pkl_dir: str,
        tfrecords_dir: str,
        save_dir: str,
        transform: WaymoTargetBuilder,
        token_processor: TokenProcessor,
        scenario_id: str,
    ):

    try:

        tqdm.write(f'Processing scenario {scenario_id}')
        save_path = os.path.join(save_dir, f'{scenario_id}.pkl')
        if os.path.exists(save_path):
            return

        # load gt data
        pkl_file = os.path.join(pkl_dir, f'{scenario_id}.pkl')
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"Not found file {pkl_file}")
        tfrecord_file = os.path.join(tfrecords_dir, f'{scenario_id}.tfrecords')
        if not os.path.exists(tfrecord_file):
            raise FileNotFoundError(f"Not found file {tfrecord_file}")

        scenario_log = scenario_pb2.Scenario()
        for data in tf.data.TFRecordDataset([tfrecord_file], compression_type=''):
            scenario_log.ParseFromString(bytes(data.numpy()))
            break

        with open(pkl_file, 'rb') as f:
            log_data = pickle.load(f)

        # preprocess data
        log_data = transform._score_trained_agents(log_data)  # get `train_mask`
        log_data = token_processor._tokenize_agent(log_data)

        # convert to `JointScene` and compute features
        log_trajectories = scenario_to_trajectories(scenario_log, processed_scenario=log_data)

        evaluate_agent_ids = None
        log_features = compute_metric_features(
            log_trajectories, evaluate_agent_ids=evaluate_agent_ids, #scenario_log=scenario_log,
        )

        # save to pkl file
        with open(save_path, 'wb') as f:
            pickle.dump(log_features, f)

    except Exception as e:
        CONSOLE.log(f'[on red] Failed to process scenario {scenario_id} due to {e}.[/]')
        return


def dump_log_metric_features(log_dir, save_dir):

    buffer_size = 128

    # file loaders
    pkl_dir = os.path.join(log_dir, 'validation')
    if not os.path.exists(pkl_dir):
        raise RuntimeError(f'Not found folder {pkl_dir}')
    tfrecords_dir = os.path.join(log_dir, 'validation_tfrecords_splitted')
    if not os.path.exists(tfrecords_dir):
        raise RuntimeError(f'Not found folder {tfrecords_dir}')

    files = list(fnmatch.filter(os.listdir(pkl_dir), '*.pkl'))
    json_path = os.path.join(log_dir, 'meta_infos.json')
    meta_infos = json.load(open(json_path, 'r', encoding='utf-8'))['validation']
    CONSOLE.log(f"Loaded meta infos from {json_path}")
    available_scenarios = list(meta_infos.keys())
    df = pd.DataFrame.from_dict(meta_infos, orient='index')
    available_scenarios_set = set(available_scenarios)
    df_filtered = df[(df.index.isin(available_scenarios_set)) & (df['num_agents'] >= 8) & (df['num_agents'] < buffer_size)]
    valid_scenarios = set(df_filtered.index)
    files = list(tqdm(filter(lambda fn: fn.removesuffix('.pkl') in valid_scenarios, files), leave=False))

    scenario_ids = list(map(lambda fn: fn.removesuffix('.pkl'), files))
    CONSOLE.log(f'Loaded {len(scenario_ids)} scenarios from validation split.')

    # initialize
    transform = WaymoTargetBuilder(num_historical_steps=11,
                                   num_future_steps=80,
                                   max_num=32)

    token_processor = TokenProcessor(token_size=2048,
                                     state_token={'invalid': 0, 'valid': 1, 'enter': 2, 'exit': 3},
                                     pl2seed_radius=75)

    partial_dump_gt_metric_features = partial(
        _dump_log_metric_features, pkl_dir, tfrecords_dir, save_dir, transform, token_processor)

    for scenario_id in tqdm(scenario_ids, leave=False, desc='scenarios ...'):

        partial_dump_gt_metric_features(scenario_id)


def batch_dump_log_metric_features(log_dir, save_dir, num_workers=64):

    buffer_size = 128

    # file loaders
    pkl_dir = os.path.join(log_dir, 'validation')
    if not os.path.exists(pkl_dir):
        raise RuntimeError(f'Not found folder {pkl_dir}')
    tfrecords_dir = os.path.join(log_dir, 'validation_tfrecords_splitted')
    if not os.path.exists(tfrecords_dir):
        raise RuntimeError(f'Not found folder {tfrecords_dir}')

    files = list(fnmatch.filter(os.listdir(pkl_dir), '*.pkl'))
    json_path = os.path.join(log_dir, 'meta_infos.json')
    meta_infos = json.load(open(json_path, 'r', encoding='utf-8'))['validation']
    CONSOLE.log(f"Loaded meta infos from {json_path}")
    available_scenarios = list(meta_infos.keys())
    df = pd.DataFrame.from_dict(meta_infos, orient='index')
    available_scenarios_set = set(available_scenarios)
    df_filtered = df[(df.index.isin(available_scenarios_set)) & (df['num_agents'] >= 8) & (df['num_agents'] < buffer_size)]
    valid_scenarios = set(df_filtered.index)
    files = list(tqdm(filter(lambda fn: fn.removesuffix('.pkl') in valid_scenarios, files), leave=False))

    scenario_ids = list(map(lambda fn: fn.removesuffix('.pkl'), files))
    CONSOLE.log(f'Loaded {len(scenario_ids)} scenarios from validation split.')

    # initialize
    transform = WaymoTargetBuilder(num_historical_steps=11,
                                   num_future_steps=80,
                                   max_num=32)

    token_processor = TokenProcessor(token_size=2048,
                                     state_token={'invalid': 0, 'valid': 1, 'enter': 2, 'exit': 3},
                                     pl2seed_radius=75)

    partial_dump_gt_metric_features = partial(
        _dump_log_metric_features, pkl_dir, tfrecords_dir, save_dir, transform, token_processor)

    with multiprocessing.Pool(num_workers) as p:
        list(tqdm(p.imap_unordered(partial_dump_gt_metric_features, scenario_ids), total=len(scenario_ids)))


def aggregate_log_metric_features(load_dir):

    files = list(fnmatch.filter(os.listdir(load_dir), '*.pkl'))
    if 'total_features.pkl' in files:
        files.remove('total_features.pkl')
    CONSOLE.log(f'Loaded {len(files)} scenarios from dumpped log metric features')

    features_fields = [field.name for field in dataclasses.fields(MetricFeatures)]
    features_fields.remove('object_id')

    # load and append
    total_features = collections.defaultdict(list)
    for file in tqdm(files, leave=False, desc='scenario ...'):

        with open(os.path.join(load_dir, file), 'rb') as f:
            log_features = pickle.load(f)

        for field in features_fields:
            total_features[field].append(getattr(log_features, field))

    # aggregate
    features_info = dict()
    for field in (pbar := tqdm(features_fields, leave=False)):
        pbar.set_postfix(f=field)
        if total_features[field][0] is not None:
            total_features[field] = torch.concat(total_features[field], dim=0)  # n_agent or n_scenario 
            features_info[field] = total_features[field].shape
    CONSOLE.log(f'Aggregated log features:\n{features_info}')

    # save
    save_path = os.path.join(load_dir, 'total_features.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(total_features, f)
    CONSOLE.log(f'Saved total features to [green]{save_path}.[/]')

    return MetricFeatures(**total_features, object_id=None)


def _compute_metrics(
    metric: LongMetric,
    load_dir: str,
    verbose: bool,
    rollouts_file: str,
) -> List[long_metrics_pb2.SimAgentMetrics]: # type: ignore

    if verbose:
        print(f'Processing {rollouts_file}')

    with open(os.path.join(load_dir, rollouts_file), 'rb') as f:
        rollouts = pickle.load(f)
    # CONSOLE.log(f'Loaded rollouts from {rollouts_file}')

    return metric.compute_metrics(rollouts)


def compute_metrics(load_dir, rollouts_files):

    log_every_n_steps = 100

    metric = LongMetric('val_close_long')
    CONSOLE.log(f'metrics config:\n{metric.metrics_config}')

    i = 0
    for rollouts_file in tqdm(rollouts_files, leave=False, desc='Rollouts files ...'):

        # ! compute metrics and update
        metric.update(
            metrics=_compute_metrics(metric, load_dir, verbose=False, rollouts_file=rollouts_file)
        )

        if i % log_every_n_steps == 0:
            CONSOLE.log(f'Step={i}:\n{metric.compute()}')

        i += 1

    CONSOLE.log(f'[bold][yellow] Compute metrics completed!')
    CONSOLE.log(f'[bold][yellow] Final metrics: [/]\n {metric.compute()}')


def batch_compute_metrics(load_dir, rollouts_files, num_workers, save_dir=None):
    from queue import Queue
    from threading import Thread

    if save_dir is None:
        save_dir = load_dir

    results_buffer = Queue()

    log_every_n_steps = 20

    metric = LongMetric('val_close_long')
    CONSOLE.log(f'metrics config:\n{metric.metrics_config}')

    def _collect_result():
        while True:
            r = results_buffer.get()
            if r is None:
                break
            metric.update(metrics=r)
            results_buffer.task_done()

    collector = Thread(target=_collect_result, daemon=True)
    collector.start()

    partial_compute_metrics = partial(_compute_metrics, metric, load_dir, True)

    # ! compute metrics in batch
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # results = list(executor.map(partial_compute_metrics, rollouts_files))
        futures = [executor.submit(partial_compute_metrics, rollouts_file) for rollouts_file in rollouts_files]
        # results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), leave=False):
            results_buffer.put(future.result())

            if i % log_every_n_steps == 0:
                CONSOLE.log(f'Step={i}:\n{metric.compute()}')
                metric.dumps(save_dir)

    results_buffer.put(None)
    collector.join()

    CONSOLE.log(f'[bold][yellow] Compute metrics completed!')
    CONSOLE.log(f'[bold][yellow] Final metrics: [/]\n {metric.compute()}')

    # save results to disk
    metric.dumps(save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dump_log', action='store_true')
    parser.add_argument('--dump_sim', action='store_true')
    parser.add_argument('--aggregate_log', action='store_true')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--compute_metric', action='store_true')
    parser.add_argument('--log_dir', type=str, default='data/waymo_processed/')
    parser.add_argument('--sim_dir', type=str, default=None, required=False)
    parser.add_argument('--save_dir', type=str, default='results', required=False)
    parser.add_argument('--no_batch', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_batch', action='store_true')
    args = parser.parse_args()

    if args.dump_log:

        save_dir = os.path.join(args.log_dir, 'log_features')
        os.makedirs(save_dir, exist_ok=True)

        if args.no_batch or args.debug:
            dump_log_metric_features(args.log_dir, save_dir)
        else:
            batch_dump_log_metric_features(args.log_dir, save_dir)

    elif args.aggregate_log:

        load_dir = os.path.join(args.log_dir, 'log_features')
        aggregate_log_metric_features(load_dir)

    elif args.compute_metric:

        assert args.sim_dir is not None and os.path.exists(args.sim_dir), \
                f'Folder {args.sim_dir} does not exist!'
        rollouts_files = list(sorted(fnmatch.filter(os.listdir(args.sim_dir), 'idx_*_rollouts.pkl')))
        CONSOLE.log(f'Found {len(rollouts_files)} rollouts files.')

        os.makedirs(args.save_dir, exist_ok=True)
        if args.no_batch:
            compute_metrics(args.sim_dir, rollouts_files)

        else:
            multiprocessing.set_start_method('spawn', force=True)
            batch_compute_metrics(args.sim_dir, rollouts_files, args.num_workers, save_dir=args.save_dir)

    elif args.debug:

        debug_path = 'infgen/metrics/idx_0_0_rollouts.pkl'

        # ! for debugging
        with open(debug_path, 'rb') as f:
            rollouts = pickle.load(f)
        metric = LongMetric('debug')
        CONSOLE.log(f'metrics config: {metric.metrics_config}')

        metric.update(outputs=rollouts)
        CONSOLE.log(f'metrics:\n{metric.compute()}')


    elif args.debug_batch:

        rollouts_files = ['idx_0_rollouts.pkl'] * 1000
        CONSOLE.log(f'Found {len(rollouts_files)} rollouts files.')

        sim_dir = 'infgen/metrics/'

        os.makedirs(args.save_dir, exist_ok=True)
        multiprocessing.set_start_method('spawn', force=True)
        batch_compute_metrics(args.sim_dir, rollouts_files, args.num_workers, save_dir=args.save_dir)
