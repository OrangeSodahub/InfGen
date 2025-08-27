import torch
from torch import Tensor
from typing import Optional, Sequence, List


def compute_num_placement(
    valid: Tensor,  # [n_agent, n_step]
    state: Tensor,  # [n_agent, n_step]
    av_id: int,
    object_id: Tensor,
    agent_state: List[str],
) -> Tensor:

    enter_state = agent_state.index('enter')
    exit_state = agent_state.index('exit')

    av_index = object_id.tolist().index(av_id)
    state[av_index] = -1  # we do not incorporate the sdc

    is_bos = state == enter_state
    is_eos = state == exit_state

    num_bos = torch.sum(is_bos, dim=0)
    num_eos = torch.sum(is_eos, dim=0)

    return num_bos, num_eos


def compute_distance_placement(
    position: Tensor,
    state: Tensor,
    valid: Tensor,
    av_id: int,
    object_id: Tensor,
    agent_state: List[str],
) -> Tensor:

    enter_state = agent_state.index('enter')
    exit_state = agent_state.index('exit')

    av_index = object_id.tolist().index(av_id)
    state[av_index] = -1  # we do not incorporate the sdc
    distance = torch.norm(position - position[av_index : av_index + 1], p=2, dim=-1)

    bos_distance = distance * (state == enter_state)
    eos_distance = distance * (state == exit_state)

    return bos_distance, eos_distance
