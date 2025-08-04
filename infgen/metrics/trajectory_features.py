import torch
import numpy as np
from torch import Tensor
from typing import Tuple


def _wrap_angle(angle: Tensor) -> Tensor:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def central_diff(t: Tensor, pad_value: float) -> Tensor:
    pad_shape = (*t.shape[:-1], 1)
    pad_tensor = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
    diff_t = (t[..., 2:] - t[..., :-2]) / 2
    return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)


def central_logical_and(t: Tensor, pad_value: bool) -> Tensor:
    pad_shape = (*t.shape[:-1], 1)
    pad_tensor = torch.full(pad_shape, pad_value, dtype=torch.bool, device=t.device)
    diff_t = torch.logical_and(t[..., 2:], t[..., :-2])
    return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)


def compute_displacement_error(x, y, z, ref_x, ref_y, ref_z) -> Tensor:
    return torch.norm(
        torch.stack([x, y, z], dim=-1) - torch.stack([ref_x, ref_y, ref_z], dim=-1),
        p=2, dim=-1
    )


def compute_kinematic_features(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    heading: Tensor,
    seconds_per_step: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dpos = central_diff(torch.stack([x, y, z], dim=0), pad_value=np.nan)
    linear_speed = torch.norm(dpos, p=2, dim=0) / seconds_per_step
    linear_accel = central_diff(linear_speed, pad_value=np.nan) / seconds_per_step
    dh_step = _wrap_angle(central_diff(heading, pad_value=np.nan) * 2) / 2
    dh = dh_step / seconds_per_step
    d2h_step = _wrap_angle(central_diff(dh_step, pad_value=np.nan) * 2) / 2
    d2h = d2h_step / (seconds_per_step ** 2)
    return linear_speed, linear_accel, dh, d2h


def compute_kinematic_validity(valid: Tensor) -> Tuple[Tensor, Tensor]:
    speed_validity = central_logical_and(valid, pad_value=False)
    acceleration_validity = central_logical_and(speed_validity, pad_value=False)
    return speed_validity, acceleration_validity