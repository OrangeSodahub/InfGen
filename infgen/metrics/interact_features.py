import math
import torch
from torch import Tensor

from infgen.metrics import box_utils
from infgen.metrics import geometry_utils
from infgen.metrics import trajectory_features


EXTREMELY_LARGE_DISTANCE = 1e10
COLLISION_DISTANCE_THRESHOLD = 0.0
CORNER_ROUNDING_FACTOR = 0.7
MAX_HEADING_DIFF = math.radians(75.0)
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = math.radians(10.0)
SMALL_OVERLAP_THRESHOLD = 0.5
MAXIMUM_TIME_TO_COLLISION = 5.0


def compute_distance_to_nearest_object(
    center_x: Tensor,
    center_y: Tensor,
    center_z: Tensor,
    length: Tensor,
    width: Tensor,
    height: Tensor,
    heading: Tensor,
    valid: Tensor,
    evaluated_object_mask: Tensor,
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> Tensor:
    """Computes the distance to nearest object for each of the evaluated objects."""
    boxes = torch.stack([center_x, center_y, center_z, length, width, height, heading], dim=-1)
    num_objects, num_steps, num_features = boxes.shape

    shrinking_distance = (torch.minimum(boxes[:, :, 3], boxes[:, :, 4]) * corner_rounding_factor / 2.)

    boxes = torch.cat([
        boxes[:, :, :3],
        boxes[:, :, 3:4] - 2.0 * shrinking_distance[..., None],
        boxes[:, :, 4:5] - 2.0 * shrinking_distance[..., None],
        boxes[:, :, 5:]
    ], dim=2)

    boxes = boxes.reshape(num_objects * num_steps, num_features)

    box_corners = box_utils.get_upright_3d_box_corners(boxes)[:, :4, :2]
    box_corners = box_corners.reshape(num_objects, num_steps, 4, 2)

    eval_corners = box_corners[evaluated_object_mask]
    num_eval_objects = eval_corners.shape[0]
    other_corners = box_corners[~evaluated_object_mask]
    all_corners = torch.cat([eval_corners, other_corners], dim=0)

    eval_corners = eval_corners.unsqueeze(1).expand(num_eval_objects, num_objects, num_steps, 4, 2)
    all_corners = all_corners.unsqueeze(0).expand(num_eval_objects, num_objects, num_steps, 4, 2)

    eval_corners = eval_corners.reshape(num_eval_objects * num_objects * num_steps, 4, 2)
    all_corners = all_corners.reshape(num_eval_objects * num_objects * num_steps, 4, 2)

    neg_all_corners = -1.0 * all_corners
    minkowski_sum = geometry_utils.minkowski_sum_of_box_and_box_points(
        box1_points=eval_corners, box2_points=neg_all_corners,
    )

    assert minkowski_sum.shape[1:] == (8, 2), f"Shape mismatch: {minkowski_sum.shape}, expected (*, 8, 2)"
    signed_distances_flat = (
        geometry_utils.signed_distance_from_point_to_convex_polygon(
            query_points=torch.zeros_like(minkowski_sum[:, 0, :]),
            polygon_points=minkowski_sum,
        )
    )

    signed_distances = signed_distances_flat.reshape(num_eval_objects, num_objects, num_steps)

    eval_shrinking_distance = shrinking_distance[evaluated_object_mask]
    other_shrinking_distance = shrinking_distance[~evaluated_object_mask]
    all_shrinking_distance = torch.cat([eval_shrinking_distance, other_shrinking_distance], dim=0)

    signed_distances -= eval_shrinking_distance.unsqueeze(1)
    signed_distances -= all_shrinking_distance.unsqueeze(0)

    self_mask = torch.eye(num_eval_objects, num_objects, dtype=torch.float32)[:, :, None]
    signed_distances = signed_distances + self_mask * EXTREMELY_LARGE_DISTANCE

    eval_validity = valid[evaluated_object_mask]
    other_validity = valid[~evaluated_object_mask]
    all_validity = torch.cat([eval_validity, other_validity], dim=0)

    valid_mask = eval_validity.unsqueeze(1) & all_validity.unsqueeze(0)

    signed_distances = torch.where(valid_mask, signed_distances, EXTREMELY_LARGE_DISTANCE)

    return torch.min(signed_distances, dim=1).values


def compute_time_to_collision_with_object_in_front(
    *,
    center_x: Tensor,
    center_y: Tensor,
    length: Tensor,
    width: Tensor,
    heading: Tensor,
    valid: Tensor,
    evaluated_object_mask: Tensor,
    seconds_per_step: float,
) -> Tensor:
    """Computes the time-to-collision of the evaluated objects."""
    # `speed` shape: (num_objects, num_steps)
    speed = trajectory_features.compute_kinematic_features(
        x=center_x,
        y=center_y,
        z=torch.zeros_like(center_x),
        heading=heading,
        seconds_per_step=seconds_per_step,
    )[0]

    boxes = torch.stack([center_x, center_y, length, width, heading, speed], dim=-1)
    boxes = boxes.permute(1, 0, 2)  # (num_steps, num_objects, 6)
    valid = valid.permute(1, 0)

    eval_boxes = boxes[:, evaluated_object_mask]
    ego_xy, ego_sizes, ego_yaw, ego_speed = torch.split(eval_boxes, [2, 2, 1, 1], dim=-1)
    other_xy, other_sizes, other_yaw, _ = torch.split(boxes, [2, 2, 1, 1], dim=-1)

    yaw_diff = torch.abs(other_yaw[:, None] - ego_yaw[:, :, None])
    yaw_diff_cos = torch.cos(yaw_diff)
    yaw_diff_sin = torch.sin(yaw_diff)

    other_long_offset = geometry_utils.dot_product_2d(
        other_sizes[:, None] / 2.0, torch.abs(torch.cat([yaw_diff_cos, yaw_diff_sin], dim=-1))
    )
    other_lat_offset = geometry_utils.dot_product_2d(
        other_sizes[:, None] / 2.0, torch.abs(torch.cat([yaw_diff_sin, yaw_diff_cos], dim=-1))
    )

    other_relative_xy = geometry_utils.rotate_2d_points(
        (other_xy[:, None] - ego_xy[:, :, None]), -ego_yaw
    )
    
    long_distance = (
        other_relative_xy[..., 0] - ego_sizes[:, :, None, 0] / 2.0 - other_long_offset
    )
    lat_overlap = (
        torch.abs(other_relative_xy[..., 1]) - ego_sizes[:, :, None, 1] / 2.0 - other_lat_offset
    )
    
    following_mask = _get_object_following_mask(
        long_distance, lat_overlap, yaw_diff[..., 0]
    )
    valid_mask = valid[:, None] & following_mask
    
    masked_long_distance = (
        long_distance + (1.0 - valid_mask.float()) * EXTREMELY_LARGE_DISTANCE
    )
    
    box_ahead_index = masked_long_distance.argmin(dim=-1)
    distance_to_box_ahead = torch.gather(
        masked_long_distance, -1, box_ahead_index.unsqueeze(-1)
    ).squeeze(-1)
    
    speed_broadcast = speed.T[:, None, :].expand_as(masked_long_distance)
    box_ahead_speed = torch.gather(speed_broadcast, -1, box_ahead_index.unsqueeze(-1)).squeeze(-1)
    
    rel_speed = ego_speed[..., 0] - box_ahead_speed
    time_to_collision = torch.where(
        rel_speed > 0.0,
        torch.minimum(distance_to_box_ahead / rel_speed,
                      torch.tensor(MAXIMUM_TIME_TO_COLLISION)),  # the float will be broadcasted automatically
        MAXIMUM_TIME_TO_COLLISION,
    )
    
    return time_to_collision.T


def _get_object_following_mask(
    longitudinal_distance: Tensor,
    lateral_overlap: Tensor,
    yaw_diff: Tensor,
) -> Tensor:
    """Checks whether objects satisfy criteria for following another object.

    An object on which the criteria are applied is called "ego object" in this
    function to disambiguate it from the other objects acting as obstacles.

    An "ego" object is considered to be following another object if they satisfy
    conditions on the longitudinal distance, lateral overlap, and yaw alignment
    between them.

    Args:
        longitudinal_distance: A float Tensor with shape (batch_dim, num_egos,
          num_others) containing longitudinal distances from the back side of each
          ego box to every other box.
        lateral_overlap: A float Tensor with shape (batch_dim, num_egos, num_others)
          containing lateral overlaps of other boxes over the trails of ego boxes.
        yaw_diff: A float Tensor with shape (batch_dim, num_egos, num_others)
          containing absolute yaw differences between egos and other boxes.

    Returns:
        A boolean Tensor with shape (batch_dim, num_egos, num_others) indicating for
        each ego box if it is following the other boxes.
    """
    # Check object is ahead of the ego box's front.
    valid_mask = longitudinal_distance > 0.0

    # Check alignment.
    valid_mask = torch.logical_and(valid_mask, yaw_diff <= MAX_HEADING_DIFF)

    # Check object is directly ahead of the ego box.
    valid_mask = torch.logical_and(valid_mask, lateral_overlap < 0.0)

    # Check strict alignment if the overlap is small.
    # `lateral_overlap` is a signed penetration distance: it is negative when the
    # boxes have an actual lateral overlap.
    return torch.logical_and(
        valid_mask,
        torch.logical_or(
            lateral_overlap < -SMALL_OVERLAP_THRESHOLD,
            yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP,
        ),
    )