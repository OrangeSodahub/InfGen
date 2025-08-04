import torch
from torch import Tensor
from typing import Optional, Sequence

from infgen.metrics import box_utils
from infgen.metrics import geometry_utils
from infgen.metrics.protos import map_pb2

# Constant distance to apply when distances are invalid. This will avoid the
# propagation of nans and should be reduced out when taking the minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10
# Off-road threshold, i.e. smallest distance away from the road edge that is
# considered to be a off-road.
OFFROAD_DISTANCE_THRESHOLD = 0.0

# How close the start and end point of a map feature need to be for the feature
# to be considered cyclic, in m^2.
_CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0
# Scaling factor for vertical distances used when finding the closest segment to
# a query point. This prevents wrong associations in cases with under- and
# over-passes.
_Z_STRETCH_FACTOR = 3.0

_Polyline = Sequence[map_pb2.MapPoint]


def compute_distance_to_road_edge(
    *,
    center_x: Tensor,
    center_y: Tensor,
    center_z: Tensor,
    length: Tensor,
    width: Tensor,
    height: Tensor,
    heading: Tensor,
    valid: Tensor,
    evaluated_object_mask: Tensor,
    road_edge_polylines: Sequence[_Polyline],
) -> Tensor:
    """Computes the distance to the road edge for each of the evaluated objects."""
    if not road_edge_polylines:
        raise ValueError('Missing road edges.')

    # Concatenate tensors to have the same convention as `box_utils`.
    boxes = torch.stack([center_x, center_y, center_z, length, width, height, heading], dim=-1)
    num_objects, num_steps, num_features = boxes.shape
    boxes = boxes.reshape(num_objects * num_steps, num_features)

    # Compute box corners using `box_utils`, and take the xyz coords of the bottom corners.
    box_corners = box_utils.get_upright_3d_box_corners(boxes)[:, :4]
    box_corners = box_corners.reshape(num_objects, num_steps, 4, 3)

    # Gather objects in the evaluation set
    eval_corners = box_corners[evaluated_object_mask]
    num_eval_objects = eval_corners.shape[0]

    # Flatten query points.
    flat_eval_corners = eval_corners.reshape(-1, 3)

    # Tensorize road edges.
    polylines_tensor = _tensorize_polylines(road_edge_polylines)
    is_polyline_cyclic = _check_polyline_cycles(road_edge_polylines)

    # Compute distances for all query points.
    corner_distance_to_road_edge = _compute_signed_distance_to_polylines(
        xyzs=flat_eval_corners, polylines=polylines_tensor,
        is_polyline_cyclic=is_polyline_cyclic, z_stretch=_Z_STRETCH_FACTOR
    )

    # Reshape back to (num_evaluated_objects, num_steps, 4)
    corner_distance_to_road_edge = corner_distance_to_road_edge.reshape(num_eval_objects, num_steps, 4)

    # Reduce to most off-road corner.
    signed_distances = torch.max(corner_distance_to_road_edge, dim=-1)[0]

    # Mask out invalid boxes.
    eval_validity = valid[evaluated_object_mask]

    return torch.where(eval_validity, signed_distances, -EXTREMELY_LARGE_DISTANCE)


def _tensorize_polylines(polylines):
    """Stacks a sequence of polylines into a tensor.

    Args:
        polylines: A sequence of Polyline objects.

    Returns:
        A float tensor with shape (num_polylines, max_length, 4) containing xyz
        coordinates and a validity flag for all points in the polylines. Polylines
        are padded with zeros up to the length of the longest one.
    """
    polyline_tensors = []
    max_length = 0

    for polyline in polylines:
        # Skip degenerate polylines.
        if len(polyline) < 2:
            continue
        max_length = max(max_length, len(polyline))
        polyline_tensors.append(
            torch.tensor(
                [[map_point.x, map_point.y, map_point.z, 1.0] for map_point in polyline], 
                dtype=torch.float32
            )
        )

    # Pad and stack polylines
    padded_polylines = [
        torch.cat([p, torch.zeros((max_length - p.shape[0], 4), dtype=torch.float32)], dim=0)
        for p in polyline_tensors
    ]

    return torch.stack(padded_polylines, dim=0)


def _check_polyline_cycles(polylines):
    """Checks if given polylines are cyclic and returns the result as a tensor.

    Args:
        polylines: A sequence of Polyline objects.
        tolerance: A float representing the cyclic tolerance.

    Returns:
        A bool tensor with shape (num_polylines) indicating whether each polyline is cyclic.
    """
    cycles = []
    for polyline in polylines:
        # Skip degenerate polylines.
        if len(polyline) < 2:
            continue
        first_point = torch.tensor([polyline[0].x, polyline[0].y, polyline[0].z], dtype=torch.float32)
        last_point = torch.tensor([polyline[-1].x, polyline[-1].y, polyline[-1].z], dtype=torch.float32)
        cycles.append(torch.sum((first_point - last_point) ** 2) < _CYCLIC_MAP_FEATURE_TOLERANCE_M2)

    return torch.stack(cycles, dim=0)


def _compute_signed_distance_to_polylines(
    xyzs: Tensor,
    polylines: Tensor,
    is_polyline_cyclic: Optional[Tensor] = None,
    z_stretch: float = 1.0,
) -> Tensor:
    """Computes the signed distance to the 2D boundary defined by polylines.

    Negative distances correspond to being inside the boundary (e.g. on the
    road), positive distances to being outside (e.g. off-road).

    The polylines should be oriented such that port side is inside the boundary
    and starboard is outside, a.k.a counterclockwise winding order.

    The altitudes i.e. the z-coordinates of query points and polyline segments
    are used to pair each query point with the most relevant segment, that is
    closest and at the right altitude. The distances returned are 2D distances in
    the xy plane.

    Args:
      xyzs: A float Tensor of shape (num_points, 3) containing xyz coordinates of
        query points.
      polylines: Tensor with shape (num_polylines, num_segments+1, 4) containing
        sequences of xyz coordinates and validity, representing start and end
        points of consecutive segments.
      is_polyline_cyclic: A boolean Tensor with shape (num_polylines) indicating
        whether each polyline is cyclic. If None, all polylines are considered
        non-cyclic.
      z_stretch: Factor by which to scale distances over the z axis. This can be
        done to ensure edge points from the wrong level (e.g. overpasses) are not
        selected. Defaults to 1.0 (no stretching).

    Returns:
      A tensor of shape (num_points), containing the signed 2D distance from
      queried points to the nearest polyline.
    """
    num_points = xyzs.shape[0]
    assert xyzs.shape == (num_points, 3), f"Expected shape ({num_points}, 3), but got {xyzs.shape}"
    num_polylines = polylines.shape[0]
    num_segments = polylines.shape[1] - 1
    assert polylines.shape == (num_polylines, num_segments + 1, 4), \
        f"Expected shape ({num_polylines}, {num_segments + 1}, 4), but got {polylines.shape}"

    # shape: (num_polylines, num_segments+1)
    is_point_valid = polylines[:, :, 3].bool()
    # shape: (num_polylines, num_segments)
    is_segment_valid = is_point_valid[:, :-1] & is_point_valid[:, 1:]

    if is_polyline_cyclic is None:
        is_polyline_cyclic = torch.zeros(num_polylines, dtype=torch.bool)
    else:
        assert is_polyline_cyclic.shape == (num_polylines,), \
            f"Expected shape ({num_polylines},), but got {is_polyline_cyclic.shape}"

    # Get distance to each segment.
    # shape: (num_points, num_polylines, num_segments, 3)
    xyz_starts = polylines[None, :, :-1, :3]
    xyz_ends = polylines[None, :, 1:, :3]
    start_to_point = xyzs[:, None, None, :3] - xyz_starts
    start_to_end = xyz_ends - xyz_starts

    # Relative coordinate of point projection on segment.
    # shape: (num_points, num_polylines, num_segments)
    numerator = geometry_utils.dot_product_2d(
        start_to_point[..., :2], start_to_end[..., :2]
    )
    denominator = geometry_utils.dot_product_2d(
        start_to_end[..., :2], start_to_end[..., :2]
    )
    rel_t = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))

    # Negative if point is on port side of segment, positive if point on
    # starboard side of segment.
    # shape: (num_points, num_polylines, num_segments)
    n = torch.sign(
        geometry_utils.cross_product_2d(
            start_to_point[..., :2], start_to_end[..., :2]
        )
    )

    # Compute the absolute 3D distance to segment.
    # The vertical component is scaled by `z-stretch` to increase the separation
    # between different road altitudes.
    # shape: (num_points, num_polylines, num_segments, 3)
    segment_to_point = start_to_point - (
        start_to_end * torch.clamp(rel_t, 0.0, 1.0)[..., None]
    )
    stretch_vector = torch.tensor([1.0, 1.0, z_stretch], dtype=torch.float32)
    distance_to_segment_3d = torch.norm(
        segment_to_point * stretch_vector[None, None, None],
        dim=-1,
    )
    
    # Absolute planar distance to segment.
    # shape: (num_points, num_polylines, num_segments)
    distance_to_segment_2d = torch.norm(segment_to_point[..., :2], dim=-1)

    # Padded start-to-end segments.
    # shape: (num_points, num_polylines, num_segments+2, 2)
    start_to_end_padded = torch.cat(
        [
            start_to_end[:, :, -1:, :2],
            start_to_end[..., :2],
            start_to_end[:, :, :1, :2],
        ],
        dim=-2,
    )

    # shape: (num_points, num_polylines, num_segments+1)
    is_locally_convex = torch.gt(
        geometry_utils.cross_product_2d(
            start_to_end_padded[:, :, :-1], start_to_end_padded[:, :, 1:]
        ),
        0.,
    )

    # Get shifted versions of `n` and `is_segment_valid`. If the polyline is
    # cyclic, the tensors are rolled, else they are padded with their edge value.
    # shape: (num_points, num_polylines, num_segments)
    n_prior = torch.cat(
        [
            torch.where(
                is_polyline_cyclic[None, :, None],
                n[:, :, -1:],
                n[:, :, :1],
            ),
            n[:, :, :-1],
        ],
        dim=-1,
    )
    n_next = torch.cat(
        [
            n[:, :, 1:],
            torch.where(
                is_polyline_cyclic[None, :, None],
                n[:, :, :1],
                n[:, :, -1:],
            ),
        ],
        dim=-1,
    )
    # shape: (num_polylines, num_segments)
    is_prior_segment_valid = torch.cat(
        [
            torch.where(
                is_polyline_cyclic[:, None],
                is_segment_valid[:, -1:],
                is_segment_valid[:, :1],
            ),
            is_segment_valid[:, :-1],
        ],
        dim=-1,
    )
    is_next_segment_valid = torch.cat(
        [
            is_segment_valid[:, 1:],
            torch.where(
                is_polyline_cyclic[:, None],
                is_segment_valid[:, :1],
                is_segment_valid[:, -1:],
            ),
        ],
        dim=-1,
    )

    # shape: (num_points, num_polylines, num_segments)
    sign_if_before = torch.where(
        is_locally_convex[:, :, :-1],
        torch.maximum(n, n_prior),
        torch.minimum(n, n_prior),
    )
    sign_if_after = torch.where(
        is_locally_convex[:, :, 1:], torch.maximum(n, n_next), torch.minimum(n, n_next)
    )

    # shape: (num_points, num_polylines, num_segments)
    sign_to_segment = torch.where(
        (rel_t < 0.0) & is_prior_segment_valid,
        sign_if_before,
        torch.where((rel_t > 1.0) & is_next_segment_valid, sign_if_after, n),
    )

    # Flatten polylines together.
    # shape: (num_points, all_segments)
    distance_to_segment_3d = distance_to_segment_3d.view(num_points, num_polylines * num_segments)
    distance_to_segment_2d = distance_to_segment_2d.view(num_points, num_polylines * num_segments)
    sign_to_segment = sign_to_segment.view(num_points, num_polylines * num_segments)

    # Mask out invalid segments.
    # shape: (all_segments)
    is_segment_valid = is_segment_valid.view(num_polylines * num_segments)
    # shape: (num_points, all_segments)
    distance_to_segment_3d = torch.where(
        is_segment_valid[None],
        distance_to_segment_3d,
        EXTREMELY_LARGE_DISTANCE,
    )
    distance_to_segment_2d = torch.where(
        is_segment_valid[None],
        distance_to_segment_2d,
        EXTREMELY_LARGE_DISTANCE,
    )

    # Get closest segment according to absolute 3D distance and return the
    # corresponding signed 2D distance.
    # shape: (num_points)
    closest_segment_index = torch.argmin(distance_to_segment_3d, dim=-1)
    distance_sign = torch.gather(sign_to_segment, 1, closest_segment_index.unsqueeze(-1)).squeeze(-1)
    distance_2d = torch.gather(distance_to_segment_2d, 1, closest_segment_index.unsqueeze(-1)).squeeze(-1)

    return distance_sign * distance_2d
