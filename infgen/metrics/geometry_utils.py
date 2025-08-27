import torch
import numpy as np
from torch import Tensor
from typing import Tuple


NUM_VERTICES_IN_BOX = 4


def minkowski_sum_of_box_and_box_points(box1_points: Tensor, 
                                        box2_points: Tensor) -> Tensor:
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy)."""
    point_order_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    point_order_2 = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long)

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(
        box1_points)
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(
        box2_points)

    condition = (cross_product_2d(downmost_box1_edge_direction, downmost_box2_edge_direction) >= 0.)
    condition = condition.repeat(1, 8)

    box1_point_order = torch.where(condition, point_order_2, point_order_1)
    box1_point_order = (box1_point_order + box1_start_idx) % NUM_VERTICES_IN_BOX
    ordered_box1_points = torch.gather(
        box1_points, 1, box1_point_order.unsqueeze(-1).expand(-1, -1, 2))

    box2_point_order = torch.where(condition, point_order_1, point_order_2)
    box2_point_order = (box2_point_order + box2_start_idx) % NUM_VERTICES_IN_BOX
    ordered_box2_points = torch.gather(
        box2_points, 1, box2_point_order.unsqueeze(-1).expand(-1, -1, 2))

    minkowski_sum = ordered_box1_points + ordered_box2_points

    return minkowski_sum


def signed_distance_from_point_to_convex_polygon(query_points: Tensor, polygon_points: Tensor) -> Tensor:
    """Finds the signed distances from query points to convex polygons."""
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = _get_edge_info(
        polygon_points)

    query_points = query_points.unsqueeze(1)
    vertices_to_query_vectors = query_points - polygon_points
    vertices_distances = torch.norm(vertices_to_query_vectors, dim=-1)

    edge_signed_perp_distances = torch.sum(-normal_unit_vectors * vertices_to_query_vectors, dim=-1)

    is_inside = torch.all(edge_signed_perp_distances <= 0, dim=-1)

    projection_along_tangent = torch.sum(tangent_unit_vectors * vertices_to_query_vectors, dim=-1)
    projection_along_tangent_proportion = projection_along_tangent / edge_lengths

    is_projection_on_edge = (projection_along_tangent_proportion >= 0.) & (
                             projection_along_tangent_proportion <= 1.)

    edge_perp_distances = edge_signed_perp_distances.abs()
    edge_distances = torch.where(is_projection_on_edge, edge_perp_distances, torch.tensor(np.inf))

    edge_and_vertex_distance = torch.cat([edge_distances, vertices_distances], dim=-1)
    min_distance = torch.min(edge_and_vertex_distance, dim=-1)[0]

    signed_distances = torch.where(is_inside, -min_distance, min_distance)

    return signed_distances


def _get_downmost_edge_in_box(box: Tensor) -> Tuple[Tensor, Tensor]:
    """Finds the downmost (lowest y-coordinate) edge in the box."""
    downmost_vertex_idx = torch.argmin(box[..., 1], dim=-1, keepdim=True)

    edge_start_vertex = torch.gather(box, 1, downmost_vertex_idx.unsqueeze(-1).expand(-1, -1, 2))
    edge_end_idx = (downmost_vertex_idx + 1) % NUM_VERTICES_IN_BOX
    edge_end_vertex = torch.gather(box, 1, edge_end_idx.unsqueeze(-1).expand(-1, -1, 2))
    
    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = torch.norm(downmost_edge, dim=-1, keepdim=True)
    downmost_edge_direction = downmost_edge / downmost_edge_length
    
    return downmost_vertex_idx, downmost_edge_direction


def cross_product_2d(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def dot_product_2d(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _get_edge_info(polygon_points: Tensor):
    """
    Computes properties about the edges of a polygon.

    Args:
        polygon_points: Tensor containing the vertices of each polygon, with
          shape (num_polygons, num_points_per_polygon, 2). Each polygon is assumed
          to have an equal number of vertices.

    Returns:
        tangent_unit_vectors: A unit vector in (x,y) with the same direction as
          the tangent to the edge. Shape: (num_polygons, num_points_per_polygon, 2).
        normal_unit_vectors: A unit vector in (x,y) with the same direction as
          the normal to the edge.
          Shape: (num_polygons, num_points_per_polygon, 2).
        edge_lengths: Lengths of the edges.
          Shape (num_polygons, num_points_per_polygon).
    """
    # Shift the polygon points by 1 position to get the edges.
    first_point_in_polygon = polygon_points[:, 0:1, :]  # Shape: (num_polygons, 1, 2)
    shifted_polygon_points = torch.cat([polygon_points[:, 1:, :], first_point_in_polygon], dim=1)  
    # Shape: (num_polygons, num_points_per_polygon, 2)

    edge_vectors = shifted_polygon_points - polygon_points  # Shape: (num_polygons, num_points_per_polygon, 2)
    edge_lengths = torch.norm(edge_vectors, dim=-1)  # Shape: (num_polygons, num_points_per_polygon)

    # Avoid division by zero by adding a small epsilon
    eps = torch.finfo(edge_lengths.dtype).eps
    tangent_unit_vectors = edge_vectors / (edge_lengths[..., None] + eps)  # Shape: (num_polygons, num_points_per_polygon, 2)

    normal_unit_vectors = torch.stack(
        [-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], dim=-1
    )  # Shape: (num_polygons, num_points_per_polygon, 2)

    return tangent_unit_vectors, normal_unit_vectors, edge_lengths


def rotate_2d_points(xys: Tensor, rotation_yaws: Tensor) -> Tensor:
    """Rotates `xys` counterclockwise using the `rotation_yaws`."""
    cos_yaws = torch.cos(rotation_yaws)
    sin_yaws = torch.sin(rotation_yaws)
    
    rotated_x = cos_yaws * xys[..., 0] - sin_yaws * xys[..., 1]
    rotated_y = sin_yaws * xys[..., 0] + cos_yaws * xys[..., 1]
    
    return torch.stack([rotated_x, rotated_y], axis=-1)
