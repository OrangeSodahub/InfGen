import torch
from torch import Tensor


def get_yaw_rotation_2d(yaw):
    """
    Gets a 2D rotation matrix given a yaw angle.

    Args:
        yaw: torch.Tensor, rotation angle in radians. Can be any shape except empty.

    Returns:
        rotation: torch.Tensor with shape [..., 2, 2], where `...` matches input shape.
    """
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rotation = torch.stack([
        torch.stack([cos_yaw, -sin_yaw], dim=-1),
        torch.stack([sin_yaw,  cos_yaw], dim=-1),
    ], dim=-2)  # Shape: [..., 2, 2]

    return rotation


def get_yaw_rotation(yaw):
    """
    Computes a 3D rotation matrix given a yaw angle (rotation around the Z-axis).

    Args:
        yaw: torch.Tensor of any shape, representing yaw angles in radians.

    Returns:
        rotation: torch.Tensor of shape [input_shape, 3, 3], representing the rotation matrices.
    """
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)

    return torch.stack([
        torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
        torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=-2)


def get_transform(rotation, translation):
    """
    Combines an NxN rotation matrix and an Nx1 translation vector into an (N+1)x(N+1) transformation matrix.

    Args:
        rotation: torch.Tensor of shape [..., N, N], representing rotation matrices.
        translation: torch.Tensor of shape [..., N], representing translation vectors.
                    This must have the same dtype as rotation.

    Returns:
        transform: torch.Tensor of shape [..., (N+1), (N+1)], representing the transformation matrices.
                   This has the same dtype as rotation.
    """
    # [..., N, 1]
    translation_n_1 = translation.unsqueeze(-1)
    
    # [..., N, N+1] - Combine rotation and translation
    transform = torch.cat([rotation, translation_n_1], dim=-1)
    
    # [..., N] - Create the last row, which is [0, 0, ..., 0, 1]
    last_row = torch.zeros_like(translation)
    last_row = torch.cat([last_row, torch.ones_like(last_row[..., :1])], dim=-1)
    
    # [..., N+1, N+1] - Append the last row to form the final transformation matrix
    transform = torch.cat([transform, last_row.unsqueeze(-2)], dim=-2)
    
    return transform


def get_upright_3d_box_corners(boxes: Tensor):
    """
    Given a set of upright 3D bounding boxes, return its 8 corner points.

    Args:
        boxes: torch.Tensor [N, 7]. The inner dims are [center{x,y,z}, length, width,
               height, heading].

    Returns:
        corners: torch.Tensor [N, 8, 3].
    """
    center_x, center_y, center_z, length, width, height, heading = boxes.unbind(dim=-1)

    # Compute rotation matrix [N, 3, 3]
    rotation = get_yaw_rotation(heading)

    # Translation [N, 3]
    translation = torch.stack([center_x, center_y, center_z], dim=-1)

    l2, w2, h2 = length * 0.5, width * 0.5, height * 0.5

    # Define the 8 corners in local coordinates [N, 8, 3]
    corners_local = torch.stack([
        torch.stack([ l2,  w2, -h2], dim=-1),
        torch.stack([-l2,  w2, -h2], dim=-1),
        torch.stack([-l2, -w2, -h2], dim=-1),
        torch.stack([ l2, -w2, -h2], dim=-1),
        torch.stack([ l2,  w2,  h2], dim=-1),
        torch.stack([-l2,  w2,  h2], dim=-1),
        torch.stack([-l2, -w2,  h2], dim=-1),
        torch.stack([ l2, -w2,  h2], dim=-1),
    ], dim=1)  # Shape: [N, 8, 3]

    # Rotate and translate the corners
    corners = torch.einsum('n i j, n k j -> n k i', rotation, corners_local) + translation.unsqueeze(1)

    return corners
