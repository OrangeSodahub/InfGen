import torch
import torch.nn as nn
import numpy as np

from infgen.utils.func import wrap_angle, angle_between_2d_vectors


class Attr_Tokenizer(nn.Module):

    def __init__(self, grid_range, grid_interval, radius, angle_interval):
        super().__init__()
        self.grid_range = grid_range
        self.grid_interval = grid_interval
        self.radius = radius
        self.angle_interval = angle_interval
        self.heading = torch.pi / 2
        self._prepare_grid()

        self.grid_size = self.grid.shape[0]
        self.angle_size = int(360. / self.angle_interval)

        assert torch.all(self.grid[self.grid_size // 2] == 0.)

    def _prepare_grid(self):
        num_grid = int(self.grid_range / self.grid_interval) + 1 # Do not use '//'

        x = torch.linspace(0, num_grid - 1, steps=num_grid)
        y = torch.linspace(0, num_grid - 1, steps=num_grid)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (n^2, 2)
        grid = grid.reshape(num_grid, num_grid, 2).flip(dims=[0]).reshape(-1, 2)
        grid = (grid - x.shape[0] // 2) * self.grid_interval

        distance = (grid ** 2).sum(-1).sqrt()
        square_mask = ((distance <= self.radius) & (distance >= 0.)) | (distance == 0.)
        self.register_buffer('grid', grid[square_mask])
        self.register_buffer('dist', torch.norm(self.grid, p=2, dim=-1))
        head_vector = torch.stack([torch.tensor(self.heading).cos(), torch.tensor(self.heading).sin()])
        self.register_buffer('dir', angle_between_2d_vectors(ctr_vector=head_vector.unsqueeze(0),
                                            nbr_vector=self.grid))  # (-pi, pi]

        self.num_grid = num_grid
        self.square_mask = square_mask.numpy()

    def _apply_rot(self, x, theta):
        # x: (b, l, 2) e.g. (num_step, num_agent, 2)
        # theta: (b,) e.g. (num_step,)
        cos, sin = theta.cos(), theta.sin()
        rot_mat = torch.zeros((theta.shape[0], 2, 2), device=theta.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        x = torch.bmm(x, rot_mat)
        return x

    def pad_square(self, prob, indices=None):
        # square_mask: bool array of shape (n^2,)
        # prob: float array of shape (num_step, m)
        pad_prob = np.zeros((*prob.shape[:-1], self.square_mask.shape[0]))
        pad_prob[..., self.square_mask] = prob

        square_indices = np.arange(self.square_mask.shape[0])
        circle_indices = np.concatenate([square_indices[self.square_mask], [-1]])
        if indices is not None:
            indices = circle_indices[indices]

        return pad_prob, indices

    def get_grid(self, x, theta=None):
        x = x.reshape(-1, 2)
        grid = self.grid[None, ...].to(x.device)
        if theta is not None:
            grid = self._apply_rot(grid, (theta - self.heading).expand(x.shape[0]))
        return x[:, None] + grid

    def encode_pos(self, x, y, theta_y=None):
        assert x.dim() == y.dim() and x.shape[-1] == 2 and y.shape[-1] == 2, \
                    f"Invalid input shape x: {x.shape}, y: {y.shape}."
        centered_x = x - y
        if theta_y is not None:
            centered_x = self._apply_rot(centered_x[:, None], -(theta_y - self.heading).expand(x.shape[0]))[:, 0]
        distance = ((centered_x[:, None] - self.grid.to(x.device)[None, ...]) ** 2).sum(-1).sqrt()
        index = torch.argmin(distance, dim=-1)

        grid_xy = self.grid[index]
        offset_xy = centered_x - grid_xy

        return index.long(), offset_xy

    def decode_pos(self, index, y=None, theta_y=None):
        assert torch.all((index >= 0) & (index < self.grid_size))
        centered_x = self.grid.to(index.device)[index.long()]
        if y is not None:
            if theta_y is not None:
                centered_x = self._apply_rot(centered_x[:, None], (theta_y - self.heading).expand(centered_x.shape[0]))[:, 0]
            x = centered_x + y
            return x.float()
        return centered_x.float()

    def encode_heading(self, heading):
        heading = (wrap_angle(heading) + torch.pi) / (2 * torch.pi) * 360
        index = heading // self.angle_interval
        return index.long()

    def decode_heading(self, index):
        assert torch.all(index >= 0) and torch.all(index < (360 / self.angle_interval))
        angles = index * self.angle_interval - 180
        angles = angles / 360 * (2 * torch.pi)
        return angles.float()
