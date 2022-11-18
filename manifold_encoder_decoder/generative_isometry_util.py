import numpy as np
import torch


def integrated_angle_metric(angles):
    angles_rolled = np.roll(angles, 1, axis=-1)
    return np.abs(angles - angles_rolled)


def torch_integrated_angle_metric(angles):
    angles_rolled = torch.roll(angles, 1, dims=-1)
    return torch.abs(angles - angles_rolled)


def integrated_point_metric(points):
    rolled_points = torch.roll(points, 1, dims=-2)
    return torch.sum(torch.sqrt(torch.sum(torch.square(points[..., 1:, :] - rolled_points[..., 1:, :]), dim=-1) + 1e-13), dim=-1)


def densely_sample_angles(angles, n_samples):
    angles_rolled = np.roll(angles, 1, axis=1)
    return np.linspace(start=angles, stop=angles_rolled, num=n_samples, axis=-1)


def angles_to_ring(angles):
    points_x = np.cos(angles)
    points_y = np.sin(angles)
    return np.stack([points_x, points_y], axis=-1)


def torch_angles_to_ring(angles):
    points_x = torch.cos(angles)
    points_y = torch.sin(angles)
    return torch.stack([points_x, points_y], dim=-1)


@torch.jit.script
def torch_linspace(start, stop, num: int):
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    out = start[None] + steps * (stop - start)[None]
    return out


def resample_points_ring(points, n_samples):
    angles = torch.atan2(points[:, :, 1], points[:, :, 0])
    rand_mask = torch.randint(0, 2, angles.shape) * 2 * np.pi
    angles_remap = torch.remainder(angles, 2 * np.pi) - rand_mask.to(angles.device)
    rolled_angles = torch.roll(angles_remap, 1, dims=-1)
    angle_distances = torch.abs(angles_remap - rolled_angles)
    sampled_angles = torch.transpose(torch.transpose(torch_linspace(angles_remap, rolled_angles, n_samples), 0, 2), 0, 1)
    new_points = torch_angles_to_ring(sampled_angles)
    return angles, new_points, angle_distances


class CircleDistance(torch.nn.Module):
    def __init__(self, init_shift=torch.from_numpy(np.array([0.0, 0.0])), init_rad=torch.from_numpy(np.array([1.0]))):
        super().__init__()
        self.shift = torch.nn.Parameter(init_shift)
        self.rad = torch.nn.Parameter(init_rad)

    def forward(self, inputs):
        return torch.square(torch.sum(torch.square(inputs - self.shift), dim=-1) - torch.square(self.rad))

    def circularize(self, inputs):
        return (inputs - self.shift)/self.rad

    def phases(self, inputs):
        return torch.atan2(inputs[..., 1], inputs[..., 0])

