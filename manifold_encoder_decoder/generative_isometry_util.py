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
    sampled_angles = torch.transpose(torch.transpose(torch_linspace(angles, torch.roll(angles, 1, dims=-1), n_samples), 0, 2), 0, 1)
    points = torch_angles_to_ring(sampled_angles)
    return angles, points


def resample_points_ring_directed(points, n_samples, direction):
    angles = torch.atan2(points[:, :, 1], points[:, :, 0])
    rand_mask = direction * 2 * np.pi
    angles_remap = torch.remainder(angles, 2 * np.pi) - rand_mask.to(angles.device)
    rolled_angles = torch.roll(angles_remap, 1, dims=-1)
    angle_distances = torch.abs(angles_remap - rolled_angles)
    sampled_angles = torch.transpose(torch.transpose(torch_linspace(angles_remap, rolled_angles, n_samples), 0, 2), 0, 1)
    new_points = torch_angles_to_ring(sampled_angles)
    return angles, new_points, angle_distances


def walk_manifold(start_phases, end_phases, n_points):
    manifold_dim = start_phases.shape[-1]
    all_start = torch.tile(torch.unsqueeze(start_phases, dim=-2), (n_points, 1))
    all_end = torch.tile(torch.unsqueeze(end_phases, dim=-2), (n_points, 1))
    walked_phases = torch_linspace(start_phases, end_phases, n_points)
    perm = list(np.roll(list(range(len(walked_phases.shape))), -1))
    walked_phases = torch.permute(walked_phases, perm)
    walked_phases = torch.transpose(walked_phases, -1, -2)
    outputs = None
    for i in range(manifold_dim):
        this_ouput = torch.cat([all_end[...,:, :i], torch.unsqueeze(walked_phases[..., :, i], -1) , all_start[..., :, i+1:]], dim=-1)
        if outputs is None:
            outputs = this_ouput
        else:
            outputs = torch.cat([outputs, this_ouput[..., 1:, :]], dim=-2)
    return outputs


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

