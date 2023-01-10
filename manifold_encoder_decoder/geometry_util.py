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


def periodic_reflections(point_list):
    point_list_up = torch.tile(torch.unsqueeze(point_list, -1), [3])
    return point_list_up - torch.tensor([-2 * np.pi, 0, 2 * np.pi]).to(point_list.device)


def minimum_periodic_distance(point_list_1, point_list_2):
    list_1_up = torch.flatten(torch.tile(torch.unsqueeze(periodic_reflections(point_list_1), -1), [3]), start_dim=-2)
    list_2_up = torch.flatten(torch.tile(torch.unsqueeze(periodic_reflections(point_list_2), -2), [3, 1]), start_dim=-2)
    dists = torch.square(list_2_up - list_1_up)
    min_dists, min_dist_idxs = torch.min(dists, dim=-1)
    total_min_dists = torch.sqrt(torch.sum(min_dists, dim=-1) + 1e-13)
    remapped_1 = torch.squeeze(torch.gather(list_1_up, -1, torch.unsqueeze(min_dist_idxs, -1)), -1)
    remapped_2 = torch.squeeze(torch.gather(list_2_up, -1, torch.unsqueeze(min_dist_idxs, -1)), -1)
    return total_min_dists, remapped_1, remapped_2


def closest_points_periodic(point_list):
    point_list = torch.swapaxes(point_list, -2, -1)
    n_points_compare = point_list.size(-1)
    list_a = torch.moveaxis(torch.tile(torch.unsqueeze(point_list, -1), [n_points_compare]), -3, -1)
    list_b = torch.moveaxis(torch.tile(torch.unsqueeze(point_list, -2), [n_points_compare, 1]), -3, -1)
    list_a_flat = torch.flatten(list_a, end_dim=-2)
    list_b_flat = torch.flatten(list_b, end_dim=-2)
    distances_flat, remapped_a, remapped_b = minimum_periodic_distance(list_a_flat, list_b_flat)
    remapped_a_unflat = torch.reshape(remapped_a, list_a.size())
    remapped_b_unflat = torch.reshape(remapped_b, list_b.size())
    distances = torch.reshape(distances_flat, list_a.size()[:-1])
    sorted_order = torch.argsort(distances, dim=-1)
    matches = sorted_order[..., 1]
    min_distances = torch.squeeze(torch.gather(distances, -1, torch.unsqueeze(matches, -1)), dim=-1)
    matched_points_a = torch.squeeze(torch.gather(remapped_a_unflat, -2,
                                                  torch.tile(torch.unsqueeze(torch.unsqueeze(matches, -1), -1),
                                                             [list_b.size(-1)])), -2)
    matched_points_b = torch.squeeze(torch.gather(remapped_b_unflat, -2, torch.tile(torch.unsqueeze(torch.unsqueeze(matches,-1), -1), [list_b.size(-1)])), -2)
    return min_distances, matched_points_a, matched_points_b, matches


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
