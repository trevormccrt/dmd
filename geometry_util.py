import numpy as np
import torch


def integrated_point_metric(points):
    rolled_points = torch.roll(points, 1, dims=-2)
    return torch.sum(torch.sqrt(torch.sum(torch.square(points[..., 1:, :] - rolled_points[..., 1:, :]), dim=-1) + 1e-13), dim=-1)


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


def periodic_reflections(point_list):
    point_list_up = torch.tile(torch.unsqueeze(point_list, -1), [3])
    return point_list_up - torch.tensor([-2 * torch.pi, 0, 2 * torch.pi]).to(point_list.device)


def minimum_periodic_distance(point_list_1, point_list_2):
    list_1_up = torch.tile(torch.unsqueeze(point_list_1, -1), [3])
    list_2_up = periodic_reflections(point_list_2)
    dists = torch.square(list_2_up - list_1_up)
    min_dists, min_dist_idxs = torch.min(dists, dim=-1)
    total_min_dists = torch.sqrt(torch.sum(min_dists, dim=-1) + 1e-13)
    remapped_2 = torch.squeeze(torch.gather(list_2_up, -1, torch.unsqueeze(min_dist_idxs, -1)), -1)
    return total_min_dists, remapped_2


def closest_points_periodic(point_list):
    point_list = torch.swapaxes(point_list, -2, -1)
    n_points_compare = point_list.size(-1)
    list_a = torch.moveaxis(torch.tile(torch.unsqueeze(point_list, -1), [n_points_compare]), -3, -1)
    list_b = torch.moveaxis(torch.tile(torch.unsqueeze(point_list, -2), [n_points_compare, 1]), -3, -1)
    list_a_flat = torch.flatten(list_a, end_dim=-2)
    list_b_flat = torch.flatten(list_b, end_dim=-2)
    distances_flat,remapped_b = minimum_periodic_distance(list_a_flat, list_b_flat)
    distances = torch.reshape(distances_flat, list_a.size()[:-1])
    matches = torch.argsort(distances, dim=-1)[..., 1]
    min_distances = torch.squeeze(torch.gather(distances, -1, torch.unsqueeze(matches, -1)), dim=-1)
    matched_points = torch.squeeze(torch.gather(torch.reshape(remapped_b, list_b.size()),
                                                -2, torch.tile(torch.unsqueeze(torch.unsqueeze(matches,-1), -1),
                                                               [list_b.size(-1)])), -2)
    return min_distances, matched_points, matches


def reference_periodic_phases(phases):
    phases_refd = phases - phases[0]
    phases_refd = np.arctan2(np.sin(phases_refd), np.cos(phases_refd))
    return phases_refd * np.sign(phases_refd[int(np.shape(phases)[0]/4)])
