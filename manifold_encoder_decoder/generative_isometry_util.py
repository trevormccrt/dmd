import numpy as np
import torch


def integrated_angle_metric(angles):
    angles_rolled = np.roll(angles, 1, axis=-1)
    return np.abs(angles - angles_rolled)


def integrated_point_metric(points):
    rolled_points = torch.roll(points, 1, dims=-2)
    return torch.sum(torch.sqrt(torch.sum(torch.square(points[:, :, 1:, :] - rolled_points[:, :, 1:, :]), dim=-1) + 1e-13), dim=-1)


def densely_sample_angles(angles, n_samples):
    angles_rolled = np.roll(angles, 1, axis=1)
    return np.linspace(start=angles, stop=angles_rolled, num=n_samples, axis=-1)


def angles_to_ring(angles):
    points_x = np.cos(angles)
    points_y = np.sin(angles)
    return np.stack([points_x, points_y], axis=-1)
