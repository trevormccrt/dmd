import numpy as np
import torch

from manifold_encoder_decoder import geometry_util
from manifold_encoder_decoder.geometry_util import torch_linspace, torch_angles_to_ring


def _integrated_angle_metric(angles):
    angles_rolled = np.roll(angles, 1, axis=-1)
    return np.abs(angles - angles_rolled)


def _densely_sample_angles(angles, n_samples):
    angles_rolled = np.roll(angles, 1, axis=1)
    return np.linspace(start=angles, stop=angles_rolled, num=n_samples, axis=-1)


def _resample_points_ring(points, n_samples):
    angles = torch.atan2(points[:, :, 1], points[:, :, 0])
    sampled_angles = torch.transpose(torch.transpose(torch_linspace(angles, torch.roll(angles, 1, dims=-1), n_samples), 0, 2), 0, 1)
    points = torch_angles_to_ring(sampled_angles)
    return angles, points


def test_integrated_metric():
    n_resamples = 200
    angles = np.sort(np.random.uniform(-np.pi, np.pi, (1, 4)), axis=1)
    angular_distance = _integrated_angle_metric(angles)
    resampled_angles = _densely_sample_angles(angles, n_resamples)
    points = torch.from_numpy(geometry_util.angles_to_ring(resampled_angles))
    point_metric = geometry_util.integrated_point_metric(points)
    np.testing.assert_allclose(point_metric, angular_distance, rtol=1e-3)


def test_ring_interpolation():
    n_samples = 100
    actual_angles = torch.from_numpy(np.random.uniform(-np.pi, np.pi, (1, 10)))
    points = geometry_util.torch_angles_to_ring(actual_angles)
    pred_angles, interpolated_points = _resample_points_ring(points, n_samples)
    np.testing.assert_allclose(actual_angles, pred_angles)
    point_metric = geometry_util.integrated_point_metric(interpolated_points)
    angular_metric = _integrated_angle_metric(actual_angles)
    np.testing.assert_allclose(point_metric, angular_metric, rtol=1e-3)


def test_minimum_periodic_dist():
    test_1d_1 = torch.tensor([[2 * np.pi - 0.5], [1]])
    test_1d_2 = torch.tensor([[0.5], [1.2]])
    with torch.no_grad():
        min_dist, p1_shift, p2_shift = geometry_util.minimum_periodic_distance(
            test_1d_1, test_1d_2)
    np.testing.assert_allclose(min_dist, [1, 0.2], rtol=1e-6)

    test_2d_1 = torch.tensor([[0.5, 0.5], [0, 0]])
    test_2d_2 = torch.tensor([[2 * np.pi - 0.5, 2 * np.pi - 0.5], [1, 1]])
    with torch.no_grad():
        min_dist, p1_shift, p2_shift = geometry_util.minimum_periodic_distance(
            test_2d_1, test_2d_2)
    np.testing.assert_allclose(min_dist, [np.sqrt(2), np.sqrt(2)], rtol=1e-6)


def closest_points_periodic_test():
    points = torch.from_numpy(np.array([[0, 0.1], [0, 2 * np.pi - 0.1], [0.2, 2 * np.pi - 0.2], [0.2, 0.2]]))
    with torch.no_grad():
        min_dist, matched_a, matched_b, matches = geometry_util.closest_points_periodic(points)
        in_ring = geometry_util.torch_angles_to_ring(points)
        matched_a_ring = geometry_util.torch_angles_to_ring(matched_a)
        matched_b_ring = geometry_util.torch_angles_to_ring(matched_b)
        matches_ring = geometry_util.angles_to_ring(torch.gather(points, 0, torch.tile(torch.unsqueeze(matches, -1), [points.size(-1)])))
    np.testing.assert_allclose(matches, [1, 0, 1, 0])
    np.testing.assert_allclose(min_dist, [0.2, 0.2, np.sqrt(0.2**2 + 0.1**2), np.sqrt(0.2**2 + 0.1**2)], rtol=1e-6)
    np.testing.assert_allclose(in_ring, matched_a_ring, atol=1e-6)
    np.testing.assert_allclose(matched_b_ring, matches_ring, atol=1e-6)
