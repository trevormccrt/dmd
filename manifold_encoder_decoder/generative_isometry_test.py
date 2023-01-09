import numpy as np
import torch

from manifold_encoder_decoder import generative_isometry_util


def test_integrated_metric():
    n_resamples = 200
    angles = np.sort(np.random.uniform(-np.pi, np.pi, (1, 4)), axis=1)
    angular_distance = generative_isometry_util.integrated_angle_metric(angles)
    resampled_angles = generative_isometry_util.densely_sample_angles(angles, n_resamples)
    points = torch.from_numpy(generative_isometry_util.angles_to_ring(resampled_angles))
    point_metric = generative_isometry_util.integrated_point_metric(points)
    np.testing.assert_allclose(point_metric, angular_distance, rtol=1e-3)


def test_ring_interpolation():
    n_samples = 100
    actual_angles = torch.from_numpy(np.random.uniform(-np.pi, np.pi, (1, 10)))
    points = generative_isometry_util.torch_angles_to_ring(actual_angles)
    pred_angles, interpolated_points = generative_isometry_util.resample_points_ring(points, n_samples)
    np.testing.assert_allclose(actual_angles, pred_angles)
    point_metric = generative_isometry_util.integrated_point_metric(interpolated_points)
    angular_metric = generative_isometry_util.integrated_angle_metric(actual_angles)
    np.testing.assert_allclose(point_metric, angular_metric, rtol=1e-3)


def test_walk_manifold():
    n_dims = 5
    n_samples = 10
    start = np.random.uniform(-1, 1, n_dims)
    end = np.random.uniform(-1, 1, n_dims)
    start_phases = torch.from_numpy(start)
    end_phases = torch.from_numpy(end)
    walked = generative_isometry_util.walk_manifold(start_phases, end_phases, n_samples).detach().numpy()
    np.testing.assert_allclose(walked[0, :], start_phases)
    np.testing.assert_allclose(walked[-1, :], end_phases)


def test_circularization():
    with torch.no_grad():
        true_shift = np.random.uniform(-1, 1, 2)
        true_rad = np.random.uniform(1, 3, 1)
        angles = np.arange(start=0, stop=2 * np.pi, step=0.01)
        x_vals = true_rad * np.cos(angles)
        y_vals = true_rad * np.sin(angles)
        points = np.stack([x_vals, y_vals], axis=-1) + true_shift

        circularizer = generative_isometry_util.CircleDistance(torch.from_numpy(true_shift), torch.from_numpy(true_rad))
        circed = circularizer.circularize(torch.from_numpy(points)).numpy()
        rad = np.sum(np.square(circed), axis=1)
        np.testing.assert_allclose(rad, np.ones_like(rad))


def test_center_finding():
    true_shift = np.random.uniform(-1, 1, 2)
    true_rad = np.random.uniform(1, 3, 1)
    angles = np.arange(start=0, stop=2 * np.pi, step=0.01)
    x_vals = true_rad * np.cos(angles)
    y_vals = true_rad * np.sin(angles)
    points = torch.from_numpy(np.stack([x_vals, y_vals], axis=-1) + true_shift)
    n_iter = 200
    circularizer = generative_isometry_util.CircleDistance()
    opt = torch.optim.Adam(circularizer.parameters(), lr=0.05)
    for _ in range(n_iter):
        opt.zero_grad()
        distances = circularizer(points)
        loss = torch.sum(distances)
        loss.backward()
        opt.step()
    np.testing.assert_allclose(true_shift, circularizer.shift.detach().numpy(), rtol=1e-3)
    np.testing.assert_allclose(true_rad, circularizer.rad.detach().numpy(), rtol=1e-3)


def test_minimum_periodic_dist():
    test_1d_1 = torch.tensor([[2 * np.pi - 0.5], [1]])
    test_1d_2 = torch.tensor([[0.5], [1.2]])
    with torch.no_grad():
        min_dist, p1_shift, p2_shift = generative_isometry_util.minimum_periodic_distance(
            test_1d_1, test_1d_2)
    np.testing.assert_allclose(min_dist, [1, 0.2], rtol=1e-6)

    test_2d_1 = torch.tensor([[0.5, 0.5], [0, 0]])
    test_2d_2 = torch.tensor([[2 * np.pi - 0.5, 2 * np.pi - 0.5], [1, 1]])
    with torch.no_grad():
        min_dist, p1_shift, p2_shift = generative_isometry_util.minimum_periodic_distance(
            test_2d_1, test_2d_2)
    np.testing.assert_allclose(min_dist, [np.sqrt(2), np.sqrt(2)], rtol=1e-6)
