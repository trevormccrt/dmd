import numpy as np
import torch

from manifold_encoder_decoder import generative_isometry_util


def test_integrated_metric():
    n_resamples = 200
    angles = np.sort(np.random.uniform(-np.pi, np.pi, (1, 4)), axis=1)
    angular_distance = generative_isometry_util.integrated_angle_metric(angles)
    resampled_angles = generative_isometry_util.densely_sample_angles(angles, n_resamples)
    points_x = np.cos(resampled_angles)
    points_y = np.sin(resampled_angles)
    points = torch.tensor(np.stack([points_x, points_y], axis=-1), dtype=torch.get_default_dtype())
    point_metric = generative_isometry_util.integrated_point_metric(points)
    np.testing.assert_allclose(point_metric, angular_distance, rtol=1e-3)
