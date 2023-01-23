import numpy as np
import torch

import encoder_decoder_core, geometry_util


def test_periodic_euclid_vs_arclength():
    manifold_dim = 6
    embeded_dim = 90
    batch_size = 100
    net = encoder_decoder_core.Encoder1D(embeded_dim, 3, manifold_dim - 3)
    start_phases = torch.tensor(np.random.uniform(0, 2 * np.pi, (batch_size, manifold_dim)), dtype=torch.get_default_dtype())
    end_phases = torch.tensor(np.random.uniform(0, 2 * np.pi, (batch_size, manifold_dim)), dtype=torch.get_default_dtype())
    with torch.no_grad():
        start_embedded = net(start_phases)
        end_embedded = net(end_phases)
        euclid_dist = torch.sqrt(torch.sum(torch.square(end_embedded - start_embedded), dim=-1))
        arclength = net.minimum_straight_line_distance(start_phases, end_phases)[1]
    np.testing.assert_array_less(euclid_dist, arclength)


def test_encode_decode():
    n_circular_dimensions = 4
    n_linear_dimensions = 2
    embedded_dimension = 12
    decoder = encoder_decoder_core.Decoder1D(embedded_dimension, n_circular_dimensions, n_linear_dimensions)
    data = torch.tensor(np.random.uniform(-10, 10, (1000, embedded_dimension)), dtype=torch.get_default_dtype())
    with torch.no_grad():
        phases = decoder(data)
    assert phases.size()[-1] == n_circular_dimensions + n_linear_dimensions
    np.testing.assert_array_less(phases, np.pi)
    np.testing.assert_array_less(-1 * phases, np.pi)

    encoder = encoder_decoder_core.Encoder1D(embedded_dimension, n_circular_dimensions, n_linear_dimensions)
    with torch.no_grad():
        re_embed = encoder(phases)
    assert re_embed.size()[-1] == embedded_dimension


def test_mixed_distance():
    embedded_dimension = 12
    linear_encoder = encoder_decoder_core.Encoder1D(embedded_dimension, 0, 1)
    circular_encoder = encoder_decoder_core.Encoder1D(embedded_dimension, 1, 0)
    test_start_phases = torch.tensor(np.zeros((1, 1)) + 0.1, dtype=torch.get_default_dtype())
    test_end_phases = torch.tensor(np.ones((1, 1)) * 2 * np.pi - 0.1, dtype=torch.get_default_dtype())
    with torch.no_grad():
        linear_dist, _, _ = linear_encoder.manifold_distance(test_start_phases, test_end_phases)
        circular_dist, _, _ = circular_encoder.manifold_distance(test_start_phases, test_end_phases)
    np.testing.assert_allclose(linear_dist, 2 * np.pi - 0.2, atol=1e-6)
    np.testing.assert_allclose(circular_dist, 0.2, atol=1e-6)

    mixed_encoder = encoder_decoder_core.Encoder1D(embedded_dimension, 1, 1)
    test_start_phases = torch.tensor([[0, 0.1], [0.1, 0], [0.1, 0.1]], dtype=torch.get_default_dtype())
    test_end_phases = torch.tensor([[0, 2 * np.pi - 0.1], [2 * np.pi - 0.1, 0], [2 * np.pi - 0.1, 2 * np.pi - 0.1]], dtype=torch.get_default_dtype())
    with torch.no_grad():
        mixed_dist, _, _ = mixed_encoder.manifold_distance(test_start_phases, test_end_phases)
    np.testing.assert_allclose(mixed_dist, [2 * np.pi - 0.2, 0.2, np.sqrt((2 * np.pi - 0.2) **2 + 0.2**2)], atol=1e-6)


def test_closest_points_periodic():
    encoder = encoder_decoder_core.Encoder1D(12, 2, 0)
    points = torch.from_numpy(np.array([[0, 0.1], [0, 2 * np.pi - 0.1], [0.2, 2 * np.pi - 0.2], [0.2, 0.2]]))
    with torch.no_grad():
        min_dist, matched_a, matched_b, matches = encoder.closest_points_on_manifold(points)
        in_ring = geometry_util.torch_angles_to_ring(points)
        matched_a_ring = geometry_util.torch_angles_to_ring(matched_a)
        matched_b_ring = geometry_util.torch_angles_to_ring(matched_b)
        matches_ring = geometry_util.angles_to_ring(torch.gather(points, 0, torch.tile(torch.unsqueeze(matches, -1), [points.size(-1)])))
    np.testing.assert_allclose(matches, [1, 0, 1, 0])
    np.testing.assert_allclose(min_dist, [0.2, 0.2, np.sqrt(0.2**2 + 0.1**2), np.sqrt(0.2**2 + 0.1**2)], rtol=1e-6)
    np.testing.assert_allclose(in_ring, matched_a_ring, atol=1e-6)
    np.testing.assert_allclose(matched_b_ring, matches_ring, atol=1e-6)


def test_closet_points_mixed():
    encoder = encoder_decoder_core.Encoder1D(12, 1, 1)
    points = torch.from_numpy(np.array([[0.1, 0.1], [0.5, 0.1], [2 * np.pi - 0.1, 0.1],  [0.1, 0.5], [0.1, 2 * np.pi - 0.1], [0.5, 2 * np.pi - 0.1]]))
    with torch.no_grad():
        min_dist, matched_a, matched_b, matches = encoder.closest_points_on_manifold(points)
        in_ring = geometry_util.torch_angles_to_ring(points)
        matched_a_ring = geometry_util.torch_angles_to_ring(matched_a)
        matched_b_ring = geometry_util.torch_angles_to_ring(matched_b)
        matches_ring = geometry_util.angles_to_ring(torch.gather(points, 0, torch.tile(torch.unsqueeze(matches, -1), [points.size(-1)])))
    np.testing.assert_allclose(matches, [2, 0, 0, 0, 5, 4])
    np.testing.assert_allclose(min_dist, [0.2, 0.4, 0.2, 0.4, 0.4, 0.4], rtol=1e-6)
    np.testing.assert_allclose(in_ring, matched_a_ring, atol=1e-6)
    np.testing.assert_allclose(matched_b_ring, matches_ring, atol=1e-6)


def test_closest_points_linear():
    batch_size = 5
    n_compare = 10
    ndim = 3
    encoder = encoder_decoder_core.Encoder1D(12, 0, ndim)
    points = np.random.uniform(-np.pi, np.pi, (batch_size, n_compare, ndim))
    with torch.no_grad():
        min_distances, matched_points_a, matched_points_b, matches = encoder.closest_points_on_manifold(torch.tensor(points))
    matched_points_a = matched_points_a.numpy()
    np.testing.assert_allclose(matched_points_a, points)
    matched_points_b = matched_points_b.numpy()
    distances = np.sqrt(np.sum(np.square(points - matched_points_b), axis=-1))
    np.testing.assert_allclose(distances, min_distances)
    test_dists = np.sqrt(np.sum(np.square(points[0, :, :] - points[0, 0, :]), axis=-1))[1:]
    np.testing.assert_allclose(min_distances[0 ,0], np.min(test_dists))
    with torch.no_grad():
        gu_min, _, _, _ = geometry_util.minimum_euclidian_distance(torch.tensor(points), torch.tensor(points))
    np.testing.assert_allclose(gu_min, min_distances)
