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
        linear_dist, _ = linear_encoder.minimum_straight_line_distance(test_start_phases, test_end_phases)
        circular_dist, _ = circular_encoder.minimum_straight_line_distance(test_start_phases, test_end_phases)
    np.testing.assert_allclose(linear_dist, 2 * np.pi - 0.2, atol=1e-6)
    np.testing.assert_allclose(circular_dist, 0.2, atol=1e-6)

    mixed_encoder = encoder_decoder_core.Encoder1D(embedded_dimension, 1, 1)
    test_start_phases = torch.tensor([[0, 0.1], [0.1, 0], [0.1, 0.1]], dtype=torch.get_default_dtype())
    test_end_phases = torch.tensor([[0, 2 * np.pi - 0.1], [2 * np.pi - 0.1, 0], [2 * np.pi - 0.1, 2 * np.pi - 0.1]], dtype=torch.get_default_dtype())
    with torch.no_grad():
        mixed_dist, _ = mixed_encoder.minimum_straight_line_distance(test_start_phases, test_end_phases)
    np.testing.assert_allclose(mixed_dist, [2 * np.pi - 0.2, 0.2, np.sqrt((2 * np.pi - 0.2) **2 + 0.2**2)], atol=1e-6)


test_periodic_euclid_vs_arclength()
