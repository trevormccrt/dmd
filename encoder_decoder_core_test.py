import numpy as np
import torch

import encoder_decoder_core, geometry_util


def test_periodic_euclid_vs_arclength():
    manifold_dim = 6
    embeded_dim = 90
    batch_size = 100
    net = encoder_decoder_core.AllPeriodicEncoder(embeded_dim, manifold_dim)
    start_phases = torch.tensor(np.random.uniform(0, 2 * np.pi, (batch_size, manifold_dim)), dtype=torch.get_default_dtype())
    end_phases = torch.tensor(np.random.uniform(0, 2 * np.pi, (batch_size, manifold_dim)), dtype=torch.get_default_dtype())
    with torch.no_grad():
        start_points = geometry_util.torch_angles_to_ring(start_phases)
        end_points = geometry_util.torch_angles_to_ring(end_phases)
        start_embedded = net(start_points)
        end_embedded = net(end_points)
        euclid_dist = torch.sqrt(torch.sum(torch.square(end_embedded - start_embedded), dim=-1))
        arclength = net.minimum_straight_line_distance(start_phases, end_phases)[1]
    np.testing.assert_array_less(euclid_dist, arclength)


def test_circular_vs_linear():
    n_circular_dimensions = 4
    n_linear_dimensions = 2
    embedded_dimension = 12
    decoder = encoder_decoder_core.Decoder1D(embedded_dimension, n_circular_dimensions, n_linear_dimensions)
    data = torch.tensor(np.random.uniform(-10, 10, (1000, embedded_dimension)), dtype=torch.get_default_dtype())
    with torch.no_grad():
        circ_phases, linear_phases = decoder(data)
    all_phases = torch.concatenate([circ_phases, linear_phases], -1)
    np.testing.assert_array_less(all_phases, np.pi)
    np.testing.assert_array_less(-1 * all_phases, np.pi)



test_circular_vs_linear()

