import numpy as np
import torch

from manifold_encoder_decoder import encoder_decoder_core, geometry_util


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

