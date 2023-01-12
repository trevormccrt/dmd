import numpy as np
import torch
from torch import nn

from manifold_encoder_decoder import geometry_util


def wide_n_deep(input_dimension, output_dimension, n_hidden, hidden_dim):
    layers = []
    layers.append(nn.Linear(input_dimension, int(hidden_dim / 2)))
    layers.append(nn.Tanh())
    for _ in range(n_hidden):
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(hidden_dim, int(hidden_dim / 2)))
    layers.append(nn.Tanh())
    layers.append(nn.Linear(int(hidden_dim / 2), output_dimension))
    return nn.Sequential(*layers)


class AllPeriodicDecoder(nn.Module):
    def __init__(self, encoded_dimension, n_decoded_dimensions, hidden_layer_dimension=1000, n_hidden_layers=1):
        super().__init__()
        self.net = wide_n_deep(encoded_dimension, 2 * n_decoded_dimensions, n_hidden_layers, hidden_layer_dimension)
        self.n_decoded_dimensions = n_decoded_dimensions

    def forward(self, x):
        encoded = self.net(x)
        encoded_reshape = torch.reshape(encoded, [*encoded.shape[:-1], -1, 2])
        r_encoded = torch.sqrt(torch.sum(torch.square(encoded_reshape), dim=-1))
        scaled_encoded = torch.einsum("...k, ...kj -> ...kj", 1 / r_encoded, encoded_reshape)
        return scaled_encoded, torch.atan2(scaled_encoded[..., 1], scaled_encoded[..., 0])


class AllPeriodicEncoder(nn.Module):
    def __init__(self, encoded_dimension, n_decoded_dimensions, hidden_layer_dimension=1000, n_hidden_layers=1):
        super().__init__()
        self.net = wide_n_deep(2 * n_decoded_dimensions, encoded_dimension , n_hidden_layers, hidden_layer_dimension)
        self.n_decoded_dimensions = n_decoded_dimensions

    def forward(self, x):
        flat_x = torch.reshape(x, (*x.shape[:-2], -1))
        return self.net(flat_x)

    def model_length(self, phases_start, phases_end, n_points_integrate=50):
        angular_length = torch.sum(torch.abs(phases_end - phases_start), dim=-1)
        resampled_angles = torch.moveaxis(geometry_util.torch_linspace(phases_start, phases_end, n_points_integrate), 0, -2)
        resampled_points = geometry_util.torch_angles_to_ring(resampled_angles)
        encoded_points = self.forward(resampled_points)
        distance = geometry_util.integrated_point_metric(encoded_points)
        return angular_length, distance

    def minimum_straight_line_distance(self, start_phases, end_phases, n_points_integrate=50):
        angular_distances, start_remap, end_remap = geometry_util.minimum_periodic_distance(
            start_phases, end_phases)
        return angular_distances, self.model_length(start_remap, end_remap, n_points_integrate)[1]


class RingEncoder(AllPeriodicEncoder):
    def __init__(self, encoded_dimension, hidden_layer_dimension=1000, n_hidden_layers=1):
        super().__init__(encoded_dimension, 1, hidden_layer_dimension, n_hidden_layers)

    def total_length(self, n_points_integrate=50):
        total_model_length_start = torch.tensor([0], dtype=torch.get_default_dtype()).to(
            next(self.net.parameters()).device)
        total_model_length_end = torch.tensor([2 * np.pi], dtype=torch.get_default_dtype()).to(
            next(self.net.parameters()).device)
        return self.model_length(total_model_length_start, total_model_length_end,
                                 n_points_integrate=n_points_integrate)[1]
