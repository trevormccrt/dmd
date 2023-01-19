import torch
from torch import nn

import geometry_util


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


class Decoder1D(nn.Module):
    def __init__(self, encoded_dimension, n_circular_dimensions, n_linear_dimensions, hidden_layer_dimension=1000, n_hidden_layers=1):
        super().__init__()
        self._circular_stop_idx = 2 * n_circular_dimensions
        self.net = wide_n_deep(encoded_dimension, self._circular_stop_idx + n_linear_dimensions, n_hidden_layers, hidden_layer_dimension)

    def forward(self, x):
        encoded = self.net(x)
        circular_outputs = torch.reshape(encoded[..., :self._circular_stop_idx], (*encoded.size()[:-1], -1, 2))
        circular_phases = torch.atan2(circular_outputs[..., 1], circular_outputs[..., 0])
        linear_outputs = encoded[..., self._circular_stop_idx:]
        linear_phases = (2 * torch.pi * torch.sigmoid(linear_outputs)) - torch.pi
        return torch.concatenate([circular_phases, linear_phases], -1)


class Encoder1D(nn.Module):
    def __init__(self, encoded_dimension, n_circular_dimensions, n_linear_dimensions, hidden_layer_dimension=1000, n_hidden_layers=1):
        super().__init__()
        self._circular_stop_idx = n_circular_dimensions
        self.net = wide_n_deep((2 * n_circular_dimensions) + n_linear_dimensions, encoded_dimension, n_hidden_layers, hidden_layer_dimension)

    def forward(self, x):
        circular_phases = x[..., :self._circular_stop_idx]
        circular_points = geometry_util.torch_angles_to_ring(circular_phases)
        flat_circular_points = torch.reshape(circular_points, (*circular_phases.size()[:-1], -1))
        linear_phases = x[..., self._circular_stop_idx:]
        return self.net(torch.concatenate([flat_circular_points, linear_phases], -1))

    def model_length(self, phases_start, phases_end, n_points_integrate=50):
        angular_length = torch.sum(torch.abs(phases_end - phases_start), dim=-1)
        resampled_angles = torch.moveaxis(geometry_util.torch_linspace(phases_start, phases_end, n_points_integrate), 0, -2)
        encoded_points = self.forward(resampled_angles)
        distance = geometry_util.integrated_point_metric(encoded_points)
        return angular_length, distance

    def minimum_straight_line_distance(self, start_phases, end_phases, n_points_integrate=50):
        angular_distances, start_remap, end_remap = geometry_util.minimum_periodic_distance(
            start_phases, end_phases)
        return angular_distances, self.model_length(start_remap, end_remap, n_points_integrate)[1]
