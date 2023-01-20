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
        self.net = wide_n_deep(encoded_dimension, self._circular_stop_idx + 2 * n_linear_dimensions, n_hidden_layers, hidden_layer_dimension)

    def forward(self, x):
        encoded = self.net(x)
        circular_outputs = torch.reshape(encoded[..., :self._circular_stop_idx], (*encoded.size()[:-1], -1, 2))
        circular_phases = torch.atan2(circular_outputs[..., 1], circular_outputs[..., 0])
        linear_outputs = encoded[..., self._circular_stop_idx:]
        linear_outputs = torch.reshape(linear_outputs, (*linear_outputs.size()[:-1], -1, 2))
        linear_phases = torch.pi * torch.cos(torch.atan2(linear_outputs[..., 1], linear_outputs[..., 0]))
        return torch.concatenate([circular_phases, linear_phases], -1)


class Encoder1D(nn.Module):
    def __init__(self, encoded_dimension, n_circular_dimensions, n_linear_dimensions, hidden_layer_dimension=1000,
                 n_hidden_layers=1, regularize_latent_space=False):
        super().__init__()
        self._circular_stop_idx = n_circular_dimensions
        net_out_dim = encoded_dimension
        if regularize_latent_space:
            net_out_dim += 1
        self.net = wide_n_deep((2 * n_circular_dimensions) + n_linear_dimensions, net_out_dim, n_hidden_layers, hidden_layer_dimension)
        self.regularizing = regularize_latent_space

    def forward(self, x):
        circular_phases = x[..., :self._circular_stop_idx]
        circular_points = geometry_util.torch_angles_to_ring(circular_phases)
        flat_circular_points = torch.reshape(circular_points, (*circular_phases.size()[:-1], -1))
        linear_phases = x[..., self._circular_stop_idx:]
        out = self.net(torch.concatenate([flat_circular_points, linear_phases], -1))
        if self.regularizing:
            norms = torch.sqrt(torch.sum(torch.square(out), -1) + 1e-13)
            out = torch.einsum("...k, ... -> ...k", out, 1/norms)
            return out[..., :-1]
        return out

    def model_length(self, phases_start, phases_end, n_points_integrate=50):
        angular_length = torch.sum(torch.abs(phases_end - phases_start), dim=-1)
        resampled_angles = torch.moveaxis(geometry_util.torch_linspace(phases_start, phases_end, n_points_integrate), 0, -2)
        encoded_points = self.forward(resampled_angles)
        distance = geometry_util.integrated_point_metric(encoded_points)
        return angular_length, distance

    def manifold_distance(self, start_phases, end_phases):
        _, start_remap, end_remap = geometry_util.minimum_periodic_distance(
            start_phases[..., :self._circular_stop_idx], end_phases[..., :self._circular_stop_idx])
        angular_terms = start_remap - end_remap
        linear_terms = start_phases[..., self._circular_stop_idx:] - end_phases[..., self._circular_stop_idx:]
        all_distances = torch.cat([angular_terms, linear_terms], dim=-1)
        all_start = torch.cat([start_remap, start_phases[..., self._circular_stop_idx:]], -1)
        all_end = torch.cat([end_remap, end_phases[..., self._circular_stop_idx:]], -1)
        phase_distances = torch.sqrt(torch.sum(torch.square(all_distances), -1) + 1e-13)
        return phase_distances, all_start, all_end


    def closest_points_on_manifold(self):


    def minimum_straight_line_distance(self, start_phases, end_phases, n_points_integrate=50):
        phase_distances, all_start, all_end = self.manifold_distance(start_phases, end_phases)
        return phase_distances, self.model_length(all_start, all_end, n_points_integrate)[1]
