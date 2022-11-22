import numpy as np
import torch
from torch import nn

from manifold_encoder_decoder import generative_isometry_util


def wide_n_deep(input_dimension, output_dimension, n_hidden, hidden_dim):
    layers = []
    layers.append(nn.Linear(input_dimension, int(hidden_dim / 2)))
    layers.append(nn.ReLU())
    for _ in range(n_hidden):
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, int(hidden_dim / 2)))
    layers.append(nn.ReLU())
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
        return scaled_encoded, torch.atan2(scaled_encoded[..., 0], scaled_encoded[..., 1])


class AllPeriodicEncoder(nn.Module):
    def __init__(self, encoded_dimension, n_decoded_dimensions, hidden_layer_dimension=1000, n_hidden_layers=1):
        super().__init__()
        self.net = wide_n_deep(2 * n_decoded_dimensions, encoded_dimension , n_hidden_layers, hidden_layer_dimension)
        self.n_decoded_dimensions = n_decoded_dimensions

    def forward(self, x):
        flat_x = torch.reshape(x, (*x.shape[:-2], -1))
        return self.net(flat_x)

    def model_length(self, phases_start, phases_end, n_points_integrate=50, direction=None):
        if direction is None:
            direction = torch.zeros_like(phases_start).to(phases_start.device)
        rand_mask = direction * 2 * np.pi
        start_remap = torch.remainder(phases_start, 2 * np.pi) - rand_mask
        end_remap = torch.remainder(phases_end, 2 * np.pi)
        angular_length = torch.sum(torch.abs(end_remap - start_remap), dim=-1)
        resampled_angles = generative_isometry_util.walk_manifold(start_remap, end_remap, n_points_integrate)
        resampled_points = generative_isometry_util.torch_angles_to_ring(resampled_angles)
        a = resampled_angles.cpu().detach().numpy()
        encoded_points = self.forward(resampled_points)
        distance = generative_isometry_util.integrated_point_metric(encoded_points)
        return angular_length, distance
