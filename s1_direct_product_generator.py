import copy
import numpy as np
import torch

import geometry_util
import encoder_decoder_core


def train(manifold_dimension, embedding_dimension, device, encoder_hidden_dim=1500, encoder_n_hidden=1, decoder_hidden_dim=1500,
          decoder_n_hidden=1, integration_resamples=20, n_points_compare=20,
          batch_size=50, n_training_iterations=3000, loss_stop_thresh=1e-4):
    encoder_net = encoder_decoder_core.Encoder1D(embedding_dimension, manifold_dimension, encoder_hidden_dim,
                                                 encoder_n_hidden).to(device)
    decoder_net = encoder_decoder_core.Decoder1D(embedding_dimension, manifold_dimension, decoder_hidden_dim,
                                                 decoder_n_hidden).to(device)

    params = list(encoder_net.parameters()) + list(decoder_net.parameters())
    opt = torch.optim.Adam(params)

    best_loss = np.inf
    best_encoder = copy.deepcopy(encoder_net)
    best_decoder = copy.deepcopy(decoder_net)
    for i in range(n_training_iterations):
        sample_phases = torch.tensor(np.random.uniform(0, 2 * np.pi, (batch_size, n_points_compare, manifold_dimension)), dtype=torch.get_default_dtype()).to(device)
        sample_points = geometry_util.torch_angles_to_ring(sample_phases)
        embedded_points = encoder_net(sample_points)
        re_encoded_points, _ = decoder_net(embedded_points)
        decoding_cost = torch.mean(torch.square(re_encoded_points - sample_points))

        rolled_sample_phases = torch.roll(sample_phases, 1, -2)
        angular_distances, model_distances = encoder_net.minimum_straight_line_distance(sample_phases, rolled_sample_phases, integration_resamples)
        normed_angular_distance = angular_distances / torch.mean(angular_distances)
        normed_model_distance = model_distances / torch.mean(model_distances)
        distance_cost = torch.mean(torch.square(normed_angular_distance - normed_model_distance))

        loss = decoding_cost + distance_cost

        loss.backward()
        opt.step()

        if loss < best_loss:
            best_loss = loss
            best_encoder = copy.deepcopy(encoder_net)
            best_decoder = copy.deepcopy(decoder_net)
            print("iteration: {}, decoding loss: {}, distance cost: {}".format(i, decoding_cost, distance_cost))

        if loss < loss_stop_thresh:
            break
    return best_encoder, best_decoder
