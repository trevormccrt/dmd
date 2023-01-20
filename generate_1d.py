import copy
import numpy as np
import torch

import geometry_util
import encoder_decoder_core


def encode_decode_cost(encoder_net, decoder_net, sample_phases):
    embedded_points = encoder_net(sample_phases)
    re_encoded_phases = decoder_net(embedded_points)
    decoding_cost = torch.mean(geometry_util.minimum_periodic_distance(sample_phases, re_encoded_phases)[0])
    return embedded_points, re_encoded_phases, decoding_cost


def train(n_circular_dimensions, n_linear_dimensions, embedding_dimension, device, encoder_hidden_dim=1500, encoder_n_hidden=1, decoder_hidden_dim=1500,
          decoder_n_hidden=1, integration_resamples=20, n_points_compare=20,
          batch_size=50, n_training_iterations=3000, loss_stop_thresh=1e-4, verbose=True):
    encoder_net = encoder_decoder_core.Encoder1D(embedding_dimension, n_circular_dimensions, n_linear_dimensions, encoder_hidden_dim,
                                                 encoder_n_hidden, regularize_latent_space=False).to(device)
    decoder_net = encoder_decoder_core.Decoder1D(embedding_dimension, n_circular_dimensions, n_linear_dimensions, decoder_hidden_dim,
                                                 decoder_n_hidden).to(device)

    params = list(encoder_net.parameters()) + list(decoder_net.parameters())
    opt = torch.optim.Adam(params)

    best_loss = np.inf
    best_encoder = copy.deepcopy(encoder_net)
    best_decoder = copy.deepcopy(decoder_net)
    best_costs = np.array([np.inf, np.inf])
    for i in range(n_training_iterations):
        sample_phases = torch.tensor(np.random.uniform(-np.pi, np.pi, (batch_size, n_points_compare, n_circular_dimensions + n_linear_dimensions)), dtype=torch.get_default_dtype()).to(device)
        embedded_points, re_encoded_phases, decoding_cost = encode_decode_cost(encoder_net, decoder_net, sample_phases)
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
            best_costs[0] = decoding_cost.cpu().detach().numpy()
            best_costs[1] = distance_cost.cpu().detach().numpy()
            if verbose:
                print("iteration: {}, decoding loss: {}, distance cost: {}".format(i, decoding_cost, distance_cost))

        if loss < loss_stop_thresh:
            break
    return best_encoder, best_decoder, best_costs
