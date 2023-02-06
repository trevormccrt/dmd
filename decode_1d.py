import copy
import numpy as np
import torch

import encoder_decoder_core


def decode_encode_cost(decoder, encoder, data):
    decoded_angles = decoder(data)
    re_encoded_points = encoder(decoded_angles)
    decoder_loss = torch.mean(torch.square(re_encoded_points - data))
    return decoded_angles, re_encoded_points, decoder_loss


def distance_costs(encoder, re_encoded_points, decoded_angles, integration_resamples):
    nearest_angular_distances, nearest_end, nearest_matches = encoder.closest_points_on_manifold(
        decoded_angles)
    nearest_re_encoded = torch.gather(re_encoded_points, -2,
                                      torch.tile(torch.unsqueeze(nearest_matches, -1), [re_encoded_points.size(-1)]))
    euclid_dist = torch.sqrt(torch.sum(torch.square(nearest_re_encoded - re_encoded_points), dim=-1) + 1e-13)
    model_arclengths = \
    encoder.minimum_straight_line_distance(decoded_angles, nearest_end, n_points_integrate=integration_resamples)[1]

    normed_angular_distance = nearest_angular_distances / torch.mean(nearest_angular_distances)
    normed_model_distance = model_arclengths / torch.mean(model_arclengths)

    extra_length_cost = torch.mean((model_arclengths - euclid_dist) / euclid_dist)
    isometry_cost = torch.mean(torch.square(normed_angular_distance - normed_model_distance))

    random_sample_phases = torch.tensor(np.random.uniform(-np.pi, np.pi, decoded_angles.shape),
                                        dtype=decoded_angles.dtype).to(decoded_angles.device)
    random_matches = encoder.closest_points_on_manifold(
        random_sample_phases)[1]
    random_arclengths = encoder.minimum_straight_line_distance(random_sample_phases, random_matches, n_points_integrate=integration_resamples)[1]
    mean_in_dist_arc = torch.mean(model_arclengths)
    mean_dist = torch.mean(random_arclengths)
    extent_cost = torch.abs((mean_dist - mean_in_dist_arc)/mean_dist)

    return extra_length_cost, isometry_cost, extent_cost


def train(data, n_circular_dimensions, n_linear_dimensions, device, encoder_hidden_dim=1500, encoder_n_hidden=1, decoder_hidden_dim=1500,
          decoder_n_hidden=1, integration_resamples=20, n_points_compare=20,
          batch_size=50, n_training_iterations=3000, loss_stop_thresh=1e-4, decoder_weight=1, order_red_weight=1, div_weight=1, verbose=True):

    embedded_dim = np.shape(data)[1] # we will give the NN points on a ring in 2D as input
    encoder_net = encoder_decoder_core.Encoder1D(embedded_dim, n_circular_dimensions, n_linear_dimensions,
                                                 encoder_hidden_dim, encoder_n_hidden).to(device)
    decoder_net = encoder_decoder_core.Decoder1D(embedded_dim, n_circular_dimensions, n_linear_dimensions,
                                                 decoder_hidden_dim, decoder_n_hidden).to(device)

    params = list(encoder_net.parameters()) + list(decoder_net.parameters())
    opt = torch.optim.Adam(params)
    sample_range = np.arange(start=0, stop=np.shape(data)[0], step=1)
    best_loss = np.inf
    best_encoder = copy.deepcopy(encoder_net)
    best_decoder = copy.deepcopy(decoder_net)
    best_costs = np.array([np.inf, np.inf, np.inf])
    for i in range(n_training_iterations):
        samples = np.array([data[np.random.choice(sample_range, size=n_points_compare, replace=False), :] for i in range(batch_size)])
        data_samples = torch.tensor(samples, dtype=torch.get_default_dtype()).to(device)
        opt.zero_grad()
        decoded_angles, re_encoded_points, decoder_loss = decode_encode_cost(decoder_net, encoder_net, data_samples)
        norm_loss, distance_cost, distribution_cost = distance_costs(encoder_net, re_encoded_points, decoded_angles, integration_resamples)
        loss = (decoder_weight * decoder_loss) + distance_cost + (norm_loss * order_red_weight) + (div_weight * distribution_cost)
        loss.backward()
        opt.step()

        if loss < best_loss:
            best_loss = loss
            best_encoder = copy.deepcopy(encoder_net)
            best_decoder = copy.deepcopy(decoder_net)
            best_costs = np.array([decoder_loss.cpu().detach().numpy(),
                                   distance_cost.cpu().detach().numpy(),
                                   norm_loss.cpu().detach().numpy()])
            if verbose:
                print("iteration: {}, decoding loss: {}, distance cost: {}, order reduction: {}, extent: {}".format(i, decoder_loss, distance_cost, norm_loss, distribution_cost))

        if loss < loss_stop_thresh:
            break

    return best_encoder, best_decoder, best_costs
