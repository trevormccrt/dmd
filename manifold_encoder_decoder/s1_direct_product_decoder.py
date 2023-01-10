import copy
import numpy as np
import os
import torch

from manifold_encoder_decoder import geometry_util, encoder_decoder_core

if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load some manifold data
data_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/torus/2023-01-09-17-05-24")
data = np.load(os.path.join(data_dir, "encoded_points.npy"))


def train(data, manifold_dim, device, encoder_hidden_dim=1500, encoder_n_hidden=1, decoder_hidden_dim=1500,
          decoder_n_hidden=1, integration_resamples=20, n_points_compare=20,
          batch_size=50, n_training_iterations=3000, loss_stop_thresh=1e-4):

    embedded_dim = np.shape(data)[1] # we will give the NN points on a ring in 2D as input
    encoder_net = encoder_decoder_core.AllPeriodicEncoder(embedded_dim, manifold_dim, encoder_hidden_dim, encoder_n_hidden).to(device)
    decoder_net = encoder_decoder_core.AllPeriodicDecoder(embedded_dim, manifold_dim, decoder_hidden_dim, decoder_n_hidden).to(device)

    params = list(encoder_net.parameters()) + list(decoder_net.parameters())
    opt = torch.optim.Adam(params)

    sample_range = np.arange(start=0, stop=np.shape(data)[0], step=1)
    best_loss = np.inf
    best_encoder = encoder_net
    best_decoder = decoder_net
    for i in range(n_training_iterations):
        samples = np.array([data[np.random.choice(sample_range, size=n_points_compare, replace=False), :] for i in range(batch_size)])
        data_samples = torch.tensor(samples, dtype=torch.get_default_dtype()).to(device)
        opt.zero_grad()
        decoded_points, decoded_angles = decoder_net(data_samples)
        re_encoded_points = encoder_net(decoded_points)
        decoder_loss = torch.mean(torch.square(re_encoded_points - data_samples))

        rolled_decoded_angles = torch.roll(decoded_angles, 1, dims=-2)
        angular_distances, model_distances = encoder_net.straight_line_distance(decoded_angles, rolled_decoded_angles,
                                                                                n_points_integrate=integration_resamples)
        normed_angular_distance = angular_distances / torch.mean(angular_distances)
        normed_model_distance = model_distances / torch.mean(model_distances)
        distance_cost = torch.mean(torch.square(normed_angular_distance - normed_model_distance))

        nearest_angular_distances, nearest_start, nearest_end, nearest_matches = geometry_util.closest_points_periodic(decoded_angles)
        nearest_re_encoded = torch.gather(re_encoded_points, -2, torch.tile(torch.unsqueeze(nearest_matches, -1), [re_encoded_points.size(-1)]))
        euclid_dist = torch.sqrt(torch.sum(torch.square(nearest_re_encoded - re_encoded_points), dim=-1) + 1e-13)
        model_arclengths = encoder_net.straight_line_distance(nearest_start, nearest_end, n_points_integrate=integration_resamples)[1]

        norm_loss = torch.mean((model_arclengths - euclid_dist)/euclid_dist)

        loss = decoder_loss + distance_cost + norm_loss
        loss.backward()
        opt.step()

        if loss < best_loss:
            best_loss = loss
            best_encoder = copy.deepcopy(encoder_net)
            best_decoder = copy.deepcopy(decoder_net)
            print("iteration: {}, decoding loss: {}, distance cost: {}, order reduction: {}".format(i, decoder_loss, distance_cost, norm_loss))

        if loss < loss_stop_thresh:
            break

    return best_encoder, best_decoder
