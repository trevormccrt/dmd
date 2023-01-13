import copy
import numpy as np
import torch

from manifold_encoder_decoder import geometry_util, encoder_decoder_core, s1_direct_product_decoder


def train(data, manifold_dim, device, encoder_hidden_dim=1500, encoder_n_hidden=1, decoder_hidden_dim=1500,
          decoder_n_hidden=1, integration_resamples=20, n_points_compare=20,
          batch_size=50, n_training_iterations=3000, loss_stop_thresh=1e-4, decoder_weight=1, order_red_weight=1, scrambling_weight=1):

    embedded_dim = np.shape(data)[1] # we will give the NN points on a ring in 2D as input
    encoder_net = encoder_decoder_core.AllPeriodicEncoder(embedded_dim, manifold_dim, encoder_hidden_dim, encoder_n_hidden).to(device)
    decoder_net = encoder_decoder_core.AllPeriodicDecoder(embedded_dim, manifold_dim, decoder_hidden_dim, decoder_n_hidden).to(device)

    random_weights = torch.nn.Parameter((torch.ones(manifold_dim) * -4).to(device))
    params = list(encoder_net.parameters()) + list(decoder_net.parameters()) + [random_weights]

    opt = torch.optim.Adam(params)

    sample_range = np.arange(start=0, stop=np.shape(data)[0], step=1)
    best_loss = np.inf
    best_encoder = copy.deepcopy(encoder_net)
    best_decoder = copy.deepcopy(decoder_net)
    for i in range(n_training_iterations):
        samples = np.array([data[np.random.choice(sample_range, size=n_points_compare, replace=False), :] for i in range(batch_size)])
        data_samples = torch.tensor(samples, dtype=torch.get_default_dtype()).to(device)
        opt.zero_grad()
        decoded_points, decoded_angles = decoder_net(data_samples)
        random_shifts = torch.rand(decoded_angles.size()).to(device) * 2 * np.pi
        saturated_weights = torch.sigmoid(random_weights)

        weight_cost = -1 * torch.sum(saturated_weights)
        scaled_shifts = random_shifts * saturated_weights
        scrambled_angles = decoded_angles + scaled_shifts
        scrambled_angles = torch.atan2(torch.sin(scrambled_angles), torch.cos(scrambled_angles))
        scrambled_points = geometry_util.torch_angles_to_ring(scrambled_angles)
        re_encoded_points = encoder_net(scrambled_points)
        decoder_loss = torch.mean(torch.square(re_encoded_points - data_samples))

        rolled_scrambled_angles = torch.roll(scrambled_angles, 1, dims=-2)
        angular_distances, start_remap, end_remap = geometry_util.minimum_periodic_distance(
            scrambled_angles, rolled_scrambled_angles)
        angular_distances = torch.sqrt(1e-13 + torch.sum(torch.square((start_remap - end_remap) * (1 - saturated_weights)), dim=-1))
        model_distances = encoder_net.model_length(start_remap, end_remap, integration_resamples)[1]
        normed_angular_distance = angular_distances / torch.mean(angular_distances)
        normed_model_distance = model_distances / torch.mean(model_distances)
        distance_cost = torch.mean(torch.square(normed_angular_distance - normed_model_distance))

        norm_loss = s1_direct_product_decoder.order_cost(encoder_net, re_encoded_points, scrambled_angles, integration_resamples)

        loss = (decoder_weight * decoder_loss) + distance_cost + (norm_loss * order_red_weight) + (scrambling_weight * weight_cost)
        loss.backward()
        opt.step()

        if loss < best_loss:
            best_loss = loss
            best_encoder = copy.deepcopy(encoder_net)
            best_decoder = copy.deepcopy(decoder_net)
            print("iteration: {}, decoding loss: {}, distance cost: {}, order reduction: {}, scrambling weights: {}".format(i, decoder_loss, distance_cost, norm_loss, saturated_weights))

    print('done')
    return best_encoder, best_decoder, saturated_weights