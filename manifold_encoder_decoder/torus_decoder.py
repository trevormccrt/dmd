import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from manifold_encoder_decoder import generative_isometry_util, encoder_decoder_core

# load some manifold data
data_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/torus/2023-01-09-17-05-24")
data = np.load(os.path.join(data_dir, "encoded_points.npy"))
true_phases = None # in general we won't have labels
# if we have label data we can plot actual vs predicted phase
true_phases = np.load(os.path.join(data_dir, "true_phases.npy"))


in_dim = np.shape(data)[1] # we will give the NN points on a ring in 2D as input
encoder_hidden_dim = 1500
encoder_n_hidden = 1
decoder_hidden_dim = encoder_hidden_dim
decoder_n_hidden = encoder_n_hidden
n_resample = 75 # how many points to use in distance integration calculation

encoder_net = encoder_decoder_core.AllPeriodicEncoder(in_dim, 2, encoder_hidden_dim, encoder_n_hidden).to(device)
decoder_net = encoder_decoder_core.AllPeriodicDecoder(in_dim, 2, decoder_hidden_dim, decoder_n_hidden).to(device)

params = list(encoder_net.parameters()) + list(decoder_net.parameters())
opt = torch.optim.Adam(params)

batch_size = 50
n_points_compare = 20 # how many points on the ring to compare distances between
n_points_length = 1000

n_epochs = 20000


sample_range = np.arange(start=0, stop=np.shape(data)[0], step=1)
best_loss = np.inf
order_reduction_thresh = 1
do_order_reduction = False
for i in range(n_epochs):
    samples = np.array([data[np.random.choice(sample_range, size=n_points_compare, replace=False), :] for i in range(batch_size)])
    data_samples = torch.tensor(samples, dtype=torch.get_default_dtype()).to(device)
    opt.zero_grad()
    decoded_points, decoded_angles = decoder_net(data_samples)
    re_encoded_points = encoder_net(decoded_points)
    decoder_loss = torch.sum(torch.square(re_encoded_points - data_samples))/(batch_size * n_points_compare)

    rolled_decoded_angles = torch.roll(decoded_angles, 1, dims=-2)
    angular_distances, model_distances = encoder_net.straight_line_distance(decoded_angles, rolled_decoded_angles,
                                                                            n_points_integrate=n_resample)
    normed_angular_distance = angular_distances / torch.mean(angular_distances)
    normed_model_distance = model_distances / torch.mean(model_distances)
    distance_cost = torch.sum(torch.square(normed_angular_distance - normed_model_distance)) / (
                batch_size * n_points_compare)

    loss = decoder_loss + distance_cost
    loss.backward()
    opt.step()

    if loss < best_loss:
        best_loss = loss
        best_encoder = copy.deepcopy(encoder_net)
        best_decoder = copy.deepcopy(decoder_net)
        print("iteration: {}, decoding loss: {}, distance cost: {}, order reduction: {}".format(i, decoder_loss, distance_cost, 0))
