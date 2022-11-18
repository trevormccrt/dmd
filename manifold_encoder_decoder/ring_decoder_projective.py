import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from manifold_encoder_decoder import generative_isometry_util

# load some manifold data
data_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/ike_data_1")
data = np.load(os.path.join(data_dir, "encoded_points.npy"))
true_phases = None # in general we won't have labels
# if we have label data we can plot actual vs predicted phase
#true_phases = np.load(os.path.join(data_dir, "true_phases.npy"))


in_dim = np.shape(data)[1] # we will give the NN points on a ring in 2D as input
compressed_dim = 2 # whatever dimension we want to encode into
encoder_hidden_dim = 1000
encoder_n_hidden = 1
decoder_hidden_dim = encoder_hidden_dim
decoder_n_hidden = encoder_n_hidden
n_resample = 200 # how many points to use in distance integration calculation

# build the encoder NN
encoder_layers = []
encoder_layers.append(nn.Linear(in_dim, int(encoder_hidden_dim/2)))
encoder_layers.append(nn.Tanh())
for _ in range(encoder_n_hidden):
    encoder_layers.append(nn.LazyLinear(encoder_hidden_dim))
    encoder_layers.append(nn.Tanh())
encoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
encoder_layers.append(nn.Tanh())
encoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), compressed_dim))
encoder_net = nn.Sequential(*encoder_layers).to(device)

# build the decoder NN
decoder_layers = []
decoder_layers.append(nn.Linear(compressed_dim, int(encoder_hidden_dim/2)))
decoder_layers.append(nn.Tanh())
for _ in range(encoder_n_hidden):
    decoder_layers.append(nn.LazyLinear( encoder_hidden_dim))
    decoder_layers.append(nn.Tanh())
decoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
decoder_layers.append(nn.Tanh())
decoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), in_dim))
decoder_net = nn.Sequential(*decoder_layers).to(device)


params = list(encoder_net.parameters()) + list(decoder_net.parameters())
opt = torch.optim.Adam(params)

batch_size = 50
n_points_compare = 20 # how many points on the ring to compare distances between
n_points_length = 1000

n_epochs = 10000

encoder_losses = []
decoder_losses = []

iso_loss = False
sample_range = np.arange(start=0, stop=np.shape(data)[0], step=1)
best_loss = np.inf

model_dist_calc_angles = np.linspace(start=0, stop=2 * np.pi, num=1000)
dist_calc_points = torch.tensor(generative_isometry_util.angles_to_ring(model_dist_calc_angles), dtype=torch.get_default_dtype()).to(device)

order_reduction_thresh = 0.1
do_order_reduction = False
for _ in range(n_epochs):
    samples = np.array([data[np.random.choice(sample_range, size=n_points_compare, replace=False), :] for i in range(batch_size)])
    data_samples = torch.tensor(samples, dtype=torch.get_default_dtype()).to(device)
    opt.zero_grad()
    encoded = encoder_net(data_samples)

    # send to r=1
    r_encoded = torch.sqrt(torch.sum(torch.square(encoded), dim=-1))
    normed_encoded = torch.einsum("...k, ...kj -> ...kj", 1/r_encoded, encoded)
    decoded = decoder_net(normed_encoded)
    decoder_loss = torch.sum(torch.square(decoded - data_samples))/(batch_size * n_points_compare)



    encoded_angles, resampled_encoded_points, angular_distance = generative_isometry_util.resample_points_ring(normed_encoded, n_resample)
    normed_angular_distance = angular_distance/torch.mean(angular_distance)
    decoded_resample = decoder_net(resampled_encoded_points)
    decoder_distance = generative_isometry_util.integrated_point_metric(decoded_resample)
    normed_decoder_distance = decoder_distance/torch.mean(decoder_distance)
    distance_cost = 20 * torch.sum(torch.square(normed_angular_distance - normed_decoder_distance))/(batch_size * n_points_compare)

    ordered_encoded, _ = torch.sort(encoded_angles, dim=-1)
    ordered_points = generative_isometry_util.torch_angles_to_ring(ordered_encoded)
    decoded_in_order = decoder_net(ordered_points)
    total_euclid_distance = torch.sum(
        torch.sqrt(torch.sum(torch.square(decoded_in_order - torch.roll(decoded_in_order, 1, dims=-2)), dim=-1) + 1e-13),
        dim=-1)
    dist_points_decoded = decoder_net(dist_calc_points)
    model_distance = torch.sum(torch.sqrt(
        torch.sum(torch.square(dist_points_decoded - torch.roll(dist_points_decoded, 1, dims=-2)), dim=-1) + 1e-13), dim=-1)
    capacity_factor = torch.mean((model_distance - total_euclid_distance) / total_euclid_distance)

    loss = decoder_loss + distance_cost

    if loss < order_reduction_thresh and not do_order_reduction:
        do_order_reduction = True
        print("adding order reduction")
        best_loss = np.inf

    if do_order_reduction:
        loss += capacity_factor

    if loss < best_loss:
        best_loss = loss
        best_encoder = copy.deepcopy(encoder_net)
        best_decoder = copy.deepcopy(decoder_net)
        print("decoding loss: {}, distance cost: {}, order reduction: {}".format( decoder_loss, distance_cost, capacity_factor))
    loss.backward()
    opt.step()


with torch.no_grad():
    encoded = best_encoder(torch.tensor(data, dtype=torch.get_default_dtype()).to(device))
    r_encoded = torch.sqrt(torch.sum(torch.square(encoded), dim=-1))
    normed_encoded = torch.einsum("...k, ...kj -> ...kj", 1 / r_encoded, encoded)
    encoded_angles, resampled_encoded_points, encoded_distance = generative_isometry_util.resample_points_ring(torch.unsqueeze(normed_encoded, 0), n_resample)
    normed_angular_distance = encoded_distance / torch.mean(encoded_distance)
    decoded = best_decoder(resampled_encoded_points)
    decoded_distances = generative_isometry_util.integrated_point_metric(decoded)
    normed_decoder_distance = decoded_distances / torch.mean(decoded_distances)
    rel_error = (normed_angular_distance - normed_decoder_distance)/normed_decoder_distance

normed_encoded = normed_encoded.cpu().numpy()
phases = encoded_angles.cpu().numpy()
rel_error = np.squeeze(rel_error.cpu().numpy())
angles = np.arange(start=0, stop=2 * np.pi, step=0.01)
fig, axs = plt.subplots(ncols=2)

axs[0].scatter(normed_encoded[:, 0], normed_encoded[:, 1], label="Mapped Data")
axs[0].plot(np.cos(angles), np.sin(angles), color="black", linestyle="--", label="r=1 circle")
axs[0].set_title("Data Mapped to 1D Ring")
axs[0].legend()
axs[1].hist(rel_error)
axs[1].set_xlabel("Percent Distance Difference")
axs[1].set_ylabel("Counts")
axs[1].set_xlabel("Violation of Isometry")

out_dir = os.path.join(data_dir, "processing_results/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
fig.savefig(os.path.join(out_dir, "mapping.png"))


if true_phases is not None:
    true_phases_refed = true_phases - true_phases[0]
    true_phases_refed = np.arctan2(np.sin(true_phases_refed), np.cos(true_phases_refed))
    true_phases_refed = true_phases_refed * np.sign(true_phases_refed[1])
    phases = phases[0]
    estimated_phases = phases - phases[0]
    estimated_phases_refed = np.arctan2(np.sin(estimated_phases), np.cos(estimated_phases))
    estimated_phases_refed = estimated_phases_refed * np.sign(estimated_phases_refed[1])
    fig, axs = plt.subplots()
    axs.scatter(true_phases_refed, estimated_phases_refed, label="Data")
    axs.set_xlabel("Label Phase")
    axs.set_ylabel("Estimated Phase")
    line = np.arange(start=-np.pi, stop=np.pi, step=0.01)
    axs.plot(line, line, color="black", linestyle="--", label="y=x")
    axs.legend()
    fig.savefig(os.path.join(out_dir, "true_phases.png"))


