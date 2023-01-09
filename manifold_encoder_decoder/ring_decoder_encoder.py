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
data_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/2022-11-17-13-24-23")
data = np.load(os.path.join(data_dir, "encoded_points.npy"))
true_phases = None # in general we won't have labels



in_dim = np.shape(data)[1] # we will give the NN points on a ring in 2D as input
compressed_dim = 12 # whatever dimension we want to encode into
encoder_hidden_dim = 1000
encoder_n_hidden = 1
decoder_hidden_dim = encoder_hidden_dim
decoder_n_hidden = encoder_n_hidden
n_resample = 50 # how many points to use in distance integration calculation

# build the encoder NN
encoder_layers = []
encoder_layers.append(nn.Linear(in_dim, int(encoder_hidden_dim/2)))
encoder_layers.append(nn.ReLU())
for _ in range(encoder_n_hidden):
    encoder_layers.append(nn.LazyLinear(encoder_hidden_dim))
    encoder_layers.append(nn.ReLU())
encoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
encoder_layers.append(nn.ReLU())
encoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), compressed_dim))
encoder_net = nn.Sequential(*encoder_layers).to(device)

# build the decoder NN
decoder_layers = []
decoder_layers.append(nn.Linear(compressed_dim, int(encoder_hidden_dim/2)))
decoder_layers.append(nn.ReLU())
for _ in range(encoder_n_hidden):
    decoder_layers.append(nn.LazyLinear( encoder_hidden_dim))
    decoder_layers.append(nn.ReLU())
decoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
decoder_layers.append(nn.ReLU())
decoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), in_dim))
decoder_net = nn.Sequential(*decoder_layers).to(device)

# this can be used to fit a circle to some points
circularizer = generative_isometry_util.CircleDistance().to(device)

params = list(encoder_net.parameters()) + list(decoder_net.parameters()) + list(circularizer.parameters())
opt = torch.optim.Adam(params)

batch_size = 20
n_points_compare = 20 # how many points on the ring to compare distances between
n_epochs = 200

encoder_losses = []
decoder_losses = []

iso_loss = False
sample_range = np.arange(start=0, stop=np.shape(data)[0], step=1)
for _ in range(n_epochs):
    samples = np.array([data[np.random.choice(sample_range, size=n_points_compare, replace=False), :] for i in range(batch_size)])
    data_samples = torch.tensor(samples, dtype=torch.get_default_dtype()).to(device)
    opt.zero_grad()
    encoded = encoder_net(data_samples)
    circularized = circularizer.circularize(encoded)

    # disp_cost is a number related to how spread out the input samples are on the ring. Initially, we just want to make
    # sure that this is large so that the NN learns to use the entire ring (instead of just part of it)
    # if you don't get it try deleting it from the cost function and see what happens
    # this is basically helping us bootstrap our data onto something low dimensional that we can at least understand
    # (even if the mapping is not yet isometric)
    x_disp = torch.abs(torch.max(circularized[..., 0]) - torch.min(circularized[..., 0]))
    y_disp = torch.abs(torch.max(circularized[..., 0]) - torch.min(circularized[..., 0]))
    disp_cost = torch.square(x_disp - 2) + torch.square(y_disp - 2)

    # cost associated with how far the encoded data is from a circle
    ring_loss = torch.sum(circularizer(encoded))/(batch_size * n_points_compare)

    # cost associated with decoding fidelity
    decoded = decoder_net(encoded)
    decoder_loss = torch.sum(torch.square(decoded - data_samples))/(batch_size * n_points_compare)

    loss = ring_loss + decoder_loss

    distance_cost = None

    # we can't calculate distance if the inputs haven't been mapped to a ring yet, so wait for the loss to get low before doing this
    if loss < 0.001:
        iso_loss = True
    if iso_loss:
        encoded_angles, resampled_encoded_points = generative_isometry_util.resample_points_ring(encoded, n_resample)
        angular_distance = generative_isometry_util.torch_integrated_angle_metric(encoded_angles)
        normed_angular_distance = angular_distance/torch.mean(angular_distance)
        decoded_resample = decoder_net(resampled_encoded_points)
        decoder_distance = generative_isometry_util.integrated_point_metric(decoded_resample)
        normed_decoder_distance = decoder_distance/torch.mean(decoder_distance)
        distance_cost = torch.sum(torch.square(normed_angular_distance - normed_decoder_distance))/(batch_size * n_points_compare)
        loss += distance_cost

    # if we are not yet on a ring enforce the dispersion cost
    else:
        loss += disp_cost
    print("ring loss: {}, decoding loss: {}, dispersion_cost: {}, distance cost: {}".format(ring_loss, decoder_loss, disp_cost, distance_cost))
    loss.backward()
    opt.step()


with torch.no_grad():
    encoded = encoder_net(torch.from_numpy(data).to(device))
    encoded_angles, resampled_encoded_points = generative_isometry_util.resample_points_ring(torch.unsqueeze(encoded, 0), n_resample)
    encoded_distance = generative_isometry_util.torch_integrated_angle_metric(encoded_angles)
    circularized = circularizer.circularize(encoded)
    phases = circularizer.phases(encoded)
    decoded = decoder_net(resampled_encoded_points)
    decoded_distances = generative_isometry_util.integrated_point_metric(decoded)
    rat = decoded_distances/encoded_distance

circularized = circularized.cpu().numpy()
phases = phases.cpu().numpy()
rat = np.squeeze(rat.cpu().numpy())
angles = np.arange(start=0, stop=2 * np.pi, step=0.01)
fig, axs = plt.subplots(ncols=2)

axs[0].scatter(circularized[:, 0], circularized[:, 1], label="Mapped Data")
axs[0].plot(np.cos(angles), np.sin(angles), color="black", linestyle="--", label="r=1 circle")
axs[0].set_title("Data Mapped to 1D Ring")
axs[0].legend()
axs[1].hist((rat - np.mean(rat))/np.mean(rat))
axs[1].set_xlabel("Percent Distance Difference")
axs[1].set_ylabel("Counts")
axs[1].set_xlabel("Violation of Isometry")

out_dir = os.path.join(data_dir, "processing_results/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
fig.savefig(os.path.join(out_dir, "mapping.png"))

# if we have label data we can plot actual vs predicted phase
true_phases = np.load(os.path.join(data_dir, "true_phases.npy"))
if true_phases is not None:
    true_phases_refed = true_phases - true_phases[0]
    true_phases_refed = np.arctan2(np.sin(true_phases_refed), np.cos(true_phases_refed))
    true_phases_refed = true_phases_refed * np.sign(true_phases_refed[1])
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

plt.show()
