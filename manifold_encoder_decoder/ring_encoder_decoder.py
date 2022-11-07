import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn

from manifold_encoder_decoder import generative_isometry_util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


in_dim = 2 # we will give the NN points on a ring in 2D as input
out_dim = 12 # whatever dimension we want to encode into
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
encoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), out_dim))
encoder_net = nn.Sequential(*encoder_layers).to(device)

# build the decoder NN
decoder_layers = []
decoder_layers.append(nn.Linear(out_dim, int(encoder_hidden_dim/2)))
decoder_layers.append(nn.ReLU())
for _ in range(encoder_n_hidden):
    decoder_layers.append(nn.LazyLinear( encoder_hidden_dim))
    decoder_layers.append(nn.ReLU())
decoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
decoder_layers.append(nn.ReLU())
decoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), in_dim))
decoder_net = nn.Sequential(*decoder_layers).to(device)

params = list(encoder_net.parameters()) + list(decoder_net.parameters())
opt = torch.optim.Adam(params)

batch_size = 20
n_points_compare = 4 # how many points on the ring to compare distances between
n_epochs = 100

encoder_losses = []
decoder_losses = []
for _ in range(n_epochs):
    # sample input phases
    phases_numpy = np.random.uniform(-np.pi, np.pi, (batch_size, n_points_compare))

    # sample in-between input phases for integration
    phases_span = generative_isometry_util.densely_sample_angles(phases_numpy, n_resample)

    # map phases to a ring in 2D (better for input to NN)
    mapped_init_angles = torch.tensor(generative_isometry_util.angles_to_ring(phases_numpy), dtype=torch.get_default_dtype()).to(device)
    mapped_span = torch.tensor(generative_isometry_util.angles_to_ring(phases_span), dtype=torch.get_default_dtype()).to(device)

    # input tensor and true distance on the ring
    phase_batch = torch.tensor(phases_span, dtype=torch.get_default_dtype()).to(device)
    phase_metric = torch.tensor(generative_isometry_util.integrated_angle_metric(phases_numpy), dtype=torch.get_default_dtype()).to(device)
    scaled_phase_metric = phase_metric/torch.mean(phase_metric)

    opt.zero_grad()

    #forward through the encoder
    encoded_points = encoder_net.forward(torch.reshape(mapped_span, [-1, mapped_span.size(-1)]))
    encoded_points_reshaped = torch.reshape(encoded_points, [*phase_batch.size(), -1])

    # calculate distance between encoded points
    encoded_distances = generative_isometry_util.integrated_point_metric(encoded_points_reshaped)
    scaled_encoded_distances = encoded_distances/torch.mean(encoded_distances)

    # send only the original points through the decoder (no need to send the extra points we sampled for integration)
    first_encoded_points = encoded_points_reshaped[:, :, 0, :]
    flat_first_encoded_points = torch.reshape(first_encoded_points, [-1, encoded_points.size(-1)])
    decoded_points = decoder_net.forward(flat_first_encoded_points)
    decoded_points_reshaped = torch.reshape(decoded_points, mapped_init_angles.size())

    # calculate loss as sum of error in distance and error in decoding
    loss_encoding = torch.sum(torch.square(scaled_encoded_distances - scaled_phase_metric))
    loss_decoding = torch.sum(torch.square(decoded_points_reshaped - mapped_init_angles))
    loss = loss_encoding + loss_decoding
    encoder_losses.append(loss_encoding)
    decoder_losses.append(loss_decoding)
    print("encoding loss: {}, decoding loss: {}".format(loss_encoding, loss_decoding))
    loss.backward()
    opt.step()

angles = np.arange(start=-np.pi, stop=np.pi, step=0.01)
with torch.no_grad():
    test_points = torch.tensor(generative_isometry_util.angles_to_ring(angles), dtype=torch.get_default_dtype()).to(device)
    forward_pred = encoder_net.forward(test_points)
    decoder_pred = decoder_net.forward(forward_pred)
forward_pred = forward_pred.cpu().detach().numpy()
decoder_pred = decoder_pred.cpu().detach().numpy()

encoder_losses = [x.cpu().detach().numpy() for x in encoder_losses]
decoder_losses = [x.cpu().detach().numpy() for x in decoder_losses]
if out_dim == 2:
    fig, axs = plt.subplots(nrows=1, ncols=4)
    axs[0].plot(encoder_losses)
    axs[1].plot(decoder_losses)
    axs[2].scatter(forward_pred[:, 0], forward_pred[:, 1],  cmap="hsv", c=angles)
    axs[3].scatter(decoder_pred[:, 0], decoder_pred[:, 1], cmap="hsv", c=angles)
    axs[3].plot(np.cos(angles), np.sin(angles), color="black", linestyle="--")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Encoder Loss")
    axs[0].set_yscale("log")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Decoder Loss")
    axs[1].set_yscale("log")
    axs[2].set_title("Encoded Representation")
    axs[3].set_title("Decoder Output")
    fig.tight_layout()

elif out_dim == 3:
    loss_fig, loss_axs = plt.subplots(ncols=2)
    loss_axs[0].plot(encoder_losses)
    loss_axs[1].plot(decoder_losses)
    loss_axs[0].set_xlabel("Epochs")
    loss_axs[0].set_ylabel("Encoder Loss")
    loss_axs[0].set_yscale("log")
    loss_axs[1].set_xlabel("Epochs")
    loss_axs[1].set_ylabel("Decoder Loss")
    loss_axs[1].set_yscale("log")
    proj_fig = plt.figure()
    proj_axs = proj_fig.add_subplot(projection="3d")
    proj_axs.scatter(forward_pred[:, 0], forward_pred[:, 1], forward_pred[:, 2], cmap="hsv", c=angles)
    proj_axs.set_title("Encoded Representation")
    decoder_fig, decoder_axs = plt.subplots()
    decoder_axs.scatter(decoder_pred[:, 0], decoder_pred[:, 1], cmap="hsv", c=angles)
    decoder_axs.plot(np.cos(angles), np.sin(angles), color="black", linestyle="--")
    decoder_axs.set_title("Decoder Output")

else:
    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].plot(encoder_losses)
    axs[1].plot(decoder_losses)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Encoder Loss")
    axs[0].set_yscale("log")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Decoder Loss")
    axs[1].set_yscale("log")
    axs[2].scatter(decoder_pred[:, 0], decoder_pred[:, 1], cmap="hsv", c=angles)
    axs[2].plot(np.cos(angles), np.sin(angles), color="black", linestyle="--")
    axs[2].set_title("Decoder Output")
plt.show()

n_samples = 1000
samples = np.random.uniform(-np.pi, np.pi, n_samples)
point_samples = generative_isometry_util.angles_to_ring(samples)
test_data = encoder_net.forward(torch.tensor(point_samples, dtype=torch.get_default_dtype()).to(device))
out_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
np.save(os.path.join(out_dir, "true_phases.npy"), samples)
np.save(os.path.join(out_dir, "encoded_points.npy"), test_data.cpu().detach().numpy())
