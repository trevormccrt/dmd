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
data_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/noised_5")
data = np.load(os.path.join(data_dir, "encoded_points.npy"))
true_phases = None # in general we won't have labels
# if we have label data we can plot actual vs predicted phase
true_phases = np.load(os.path.join(data_dir, "true_phases.npy"))


in_dim = np.shape(data)[1] # we will give the NN points on a ring in 2D as input
compressed_dim = 2 # whatever dimension we want to encode into
encoder_hidden_dim = 1500
encoder_n_hidden = 1
decoder_hidden_dim = encoder_hidden_dim
decoder_n_hidden = encoder_n_hidden
n_resample = 50 # how many points to use in distance integration calculation

encoder_net = encoder_decoder_core.AllPeriodicEncoder(in_dim, 1, encoder_hidden_dim, encoder_n_hidden).to(device)
decoder_net = encoder_decoder_core.AllPeriodicDecoder(in_dim, 1, decoder_hidden_dim, decoder_n_hidden).to(device)

params = list(encoder_net.parameters()) + list(decoder_net.parameters())
opt = torch.optim.Adam(params)

batch_size = 50
n_points_compare = 20 # how many points on the ring to compare distances between
n_points_length = 1000

n_epochs = 10000

encoder_losses = []
decoder_losses = []

sample_range = np.arange(start=0, stop=np.shape(data)[0], step=1)
best_loss = np.inf
order_reduction_thresh = 1
do_order_reduction = False
total_model_length_start = torch.tensor([0], dtype=torch.get_default_dtype()).to(device)
total_model_length_end = torch.tensor([2 * np.pi], dtype=torch.get_default_dtype()).to(device)
total_model_length_dir = torch.tensor([1], dtype=torch.get_default_dtype()).to(device)
for i in range(n_epochs):
    samples = np.array([data[np.random.choice(sample_range, size=n_points_compare, replace=False), :] for i in range(batch_size)])
    data_samples = torch.tensor(samples, dtype=torch.get_default_dtype()).to(device)
    opt.zero_grad()
    decoded_points, decoded_angles = decoder_net(data_samples)
    re_encoded_points = encoder_net(decoded_points)
    decoder_loss = 50 * torch.sum(torch.square(re_encoded_points - data_samples))/(batch_size * n_points_compare)
    rolled_decoded_angles = torch.roll(decoded_angles, 1, dims=-2)
    distance_calculation_directions = torch.randint(0, 2, decoded_angles.shape).to(device)
    angular_distances, model_distances = encoder_net.model_length(decoded_angles, rolled_decoded_angles, direction=distance_calculation_directions, n_points_integrate=n_resample)

    normed_angular_distance = angular_distances/torch.mean(angular_distances)
    normed_model_distance = model_distances/torch.mean(model_distances)
    distance_cost = 50 * torch.sum(torch.square(normed_angular_distance - normed_model_distance))/(batch_size * n_points_compare)

    ordered_decoded, _ = torch.sort(decoded_angles, dim=-2)
    _, nearest_model_distances = encoder_net.model_length(total_model_length_start, total_model_length_end,
                                                          n_points_integrate=n_points_length,
                                                          direction=total_model_length_dir)
    ordered_points = generative_isometry_util.torch_angles_to_ring(ordered_decoded)
    decoded_in_order = encoder_net(ordered_points)
    total_euclid_distance = torch.sum(
        torch.sqrt(torch.sum(torch.square(decoded_in_order - torch.roll(decoded_in_order, 1, dims=-2)), dim=-1) + 1e-13),
        dim=-1)
    model_distance = torch.sum(nearest_model_distances)
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
        print("iteration: {}, decoding loss: {}, distance cost: {}, order reduction: {}".format(i, decoder_loss, distance_cost, capacity_factor))
    loss.backward()
    opt.step()


with torch.no_grad():
    decoded_points, decoded_angles = best_decoder(torch.tensor(data, dtype=torch.get_default_dtype()).to(device))
    sample_direction = torch.randint(0, 2, decoded_angles.shape).to(device)
    angular_distance, model_distance = best_encoder.model_length(decoded_angles, torch.roll(decoded_angles, 1, dims=-2), direction=sample_direction, n_points_integrate=n_resample)
    order = torch.squeeze(torch.argsort(decoded_angles, dim=-2))
    normed_angular_distance = angular_distance / torch.mean(angular_distance)
    normed_model_distance = model_distance / torch.mean(model_distance)
    rel_error = (normed_angular_distance - normed_model_distance)/normed_model_distance

order = order.cpu().numpy()
decoded_points = decoded_points.cpu().numpy()
phases = torch.squeeze(decoded_angles).cpu().numpy()
rel_error = np.squeeze(rel_error.cpu().numpy())
angles = np.arange(start=0, stop=2 * np.pi, step=0.01)
fig, axs = plt.subplots(ncols=2)

axs[0].scatter(decoded_points[:, 0, 0], decoded_points[:, 0, 1], label="Mapped Data")
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

ordered_points = data[order, :]
np.save(os.path.join(out_dir, "ordered_points.npy"), ordered_points)

torch.save(best_encoder.state_dict(), os.path.join(out_dir, "best_encoder_state.pt"))
torch.save(best_decoder.state_dict(), os.path.join(out_dir, "best_decoder_state.pt"))

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
