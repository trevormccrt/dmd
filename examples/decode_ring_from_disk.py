import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import s1_direct_product_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load some manifold data
data_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/2023-01-09-17-04-19")
data = np.load(os.path.join(data_dir, "encoded_points.npy"))
true_phases = np.load(os.path.join(data_dir, "true_phases.npy"))

encoder, decoder = s1_direct_product_decoder.train(data=data, manifold_dim=1, device=device,
                                                   n_training_iterations=400, decoder_weight=10, order_red_weight=0.1)

with torch.no_grad():
    decoded_points, decoded_angles = decoder(torch.tensor(data, dtype=torch.get_default_dtype()).to(device))

predicted_phases = torch.squeeze(decoded_angles).cpu().numpy()


def reference_phases(phases):
    phases_refd = phases - phases[0]
    phases_refd = np.arctan2(np.sin(phases_refd), np.cos(phases_refd))
    return phases_refd * np.sign(phases_refd[1])

true_phases_refed = reference_phases(true_phases)
predicted_phases_refd = reference_phases(predicted_phases)
fig, axs = plt.subplots()
axs.scatter(true_phases_refed, predicted_phases_refd, label="Data")
axs.set_xlabel("Label Phase")
axs.set_ylabel("Estimated Phase")
line = np.arange(start=-np.pi, stop=np.pi, step=0.01)
axs.plot(line, line, color="black", linestyle="--", label="y=x")
axs.legend()
plt.show()
