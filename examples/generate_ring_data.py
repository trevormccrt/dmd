import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import geometry_util
import generate_1d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dimension = 3

encoder, decoder = s1_direct_product_generator.train(1, embedding_dimension, device, n_training_iterations=3000)

angles = np.arange(start=0, stop=2 * np.pi, step=0.01)
with torch.no_grad():
    points = geometry_util.torch_angles_to_ring(torch.tensor(angles, dtype=torch.get_default_dtype()).to(device))
    points = torch.unsqueeze(points, -2)
    test_embedding = encoder(points)
test_embedding = test_embedding.cpu().numpy()

if embedding_dimension == 2:
    fig, axs = plt.subplots()
    axs.scatter(test_embedding[:, 0], test_embedding[:, 1], cmap="hsv", c=angles)
    plt.show()

elif embedding_dimension == 3:
    proj_fig = plt.figure()
    proj_axs = proj_fig.add_subplot(projection="3d")
    proj_axs.scatter(test_embedding[:, 0], test_embedding[:, 1], test_embedding[:, 2], cmap="hsv", c=angles)
    plt.show()


out_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
np.save(os.path.join(out_dir, "true_phases.npy"), angles)
np.save(os.path.join(out_dir, "encoded_points.npy"), test_embedding)
