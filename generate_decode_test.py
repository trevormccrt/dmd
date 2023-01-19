import numpy as np
import torch

import generate_1d, decode_1d, geometry_util


def test_ring_generate_decode_1d():
    embed_dim = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_encoder, gen_decoder, gen_costs = generate_1d.train(1, 0, embed_dim, device, n_training_iterations=3000, verbose=False)
    assert gen_costs[0] < 0.05
    assert gen_costs[1] < 0.01
    test = np.linspace(start=-np.pi, stop=np.pi, num=1000)
    with torch.no_grad():
        embedded_points = gen_encoder(torch.tensor(np.expand_dims(test, -1), dtype=torch.get_default_dtype()).to(device))
    embedded_points = embedded_points/torch.mean(torch.abs(embedded_points))
    embedded_points_np = embedded_points.cpu().numpy()
    dec_encoder, dec_decoder, dec_costs = decode_1d.train(embedded_points_np, 1, 0, device, n_training_iterations=3000,verbose=False)
    with torch.no_grad():
        predicted_phases = dec_decoder(embedded_points)
    predicted_phases = np.squeeze(predicted_phases.cpu().numpy())
    refd_test = geometry_util.reference_periodic_phases(test)
    refd_pred = geometry_util.reference_periodic_phases(predicted_phases)
    error = np.mean(np.abs(refd_test - refd_pred))
    assert error < 0.05
    assert dec_costs[0] < 0.005
    assert dec_costs[1] < 0.001
    print("")


test_ring_generate_decode_1d()
