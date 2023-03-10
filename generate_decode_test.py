import numpy as np
import torch

import generate_1d, decode_1d, geometry_util


def test_ring_generate_decode_1d():
    np.random.seed(123212)
    torch.manual_seed(564353)
    embed_dim = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_encoder, gen_decoder, gen_costs = generate_1d.train(1, 0, embed_dim, device, n_training_iterations=5000, verbose=False, encoder_hidden_dim=200, decoder_hidden_dim=200)
    assert gen_costs[0] < 0.05
    assert gen_costs[1] < 0.01
    test = np.linspace(start=-np.pi, stop=np.pi, num=1000)
    with torch.no_grad():
        embedded_points = gen_encoder(torch.tensor(np.expand_dims(test, -1), dtype=torch.get_default_dtype()).to(device))
    embedded_points = embedded_points/torch.mean(torch.abs(embedded_points))
    embedded_points_np = embedded_points.cpu().numpy()
    dec_encoder, dec_decoder, dec_costs = decode_1d.train(embedded_points_np, 1, 0, device, n_training_iterations=5000,verbose=False, decoder_weight=10, encoder_hidden_dim=300, decoder_hidden_dim=300, div_weight=0)
    with torch.no_grad():
        predicted_phases = dec_decoder(embedded_points)
    predicted_phases = np.squeeze(predicted_phases.cpu().numpy())
    refd_test = geometry_util.reference_periodic_phases(test)
    refd_pred = geometry_util.reference_periodic_phases(predicted_phases)
    with torch.no_grad():
        dists = geometry_util.minimum_periodic_distance(torch.tensor(np.expand_dims(refd_test, -1)), torch.tensor(np.expand_dims(refd_pred, -1)))[0]
    dists = dists.cpu().numpy()
    error = np.mean(dists)
    assert error < 0.1
    assert dec_costs[0] < 0.005
    assert dec_costs[1] < 0.005


def test_decode_circle():
    np.random.seed(2342389)
    torch.manual_seed(123144)
    circle_phases = np.linspace(start=0, stop=2 * np.pi, num=200)
    circle_points = geometry_util.angles_to_ring(circle_phases)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dec_encoder, dec_decoder, dec_costs = decode_1d.train(circle_points, 1, 0, device, n_training_iterations=2000,
                                                          verbose=False, integration_resamples=30,
                                                          encoder_hidden_dim=500, decoder_hidden_dim=500, div_weight=0)
    assert dec_costs[0] < 0.001
    assert dec_costs[1] < 0.001

    with torch.no_grad():
        predicted_phases = dec_decoder(torch.tensor(circle_points, dtype=torch.get_default_dtype()).to(device))
    predicted_phases = np.squeeze(predicted_phases.cpu().numpy())
    refd_test = geometry_util.reference_periodic_phases(circle_phases)
    refd_pred = geometry_util.reference_periodic_phases(predicted_phases)
    dec_error = np.mean(np.abs(refd_test - refd_pred))

    random_start_phases = np.random.uniform(0, 2 * np.pi, (20, 1))
    random_end_phases = np.random.uniform(0, 2 * np.pi, (20, 1))
    with torch.no_grad():
        ang_distances, arclength = dec_encoder.model_length(torch.tensor(random_start_phases, dtype=torch.get_default_dtype()).to(device),
                                             torch.tensor(random_end_phases, dtype=torch.get_default_dtype()).to(device))
    ang_distances = ang_distances.cpu().numpy()
    arclength = arclength.cpu().numpy()
    dist_error = np.mean(np.abs(ang_distances - arclength))
    assert dec_error < 0.03
    assert dist_error < 0.03


def test_decode_line():
    np.random.seed(2342375)
    torch.manual_seed(1233144)
    line_phases = np.linspace(start=-np.pi, stop=np.pi, num=200)
    line_points = np.zeros((len(line_phases), 2))
    line_points[:, 0] = line_phases
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dec_encoder, dec_decoder, dec_costs = decode_1d.train(line_points, 0, 1, device, n_training_iterations=2500,
                                                          verbose=False, integration_resamples=30, div_weight=0,
                                                          encoder_hidden_dim=100, decoder_hidden_dim=100)
    assert dec_costs[0] < 0.001
    assert dec_costs[1] < 0.001

    with torch.no_grad():
        predicted_phases = dec_decoder(torch.tensor(line_points, dtype=torch.get_default_dtype()).to(device))
    predicted_phases = np.squeeze(predicted_phases.cpu().numpy())
    scale_factor = 2 * np.pi/(np.max(predicted_phases) - np.min(predicted_phases))
    normed_predicted_phases = scale_factor * (predicted_phases - np.mean(predicted_phases))
    normed_predicted_phases = -1 * np.sign(normed_predicted_phases)[0] * normed_predicted_phases
    dec_error = np.mean(np.abs(normed_predicted_phases - line_phases))


    random_start_phases = np.random.uniform(np.min(predicted_phases), np.max(predicted_phases), (20, 1))
    random_end_phases = np.random.uniform(np.min(predicted_phases), np.max(predicted_phases), (20, 1))
    with torch.no_grad():
        ang_distances, arclength = dec_encoder.model_length(torch.tensor(random_start_phases, dtype=torch.get_default_dtype()).to(device),
                                             torch.tensor(random_end_phases, dtype=torch.get_default_dtype()).to(device))
    ang_distances = ang_distances.cpu().numpy()
    arclength = arclength.cpu().numpy()
    dist_error = np.mean(np.abs(ang_distances * scale_factor - arclength))
    assert dec_error < 0.05
    assert dist_error < 0.05
