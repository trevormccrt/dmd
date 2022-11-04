import numpy as np
import pytest
from scipy import integrate
import torch

# this is the special sauce
from torchdiffeq import odeint


from torch_odes import torch_hopfield


def test_symmetric_weights():
    n = 10
    weights = np.random.uniform(-1, 1, (n, n))
    weights_sym_mat = weights + np.transpose(weights)
    weights_sym = weights_sym_mat[np.triu_indices(n)]
    non_autapse_weight_mat = np.copy(weights_sym_mat)
    non_autapse_weights = non_autapse_weight_mat[np.triu_indices(n, 1)]
    np.fill_diagonal(non_autapse_weight_mat, 0)

    net_aut = torch_hopfield.SymmetricHopfield(n, torch.from_numpy(weights_sym), autapses=True)
    net_no_aut = torch_hopfield.SymmetricHopfield(n, torch.from_numpy(non_autapse_weights), autapses=False)
    aut_weights = net_aut.get_weight_matrix().detach().numpy()
    non_aut_weights = net_no_aut.get_weight_matrix().detach().numpy()
    np.testing.assert_allclose(aut_weights, weights_sym_mat)
    np.testing.assert_allclose(non_aut_weights, non_autapse_weight_mat)

    with pytest.raises(AttributeError):
        _ = torch_hopfield.SymmetricHopfield(n, torch.from_numpy(weights_sym), autapses=False)
    with pytest.raises(AttributeError):
        _ = torch_hopfield.SymmetricHopfield(n, torch.from_numpy(non_autapse_weights), autapses=True)


def test_vs_scipy():

    def sigmoid(x):
        return np.exp(x) / (np.exp(x) + 1)

    def scipy_dynmaics(t, y, weights):
        return -y + weights.dot(sigmoid(y))

    batch_size = 15
    n = 10
    final_t = 5.0
    weights = np.random.uniform(-1, 1, (n, n))
    weights_sym = weights + np.transpose(weights)
    weight_vals = weights_sym[np.triu_indices(n)]
    ode_func = torch_hopfield.SymmetricHopfield(n, torch.from_numpy(weight_vals), autapses=True)
    init_states = np.random.uniform(-1, 1, (batch_size, n))
    torch_soln = odeint(ode_func, torch.from_numpy(init_states), torch.tensor([0, final_t]))[1, :, :]
    scipy_solns = []
    for init_sate in init_states:
        scipy_solns.append(integrate.solve_ivp(lambda t,y : scipy_dynmaics(t, y, weights_sym), [0, final_t], init_sate).y[:, -1])
    scipy_solns = np.array(scipy_solns)
    torch_solns = torch_soln.detach().numpy()
    np.testing.assert_allclose(scipy_solns, torch_solns, atol=1e-3)


def test_origin_destabilization():
    # the origin should be stable until the absolute value of the hopfield net weight is larger than 1
    # an optimizer should be able to figure this out and destabilize the system
    system = torch_hopfield.SymmetricHopfield(2, torch.from_numpy(np.array([0.5])), autapses=False, nonlin=torch.tanh)
    eq_time = 5.0
    batch_dim = 20
    n_iter = 20
    opt = torch.optim.Adam(system.parameters(), lr=0.05)
    for i in range(n_iter):
        opt.zero_grad()
        init_states = np.random.uniform(-0.1, 0.1, (batch_dim, 2))
        final_states = odeint(system, torch.from_numpy(init_states), torch.tensor([0, eq_time]))[1, :, :]
        # if you get further from the origin the loss goes down
        loss = -torch.sum(torch.square(final_states))
        loss.backward()
        opt.step()
    assert loss < -0.5

