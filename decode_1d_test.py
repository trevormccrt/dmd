import numpy as np
import torch

import decode_1d


def test_div_cost():
    uniform_samples = np.random.uniform(-np.pi, np.pi, 100000)
    div_uni = decode_1d.div_cost(torch.tensor(uniform_samples), torch.tensor(10))
    np.testing.assert_allclose(div_uni, 0, atol=1e-4)
    normal_samples = np.random.normal(0, 1, 10000)
    div_normal = decode_1d.div_cost(torch.tensor(normal_samples), torch.tensor(10))
    assert div_normal > 0
