import abc
import torch
from torch import nn


class GeneralHopfieldDynamics(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, nonlin=torch.sigmoid):
        super().__init__()
        self.nonlin = nonlin

    @abc.abstractmethod
    def get_weight_matrix(self):
        pass

    def forward(self, _, y):
        return -y + torch.einsum("ij, ...j -> ...i", self.get_weight_matrix(), self.nonlin(y))


class SymmetricHopfield(GeneralHopfieldDynamics):
    def __init__(self, n, weights, nonlin=torch.sigmoid, autapses=True):
        super().__init__(nonlin)
        target_n_weights = int(n * (n + 1)/2)
        if not autapses:
            target_n_weights -= n
        if not len(weights) == target_n_weights:
            raise AttributeError("Must supply correct number of weights to a Symmetric Hopfield network")
        self.n = n
        self.weights = torch.nn.Parameter(weights)
        self.autapses = autapses

    def get_weight_matrix(self):
        offset = 0
        if not self.autapses:
            offset = 1
        weight_matrix = torch.zeros((self.n, self.n), dtype=self.weights.dtype)
        upper_tri_ind = torch.triu_indices(self.n, self.n, offset=offset)
        weight_matrix[upper_tri_ind[0, :], upper_tri_ind[1, :]] = self.weights
        return weight_matrix + torch.tril(weight_matrix.transpose(1, 0), -1)
