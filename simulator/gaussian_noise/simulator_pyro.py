import torch
import torch.distributions as td
import warnings

from benchmark.simulator_pyro import PyroSimulator


class GaussianNoise(PyroSimulator):
    def __init__(self, dim=1, fun=lambda theta: theta, scaling=0.1, seed=None):
        # Homoscedastic Gaussian noise around mean function in N-d.
        self.dim = dim
        self.fun = fun
        self.cov = scaling*torch.eye(self.dim)
        self.scaling = scaling
        if seed is not None:
            warnings.warn("Seeding not supported. Use a global session seed \
                           to control randomness")

    def dist(self, theta):
        # Inputs: theta as torch.Tensor.
        # Outputs: distribution object
        mu = self.fun(theta)
        return td.MultivariateNormal(loc=mu, covariance_matrix=self.cov)

    def forward(self, theta):
        # Inputs: theta as torch.Tensor.
        # Outputs: single sample as 1d torch.Tensor.
        return self.dist(theta).sample()

    def prob(self, theta, x):
        # Inputs: theta, x as 1d torch.Tensors.
        # Outputs: prob as 1d torch.Tensor.
        return torch.exp(self.log_prob(theta, x))

    def log_prob(self, theta, x):
        # Inputs: theta, x as 1d torch.Tensors.
        # Outputs: log_prob as 1d torch.Tensor.
        return self.dist(theta).log_prob(x)
