import torch


class GaussianNoise:
    """Homoscedastic Gaussian noise around mean function in N-d."""
    def __init__(self, dim=1, fun=lambda theta: theta, scaling=0.1):
        self.fun = fun
        self.cov = scaling*torch.eye(dim)

    def __call__(self, inputs):
        return self.forward(inputs)

    def _dist(self, theta):
        mu = self.fun(theta)
        return torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=self.cov)

    def forward(self, theta):
        return self._dist(theta).sample()
