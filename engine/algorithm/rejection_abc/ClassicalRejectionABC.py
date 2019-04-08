import pyro.distributions as dist
import torch

from .. import Algorithm


class ClassicalRejectionABC(Algorithm):
    def __init__(self, prior, simulator, obs, params={}):
        super().__init__(prior, simulator, obs, params)

    def run(self, num_samples, threshold=0.5):
        inputs = self.prior(num_samples)
        outputs = self.simulator(inputs)  # ensure torch.Tensor

        discrepancies = torch.norm(outputs - self.obs, dim=1)
        samples = outputs[discrepancies < threshold]

        emp_dist = dist.Empirical(samples, torch.zeros((samples.shape[0],)))

        return emp_dist
