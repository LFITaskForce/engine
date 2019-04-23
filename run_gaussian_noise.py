import argparse
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch

from simulators.gaussian_noise import GaussianNoise
from engine.algorithms.abc import rejection_abc

pyro.set_rng_seed(101)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_simulations', type=int, default=10000,
        help='Number of simulations')
    parser.add_argument('--threshold', type=float, default=0.1,
        help='Rejection threshold')
    args = parser.parse_args()


    simulator_instance = GaussianNoise()

    def model(num_samples=10):
        dist_prior = dist.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.]))
        inputs = pyro.sample('input', dist_prior.expand_by([num_samples]))

        outputs = simulator_instance(inputs)

        return inputs, outputs

    obs = simulator_instance(torch.tensor([[1.]]))


    posterior = rejection_abc(model, obs,
        num_simulations=args.num_simulations, threshold=args.threshold)


    plt.figure()
    plt.title('posterior samples')
    plt.hist([posterior.sample().item() for _ in range(100)])
    plt.show()
