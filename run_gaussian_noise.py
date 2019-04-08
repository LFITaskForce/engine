import argparse
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import utils.plot as plot

from simulator.gaussian_noise.simulator_pyro import GaussianNoise
from engine.algorithm import ClassicalRejectionABC

pyro.set_rng_seed(101)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100,
        help='Number of samples')
    parser.add_argument('--threshold', type=float, default=0.2,
        help='Rejection threshold')
    args = parser.parse_args()

    # Inputs: prior, simulator, observation
    def prior(num_samples=10):
        dist_prior = dist.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.]))
        inputs = pyro.sample('input', dist_prior.expand_by([num_samples]))
        return inputs

    sim = GaussianNoise()

    obs = sim(torch.tensor([[1.]]))

    # Algorithm
    algo = ClassicalRejectionABC(prior, sim, obs)
    res = algo.run(num_samples=args.num_samples,
                   threshold=args.threshold)

    # Output: Empirical distribution
    plot.hist_1d([res.sample() for _ in range(100)])
