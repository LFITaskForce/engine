import argparse
import engine
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch

from sims.gaussian_noise import GaussianNoise
from engine.algorithms.abc import rejection_abc

pyro.set_rng_seed(101)


def main(args):
    @engine.register_simulator(name='gn', simulator=GaussianNoise())
    def model(num_samples=1):
        inputs = pyro.sample('input',
            dist.Normal(loc=torch.tensor([0.]),
                        scale=torch.tensor([1.])).expand_by([num_samples]))
        outputs = engine.simulate('gn', inputs)
        return inputs, outputs

    obs = torch.tensor([[1.]])

    posterior = rejection_abc(
        model=model,
        obs=obs,
        num_simulations=args.num_simulations,
        threshold=args.threshold)

    plt.figure()
    plt.hist([posterior.sample() for _ in range(100)])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_simulations', type=int, default=10000,
        help='Number of simulations')
    parser.add_argument('--threshold', type=float, default=0.1,
        help='Rejection threshold')
    args = parser.parse_args()

    main(args)
