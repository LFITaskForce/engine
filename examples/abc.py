import argparse
import engine
dist = engine.distributions
import matplotlib.pyplot as plt

import torch

from sims.gaussian_noise import GaussianNoise
from engine.algorithms.abc import RejectionABC

engine.set_rng_seed(101)


def main(args):
    obs = torch.tensor([[1.]])

    @engine.simulator(name='gn', simulator_fn=GaussianNoise())
    def model():
        inputs = engine.sample('input', dist.Normal(loc=torch.tensor([0.]),
                                                    scale=torch.tensor([1.])))
        outputs = engine.simulate('gn', inputs, obs=obs)
        return inputs, outputs

    abc = RejectionABC(
        model=model,
        threshold=args.threshold,
        num_samples=args.num_simulations)

    posterior = abc.run()

    plt.figure()
    plt.hist( posterior.marginal('input').empirical['input']._get_samples_and_weights()[0].numpy().flatten())
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_simulations', type=int, default=10000,
        help='Number of simulations')
    parser.add_argument('--threshold', type=float, default=0.1,
        help='Rejection threshold')
    args = parser.parse_args()

    main(args)
