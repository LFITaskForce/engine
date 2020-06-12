import argparse
import engine
dist = engine.distributions
import matplotlib.pyplot as plt

import numpy as np
import numpyro
from numpyro.handlers import seed

from sims.gaussian_noise import GaussianNoise
from engine.algorithms.abc import RejectionABC

def main(args):
    obs = np.array([[1.]])

    # @engine.simulator(name='gn', simulator_fn=GaussianNoise())
    def model():
        inputs = numpyro.sample('input', dist.Normal(loc=np.array([0.]), scale=np.array([1.])))
        # outputs = numpyro.sample('gn', fn=None, obs=obs)
        return inputs

    model = engine.simulator('gn', simulator_fn=GaussianNoise)(model)
    model = seed(model, rng_seed=101)

    # import pdb; pdb.set_trace()

    abc = RejectionABC(
        model=model,
        threshold=args.threshold,
        num_samples=args.num_simulations)

    posterior = abc.run(rng_seed=1)

    plt.figure()
    plt.hist( posterior.marginal('input').empirical['input']._get_samples_and_weights()[0].flatten())
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_simulations', type=int, default=100,
        help='Number of simulations')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
        help='Rejection threshold')
    args = parser.parse_args()

    main(args)
