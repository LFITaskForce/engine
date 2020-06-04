import argparse
import engine
dist = engine.distributions
import matplotlib.pyplot as plt

import numpy as np
# import torch
from torch.utils.data import DataLoader, Dataset

from sims.gaussian_noise import GaussianNoise
from engine.algorithms.abc import RejectionABC

engine.set_rng_seed(101)

def vectorize(fn, num_samples):
    def _fn(*args, **kwargs):
        with engine.plate("num_particles_vectorized", num_samples, dim=-2):
            return fn(*args, **kwargs)
    return _fn

class SimsDataset(Dataset):
    def __init__(self, proposal, sims, transform=None):
        self.proposal = proposal
        self.sims = sims
    def __len__(self):
        return len(self.sims)

    def __getitem__(self, idx):
        sample = {'input': self.proposal[idx], 'gn': self.sims[idx]}
        return sample


def main(args):
    # First use the simulator to draw an offline dataset set
    @engine.simulator(name='gn', simulator_fn=GaussianNoise())
    def online_model():
        inputs = engine.sample('input', dist.Normal(loc=np.array([0.]),
                                                    scale=np.array([1.])))
        outputs = engine.simulate('gn', inputs)
        return inputs, outputs
    proposal, sims = vectorize(online_model, 10000)()
    offline_dset = DataLoader(SimsDataset(proposal, sims))


    obs = np.array([[1.]])
    @engine.simulator(name='gn', dataset=offline_dset)
    def model():
        inputs = engine.sample('input', dist.Normal(loc=np.array([0.]),
                                                    scale=np.array([1.])))
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
