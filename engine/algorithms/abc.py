import pyro
import pyro.distributions as pyd
import torch


def rejection_abc(model, obs, num_simulations, threshold=0.5):
    inputs, outputs = model(num_simulations)

    discrepancies = torch.norm(outputs - obs, dim=1)
    samples = inputs[discrepancies <= threshold]

    posterior = pyd.Empirical(samples, torch.zeros((samples.shape[0],)))

    return posterior
