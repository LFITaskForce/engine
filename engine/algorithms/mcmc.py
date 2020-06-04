import numpyro

from functools import partial
from numpyro.infer.mcmc import MCMC, HMC, NUTS


def mcmc(model, obs, num_samples, kernel='HMC', kernel_params={},
             mcmc_params={}, sites=['theta']):
    # NOTE: requires differentiable model

    model_conditioned = partial(model, obs=obs)

    if kernel.upper() == 'HMC':
        mcmc_kernel = HMC(model_conditioned, **kernel_params)
    elif kernel.upper() == 'NUTS':
        mcmc_kernel = NUTS(model_conditioned, **kernel_params)
    else:
        raise NotImplementedError

    mcmc = MCMC(mcmc_kernel, num_samples, **mcmc_params)
    mcmc_run = mcmc.run()

    posterior = numpyro.infer.EmpiricalMarginal(mcmc_run, sites=sites)

    return posterior
