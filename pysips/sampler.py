import numpy as np

from .metropolis import Metropolis
from .prior import Prior
from smcpy import VectorMCMCKernel, AdaptiveSampler


def sample(likelihood, proposal, generator, multiprocess=False, kwargs=None, seed=None):

    rng = np.random.default_rng(seed)

    smc_kwargs = {"num_particles": 5000, "num_mcmc_samples": 10}
    if kwargs is not None:
        smc_kwargs.update(kwargs)
    return run_smc(likelihood, proposal, generator, multiprocess, smc_kwargs, rng)


def run_smc(likelihood, proposal, generator, multiprocess, kwargs, rng):

    prior = Prior(generator)

    mcmc = Metropolis(
        likelihood=likelihood,
        proposal=proposal,
        prior=prior,
        multiprocess=multiprocess,
    )
    kernel = VectorMCMCKernel(mcmc, param_order=["f"], rng=rng)
    smc = AdaptiveSampler(kernel)
    smc._mutator._compute_cov = False  # hack to bypass covariance calc on obj
    steps, _ = smc.sample(**kwargs)

    models = steps[-1].params[:, 0].tolist()
    likelihoods = [likelihood(c) for c in models]  # fit final pop of equ

    return models, likelihoods
