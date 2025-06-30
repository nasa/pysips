"""
Sequential Monte Carlo (SMC) Sampling with Custom Prior and MCMC Kernel.

This module provides high-level functions for performing Sequential Monte Carlo
sampling using custom prior distributions and Metropolis-Hastings MCMC kernels.
It integrates with the smcpy library to provide adaptive sampling capabilities
with unique value generation.

The module is designed for scenarios where you need to sample from a parameter
space using a custom generator function while ensuring uniqueness of samples
and applying likelihood-based filtering.

Example
-------
>>> def my_likelihood(x):
...     return np.exp(-0.5 * x**2)  # Gaussian-like likelihood
>>>
>>> def my_proposal(x):
...     return x + np.random.normal(0, 0.1)  # Random walk proposal
>>>
>>> def my_generator():
...     return np.random.uniform(-5, 5)  # Uniform parameter generator
>>>
>>> models, likelihoods = sample(my_likelihood, my_proposal, my_generator)
>>> print(f"Found {len(models)} models with likelihoods")

Notes
-----
This module uses the following workflow:
1. Creates a custom Prior that generates unique values
2. Sets up a Metropolis-Hastings MCMC kernel
3. Runs adaptive SMC sampling
4. Returns the final population of models and their likelihoods

The covariance calculation is disabled in the mutator as a workaround for
object-based parameters that may not support standard covariance computation.
"""

# pylint: disable=R0913,R0917
import numpy as np
from smcpy import VectorMCMCKernel, AdaptiveSampler

from .metropolis import Metropolis
from .prior import Prior


def sample(likelihood, proposal, generator, multiprocess=False, kwargs=None, seed=None):
    """
    Perform Sequential Monte Carlo sampling with default parameters.

    This is a high-level convenience function that sets up and runs SMC sampling
    with commonly used default parameters. For more control over the sampling
    process, use run_smc directly.

    Parameters
    ----------
    likelihood : callable
        Function that computes the likelihood of a given parameter value.
        Should accept a single parameter and return a scalar likelihood value.
    proposal : callable
        Function that proposes new parameter values given a current value.
        Used in the Metropolis-Hastings MCMC steps.
    generator : callable
        Function that generates initial parameter values when called with no
        arguments. Should return hashable values for uniqueness tracking.
    multiprocess : bool, optional
        Whether to use multiprocessing for likelihood evaluations (default: False).
    kwargs : dict, optional
        Additional keyword arguments to override default SMC parameters.
        Default parameters are {"num_particles": 5000, "num_mcmc_samples": 10}.
    seed : int, optional
        Random seed for reproducible results (default: None).

    Returns
    -------
    models : list
        List of parameter values from the final SMC population.
    likelihoods : list
        List of likelihood values corresponding to each model in the final population.

    Examples
    --------
    >>> def likelihood_func(x):
    ...     return np.exp(-0.5 * (x - 2)**2)
    >>>
    >>> def proposal_func(x):
    ...     return x + np.random.normal(0, 0.5)
    >>>
    >>> def generator_func():
    ...     return np.random.uniform(-10, 10)
    >>>
    >>> models, likes = sample(likelihood_func, proposal_func, generator_func)
    >>> print(f"Sampled {len(models)} models")

    Notes
    -----
    This function internally calls run_smc with default parameters. The default
    configuration uses 5000 particles and 10 MCMC samples per SMC step, which
    provides a reasonable balance between accuracy and computational cost for
    many applications.
    """
    rng = np.random.default_rng(seed)

    smc_kwargs = {"num_particles": 5000, "num_mcmc_samples": 10}
    if kwargs is not None:
        smc_kwargs.update(kwargs)
    return run_smc(likelihood, proposal, generator, multiprocess, smc_kwargs, rng)


def run_smc(likelihood, proposal, generator, multiprocess, kwargs, rng):
    """
    Execute Sequential Monte Carlo sampling with full parameter control.

    This function implements the core SMC sampling algorithm using a custom
    prior distribution and Metropolis-Hastings MCMC kernel. It provides
    complete control over all sampling parameters.

    Parameters
    ----------
    likelihood : callable
        Function that computes the likelihood of a given parameter value.
    proposal : callable
        Function that proposes new parameter values in MCMC steps.
    generator : callable
        Function that generates unique initial parameter values.
    multiprocess : bool
        Whether to enable multiprocessing for likelihood evaluations.
    kwargs : dict
        Keyword arguments for the SMC sampler (e.g., num_particles, num_mcmc_samples).
    rng : numpy.random.Generator
        Random number generator instance for reproducible sampling.

    Returns
    -------
    models : list
        Parameter values from the final SMC population, converted to list format.
    likelihoods : list
        Likelihood values for each model in the final population, computed
        fresh to ensure consistency.
    """
    prior = Prior(generator)

    mcmc = Metropolis(
        likelihood=likelihood,
        proposal=proposal,
        prior=prior,
        multiprocess=multiprocess,
    )
    kernel = VectorMCMCKernel(mcmc, param_order=["f"], rng=rng)
    smc = AdaptiveSampler(kernel)

    # pylint: disable=W0212
    smc._mutator._compute_cov = False  # hack to bypass covariance calc on obj
    steps, _ = smc.sample(**kwargs)

    models = steps[-1].params[:, 0].tolist()
    likelihoods = [likelihood(c) for c in models]  # fit final pop of equ

    return models, likelihoods
