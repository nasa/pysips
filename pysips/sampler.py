"""
Sequential Monte Carlo (SMC) Sampling with Custom Prior and MCMC Kernel.

This module provides high-level functions for performing Sequential Monte Carlo
sampling using custom prior distributions and Metropolis-Hastings MCMC kernels.
It integrates with the smcpy library to provide adaptive sampling capabilities
with unique value generation and optional checkpointing support.

The module is designed for scenarios where you need to sample from a parameter
space using a custom generator function while ensuring uniqueness of samples
and applying likelihood-based filtering. Checkpointing allows for resuming
interrupted sampling runs and provides fault tolerance for long-running
computations.

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
>>> # Basic usage without checkpointing
>>> models, likelihoods = sample(my_likelihood, my_proposal, my_generator)
>>> print(f"Found {len(models)} models with likelihoods")
>>>
>>> # Usage with checkpointing
>>> models, likelihoods = sample(my_likelihood, my_proposal, my_generator,
...                              checkpoint_file="my_sampling.pkl")
>>> print(f"Checkpointed run completed with {len(models)} models")

Notes
-----
This module uses the following workflow:
1. Creates a custom Prior that generates unique models
2. Sets up a Metropolis-Hastings MCMC kernel
3. Optionally enables checkpointing for progress persistence
4. Runs adaptive SMC sampling
5. Returns the final population of models and their likelihoods

The covariance calculation is disabled in the mutator as a workaround for
object-based parameters that may not support standard covariance computation.

Checkpointing automatically saves sampling progress and can resume from the
last saved state if the checkpoint file exists when sampling begins.
"""

# pylint: disable=R0913,R0917
import numpy as np
from smcpy import VectorMCMCKernel, AdaptiveSampler, FixedTimeSampler, MaxStepSampler
from smcpy.utils.storage import PickleStorage

from .metropolis import Metropolis
from .prior import Prior


def sample(
    likelihood,
    proposal,
    generator,
    max_time=None,
    max_equation_evals=None,
    multiprocess=False,
    kwargs=None,
    seed=None,
    checkpoint_file=None,
    xvals=None
):
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
    max_time : float, optional
        Maximum compute time limit for the sampling, in seconds (default no time limit).
    max_equation_evals : int, optional
        Maximum number of equation evaluations allowed during sampling (default: no limit).
    multiprocess : bool, optional
        Whether to use multiprocessing for likelihood evaluations (default: False).
    kwargs : dict, optional
        Additional keyword arguments to override default SMC parameters.
        Default parameters are {"num_particles": 5000, "num_mcmc_samples": 10}.
    seed : int, optional
        Random seed for reproducible results (default: None).
    checkpoint_file : str, optional
        File path for saving and loading sampling progress. If the checkpoint
        file exists, sampling will attempt to resume from the saved state and
        continue updating the checkpoint as it proceeds. If None, no
        checkpointing is performed (default: None).

    Returns
    -------
    models : list
        List of parameter values from the final SMC population.
    likelihoods : list
        List of likelihood values corresponding to each model in the final population.
    phis : list
        List of phi values (tempering parameters) from the SMC sequence.

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
    >>> # Basic sampling without checkpointing
    >>> models, likes, phis = sample(likelihood_func, proposal_func, generator_func)
    >>> print(f"Sampled {len(models)} models")
    >>>
    >>> # Sampling with checkpointing
    >>> models, likes, phis = sample(likelihood_func, proposal_func, generator_func,
    ...                              checkpoint_file="progress.pkl")
    >>> print(f"Checkpointed sampling completed")

    Notes
    -----
    This function internally calls run_smc with default parameters. The default
    configuration uses 5000 particles and 10 MCMC samples per SMC step, which
    provides a reasonable balance between accuracy and computational cost for
    many applications.

    When checkpointing is enabled:
    - If the checkpoint file exists, sampling resumes from the saved state
    - Progress is automatically saved during the sampling process
    - The checkpoint file uses pickle format for serialization
    - Interrupted runs can be restarted from the last checkpoint

    When max_equation_evals and max_time are specified:
    - max_time takes precedence over max_equation_evals
    """
    rng = np.random.default_rng(seed)

    smc_kwargs = {"num_particles": 5000, "num_mcmc_samples": 10}
    if kwargs is not None:
        smc_kwargs.update(kwargs)
    return run_smc(
        likelihood,
        proposal,
        generator,
        max_time,
        max_equation_evals,
        multiprocess,
        smc_kwargs,
        rng,
        checkpoint_file,
        xvals
    )


def run_smc(
    likelihood,
    proposal,
    generator,
    max_time,
    max_equation_evals,
    multiprocess,
    kwargs,
    rng,
    checkpoint_file,
    xvals
):
    """
    Execute Sequential Monte Carlo sampling with full parameter control.

    This function implements the core SMC sampling algorithm using a custom
    prior distribution and Metropolis-Hastings MCMC kernel. It provides
    complete control over all sampling parameters and optional checkpointing.

    Parameters
    ----------
    likelihood : callable
        Function that computes the likelihood of a given parameter value.
    proposal : callable
        Function that proposes new parameter values in MCMC steps.
    generator : callable
        Function that generates unique initial parameter values.
    max_time : float, None
        Maximum compute time limit for the sampling, in seconds. None value indicates
        no time limit.
    max_equation_evals : int, None
        Maximum number of equation evaluations allowed during sampling.
        None value indicates no evaluation limit.
    multiprocess : bool
        Whether to enable multiprocessing for likelihood evaluations.
    kwargs : dict
        Keyword arguments for the SMC sampler (e.g., num_particles, num_mcmc_samples).
    rng : numpy.random.Generator
        Random number generator instance for reproducible sampling.
    checkpoint_file : str, None
        File path for checkpointing. If None, no checkpointing is performed.
        If provided, sampling progress will be saved to this file and can be
        resumed if the file exists from a previous run.

    Returns
    -------
    models : list
        Parameter values from the final SMC population, converted to list format.
    likelihoods : list
        Likelihood values for each model in the final population, computed
        fresh to ensure consistency.
    phis : list
        Phi values (tempering parameters) from the SMC sequence.

    Notes
    -----
    The checkpointing mechanism uses SMCPy's PickleStorage context manager:
    - Automatically detects existing checkpoint files and resumes
    - Saves progress incrementally during sampling
    - Uses append mode ('a') by default for safe restarts
    - Handles serialization of the complete sampler state
    """
    kernel = _create_mcmc_kernel(likelihood, proposal, generator, multiprocess, rng, xvals)

    # Execute sampling with or without checkpointing
    if checkpoint_file is None:
        steps, phis = _smc_call(max_time, kwargs, kernel)
    else:
        with PickleStorage(checkpoint_file):
            steps, phis = _smc_call(max_time, max_equation_evals, kwargs, kernel)

    models = steps[-1].params[:, 0].tolist()
    likelihoods = [likelihood(c) for c in models]  # fit final pop of equ

    return models, likelihoods, phis


def _create_mcmc_kernel(likelihood, proposal, generator, multiprocess, rng, xvals):
    prior = Prior(generator)
    mcmc = Metropolis(
        likelihood=likelihood,
        proposal=proposal,
        prior=prior,
        multiprocess=multiprocess,
        xvals=xvals
    )
    return VectorMCMCKernel(mcmc, param_order=["f"], rng=rng)


def _smc_call(max_time, max_equation_evals, kwargs, kernel):
    if max_time is not None:
        smc = FixedTimeSampler(kernel, max_time)
    elif max_equation_evals is not None:
        max_steps = max_equation_evals // (
            kwargs["num_particles"] * kwargs["num_mcmc_samples"]
        )
        smc = MaxStepSampler(kernel, max_steps=max_steps)
    else:
        smc = AdaptiveSampler(kernel)

    # pylint: disable=W0212
    smc._mutator._compute_cov = False  # hack to bypass covariance calc on obj
    steps, _ = smc.sample(**kwargs)
    phis = smc.phi_sequence
    return steps, phis
