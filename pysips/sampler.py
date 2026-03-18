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
from smcpy.utils.storage import InMemoryStorage, PickleStorage

from .metropolis import Metropolis


class _SingleStepStorage(InMemoryStorage):
    """In-memory storage that only retains the most recent step.

    smcpy's default InMemoryStorage accumulates every intermediate SMC
    step, each containing num_particles AGraph objects.  For large
    populations this causes significant memory growth during sampling.
    Since pysips only needs the final population, this subclass discards
    earlier steps on each save.
    """

    def save_step(self, step):
        self._step_list = [step]
        self._phi_sequence.append(step.attrs["phi"])
        self._mut_ratio_sequence.append(step.attrs["mutation_ratio"])


def sample(
    likelihood,
    proposal,
    prior,
    max_time=None,
    max_equation_evals=None,
    multiprocess=False,
    kwargs=None,
    seed=None,
    checkpoint_file=None,
    show_progress_bar=True,
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
    prior : Prior
        Custom prior distribution to use for sampling. Must implement the
        rvs() method for generating samples and logpdf() for computing log-probabilities.
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
    show_progress_bar : bool, optional
        Whether to display a progress bar during sampling. When False, the
        progress bar will be hidden, which is useful for hyperparameter
        tuning or when running multiple fits in parallel (default: True).

    Returns
    -------
    models : list
        List of parameter values from the final SMC population.
    log_likes : list
        List of log-likelihood values from the final SMC step for each model.
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
    >>> class MyCustomPrior(SamplablePrior):
    ...     def __init__(self, my_param, **kwargs):
    ...         x_dim = 1
    ...         operators = [2, 3, 4]  # Example operator codes
    ...         super().__init__(x_dim=x_dim, operators=operators, **kwargs)
    ...         self.my_param = my_param
    ...
    ...     def _logpdf_single(self, agraph):
    ...         # Custom log-probability calculation based on agraph and my_param
    ...         return -compute_some_metric(agraph, self.my_param)
    >>>
    >>> prior = MyCustomPrior(my_param=0.5, num_particles=100)
    >>> # Basic sampling without checkpointing
    >>> models, likes, phis = sample(likelihood_func, proposal_func, prior=prior)
    >>> print(f"Sampled {len(models)} models")
    >>>
    >>> # Sampling with checkpointing
    >>> models, likes, phis = sample(likelihood_func, proposal_func,
    ...                              prior=prior, checkpoint_file="progress.pkl")
    >>> print(f"Checkpointed sampling completed")
    >>>
    >>> # Sampling with max equation evaluations limit
    >>> models, likes, phis = sample(likelihood_func, proposal_func,
    ...                              prior=prior, max_equation_evals=10000)
    >>> print(f"Sampling completed with evaluation limit")

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
        prior,
        max_time,
        max_equation_evals,
        multiprocess,
        smc_kwargs,
        rng,
        checkpoint_file,
        show_progress_bar,
    )


def run_smc(
    likelihood,
    proposal,
    prior,
    max_time,
    max_equation_evals,
    multiprocess,
    kwargs,
    rng,
    checkpoint_file,
    show_progress_bar,
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
    prior : Prior, optional
        Custom prior distribution to use for sampling. If None, an
        ImproperUniformPrior with the provided generator will be used.
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
    show_progress_bar : bool
        Whether to display a progress bar during sampling. When False, progress
        updates are suppressed.

    Returns
    -------
    models : list
        Parameter values from the final SMC population, converted to list format.
    log_likes : list
        Log-likelihood values from the final SMC step for each model.
    phis : list
        Phi values (tempering parameters) from the SMC sequence.

    Notes
    -----
    The checkpointing mechanism uses SMCPy's PickleStorage context manager:
    - Automatically detects existing checkpoint files and resumes
    - Saves progress incrementally during sampling
    - Uses append mode ('a') by default for safe restarts
    - Handles serialization of the complete sampler state

    Sampling strategy selection logic:
    - If max_time is specified FixedTimeSampler is used
    - If max_equation_evals is specified (and max_time is not), MaxStepSampler is used
    - If neither is specified, AdaptiveSampler is used
    """
    kernel = _create_mcmc_kernel(likelihood, proposal, prior, multiprocess, rng)

    # Execute sampling with or without checkpointing.
    # _SingleStepStorage keeps only the latest step in memory, avoiding
    # unbounded growth from accumulating all intermediate populations.
    if checkpoint_file is None:
        with _SingleStepStorage():
            final_step, phis = _smc_call(
                max_time, max_equation_evals, kwargs, kernel, show_progress_bar
            )
    else:
        with PickleStorage(checkpoint_file):
            final_step, phis = _smc_call(
                max_time, max_equation_evals, kwargs, kernel, show_progress_bar
            )

    models = final_step.params[:, 0].tolist()
    log_likes = final_step.log_likes.ravel().tolist()
    del final_step

    return models, log_likes, phis


def _create_mcmc_kernel(likelihood, proposal, prior, multiprocess, rng):
    mcmc = Metropolis(
        likelihood=likelihood,
        proposal=proposal,
        prior=prior,
        multiprocess=multiprocess,
    )
    return VectorMCMCKernel(mcmc, param_order=["f"], rng=rng)


def _smc_call(max_time, max_equation_evals, kwargs, kernel, show_progress_bar):
    # Choose sampler based on specified constraints
    if max_time is not None:
        smc = FixedTimeSampler(kernel, max_time, show_progress_bar=show_progress_bar)
    elif max_equation_evals is not None:
        max_steps = max_equation_evals // (
            kwargs["num_particles"] * kwargs["num_mcmc_samples"]
        )
        smc = MaxStepSampler(
            kernel, max_steps=max_steps, show_progress_bar=show_progress_bar
        )
    else:
        smc = AdaptiveSampler(kernel, show_progress_bar=show_progress_bar)

    # pylint: disable=W0212
    smc._mutator._compute_cov = False  # hack to bypass covariance calc on obj
    smc.sample(**kwargs)
    phis = smc.phi_sequence
    return smc.step, phis
