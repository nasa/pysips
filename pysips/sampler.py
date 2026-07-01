"""
Sequential Monte Carlo (SMC) sampling in a continuous latent space.

This module runs vanilla SMC over an N-dimensional continuous latent space
using smcpy's native machinery.  The prior and MCMC proposal live entirely in
latent space; all symbolic-regression specifics are hidden inside a custom
likelihood (see :mod:`pysips.likelihood`).

The likelihood decodes each latent point into a symbolic expression, scores it
with the Laplace NMLL, and stores the *expression objects themselves* in the
SMC ``log_like`` slot.  Because smcpy carries ``log_like`` through accept/reject
(``np.where``) and resampling in lockstep with the latent particles, each final
particle's decoded equation is exactly the one that gated its acceptance.  The
scalar log-likelihood used for tempering/acceptance is read from each
expression's cached ``fitness`` inside
:class:`~pysips.likelihood.EquationGeometricPath`.

The sampling strategy is selected from the constraints supplied:

* ``max_time`` given -> :class:`smcpy.FixedTimeSampler`
* ``max_equation_evals`` given (and no ``max_time``) -> :class:`smcpy.MaxStepSampler`
* neither -> :class:`smcpy.AdaptiveSampler`

Optional checkpointing persists progress to a pickle file and can resume an
interrupted run.
"""

# pylint: disable=R0913,R0917
import numpy as np
from smcpy import VectorMCMCKernel, AdaptiveSampler, FixedTimeSampler, MaxStepSampler
from smcpy.utils.storage import InMemoryStorage, PickleStorage

from .likelihood import ObjectVectorMCMC, EquationGeometricPath


class _SingleStepStorage(InMemoryStorage):
    """In-memory storage that only retains the most recent step.

    smcpy's default InMemoryStorage accumulates every intermediate SMC step,
    each containing num_particles expression objects.  For large populations
    this causes significant memory growth during sampling.  Since pysips only
    needs the final population, this subclass discards earlier steps on save.
    """

    def save_step(self, step):
        self._step_list = [step]
        self._phi_sequence.append(step.attrs["phi"])
        self._mut_ratio_sequence.append(step.attrs["mutation_ratio"])


def _infer_latent_dim(prior):
    """Infer the latent-space dimension from an smcpy-style prior.

    Mirrors smcpy's own convention: ``prior`` is a sequence of scipy-style
    distributions, each contributing ``dim`` dimensions (default 1).
    """
    return int(sum(getattr(p, "dim", 1) for p in prior))


def sample(
    likelihood,
    prior,
    max_time=None,
    max_equation_evals=None,
    kwargs=None,
    seed=None,
    checkpoint_file=None,
    show_progress_bar=True,
):
    """
    Perform latent-space SMC sampling with default parameters.

    High-level convenience wrapper around :func:`run_smc` with commonly used
    defaults (5000 particles, 10 MCMC samples per step).

    Parameters
    ----------
    likelihood : callable
        Latent-space likelihood (e.g.
        :class:`pysips.likelihood.LatentLikelihood`).  Called with an array of
        latent points and returns an object array of scored expressions.
    prior : sequence of scipy-style distributions
        Prior over the latent space, passed straight through to smcpy.  Its
        length (summed over ``dim``) defines the latent dimension N.
    max_time : float, optional
        Maximum compute time in seconds (default: no limit).
    max_equation_evals : int, optional
        Maximum number of equation evaluations (default: no limit).
    kwargs : dict, optional
        Overrides for the default SMC parameters
        ``{"num_particles": 5000, "num_mcmc_samples": 10}``.
    seed : int, optional
        Random seed for reproducible results (default: None).
    checkpoint_file : str, optional
        Pickle path for saving/resuming sampling progress (default: None).
    show_progress_bar : bool, optional
        Whether to display a progress bar during sampling (default: True).

    Returns
    -------
    models : list
        Decoded expression objects from the final SMC population.
    log_likes : list
        Cached log-likelihood (``fitness``) for each model.
    phis : list
        Phi values (tempering parameters) from the SMC sequence.

    Notes
    -----
    When both ``max_time`` and ``max_equation_evals`` are given, ``max_time``
    takes precedence.
    """
    rng = np.random.default_rng(seed)

    smc_kwargs = {"num_particles": 5000, "num_mcmc_samples": 10}
    if kwargs is not None:
        smc_kwargs.update(kwargs)
    return run_smc(
        likelihood,
        prior,
        max_time,
        max_equation_evals,
        smc_kwargs,
        rng,
        checkpoint_file,
        show_progress_bar,
    )


def run_smc(
    likelihood,
    prior,
    max_time,
    max_equation_evals,
    kwargs,
    rng,
    checkpoint_file,
    show_progress_bar,
):
    """
    Execute latent-space SMC sampling with full parameter control.

    Parameters
    ----------
    likelihood : callable
        Latent-space likelihood returning an object array of scored expressions.
    prior : sequence of scipy-style distributions
        Prior over the latent space (passed through to smcpy). Defines N.
    max_time : float or None
        Maximum compute time in seconds. None means no time limit.
    max_equation_evals : int or None
        Maximum number of equation evaluations. None means no limit.
    kwargs : dict
        SMC sampler keyword arguments (e.g. num_particles, num_mcmc_samples).
    rng : numpy.random.Generator
        Random number generator instance for reproducible sampling.
    checkpoint_file : str or None
        Pickle path for checkpointing. None disables checkpointing.
    show_progress_bar : bool
        Whether to display a progress bar during sampling.

    Returns
    -------
    models : list
        Decoded expression objects from the final SMC population.
    log_likes : list
        Cached log-likelihood (``fitness``) for each model.
    phis : list
        Phi values (tempering parameters) from the SMC sequence.

    Notes
    -----
    Sampling strategy selection logic:

    * ``max_time`` specified -> FixedTimeSampler
    * ``max_equation_evals`` specified (and ``max_time`` not) -> MaxStepSampler
    * neither specified -> AdaptiveSampler
    """
    kernel = _create_mcmc_kernel(likelihood, prior, rng)

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

    # The log_like slot carries the decoded expression objects themselves.
    models = final_step.log_likes.flatten().tolist()
    log_likes = [model.fitness for model in models]
    del final_step

    return models, log_likes, phis


def _create_mcmc_kernel(likelihood, prior, rng):
    n_dims = _infer_latent_dim(prior)
    # ObjectVectorMCMC allocates an object-dtype log_like buffer so decoded
    # expressions (not floats) ride through accept/reject and resampling.
    mcmc = ObjectVectorMCMC(
        model=None,
        data=None,
        priors=prior,
        log_like_func=lambda *args: likelihood,
    )
    param_order = [f"z{i}" for i in range(n_dims)]
    return VectorMCMCKernel(
        mcmc, param_order=param_order, path=EquationGeometricPath(), rng=rng
    )


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

    smc.sample(**kwargs)
    phis = smc.phi_sequence
    return smc.step, phis
