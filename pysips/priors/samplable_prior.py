"""
Samplable Prior Base Class for SMC-based Prior Sampling.

This module provides an abstract base class for prior distributions that
support sampling via Sequential Monte Carlo (SMC). Subclasses need only
implement `_logpdf_single()` and `_get_operators()` to get
full SMC sampling capability through the inherited `rvs()` method.

The design allows for easy creation of new prior types - each prior defines
its own probability model (logpdf), while the sampling infrastructure is
shared via the base class.

Example
-------
>>> class MyCustomPrior(SamplablePrior):
...     def __init__(self, x_dim, operators, my_param, **kwargs):
...         super().__init__(x_dim=x_dim, operators=operators, **kwargs)
...         self.my_param = my_param
...
...     def _logpdf_single(self, agraph):
...         # Custom probability calculation
...         return -compute_some_metric(agraph, self.my_param)
>>>
>>> prior = MyCustomPrior(x_dim=3, operators=[2, 3, 4], my_param=0.5, num_particles=100)
>>> samples = prior.rvs(50)  # Sample 50 expressions from prior
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from bingo.expressions.agraph import AGraphExpression

from .improper_uniform_prior import ImproperUniformPrior
from ..bingo_construction import (
    BingoConstructionConfig,
    build_agraph_generator,
    build_agraph_proposal,
)
from ..sampler import sample


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals
class SamplablePrior(ABC):
    """
    Abstract base class for priors that support SMC sampling.

    This class combines shared bingo generator/proposal construction with SMC
    sampling parameters to provide a complete `rvs()` implementation.
    Subclasses only need to implement:

    - `_logpdf_single(agraph)`: Compute log-probability for single expression

    Parameters
    ----------
    x_dim : int, optional
        Number of input variables (dimension of X data). Required for
        sampling via ``rvs()``.
    operators : list of int or str, optional
        List of operators for this prior. Can be list of bingo operator codes
        (ints) or operator names (strs). Required for sampling via ``rvs()``.
    num_mcmc_samples : int, optional
        Number of MCMC steps per SMC iteration. Default is 5.
    target_ess : float, optional
        Target effective sample size ratio. Default is 0.8.
    max_time : float, optional
        Maximum sampling time in seconds. Default is None.
    max_equation_evals : int, optional
        Maximum equation evaluations. Default is None.
    checkpoint_file : str, optional
        Path for checkpointing. Default is None.
    random_state : int, optional
        Random seed. Default is None.
    multiprocess : bool, optional
        If True, use multiprocessing. Default is False.
    **kwargs
        Additional bingo construction settings.

    Notes
    -----
    The `rvs()` method uses the prior's own `logpdf` as the SMC target
    distribution, allowing sampling from complex prior distributions
    that don't have closed-form sampling methods. The ``x_dim`` and
    ``operators`` parameters must be set to use ``rvs()``.
    """

    def __init__(
        self,
        x_dim: Optional[int] = None,
        operators: Optional[Union[List[int], List[str]]] = None,
        num_mcmc_samples: int = 5,
        target_ess: float = 0.8,
        max_time: Optional[float] = None,
        max_equation_evals: Optional[int] = None,
        checkpoint_file: Optional[str] = None,
        random_state: Optional[int] = None,
        multiprocess: bool = False,
        **kwargs,
    ):
        # Validate constraints
        if max_time is not None and max_equation_evals is not None:
            raise ValueError(
                "max_time and max_equation_evals cannot both be specified. "
                "Please choose one constraint method."
            )

        self.max_complexity = kwargs.pop("max_complexity", 24)
        self.terminal_probability = kwargs.pop("terminal_probability", 0.1)
        self.constant_probability = kwargs.pop("constant_probability", None)
        self.command_probability = kwargs.pop("command_probability", 0.2)
        self.node_probability = kwargs.pop("node_probability", 0.2)
        self.parameter_probability = kwargs.pop("parameter_probability", 0.2)
        self.prune_probability = kwargs.pop("prune_probability", 0.2)
        self.fork_probability = kwargs.pop("fork_probability", 0.2)
        self.repeat_mutation_probability = kwargs.pop(
            "repeat_mutation_probability", 0.05
        )
        self.crossover_pool_size = kwargs.pop("crossover_pool_size", None)
        self.mutation_prob = kwargs.pop("mutation_prob", 0.75)
        self.crossover_prob = kwargs.pop("crossover_prob", 0.25)
        self.exclusive = kwargs.pop("exclusive", True)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected bingo construction settings: {unexpected}")

        self.x_dim = x_dim
        self.operators = operators
        self.num_mcmc_samples = num_mcmc_samples
        self.target_ess = target_ess
        self.max_time = max_time
        self.max_equation_evals = max_equation_evals
        self.checkpoint_file = checkpoint_file
        self.random_state = random_state
        self.multiprocess = multiprocess

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log-prior probability for an array of AGraph expressions.

        Parameters
        ----------
        x : array-like
            Array or list of AGraph objects.

        Returns
        -------
        ndarray
            Array of shape ``(N, 1)`` with log-probability values.
        """
        log_probs = np.array([self._logpdf_single(ag) for ag in x])
        return log_probs.reshape(-1, 1)

    @abstractmethod
    def _logpdf_single(self, agraph: AGraphExpression) -> float:
        """
        Compute log-probability for a single AGraph expression.

        Parameters
        ----------
        agraph : AGraphExpression
            The symbolic expression to evaluate.

        Returns
        -------
        float
            Log-probability of the expression under this prior.
        """

    def rvs(self, N: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample N expressions from the prior using SMC.

        Uses Sequential Monte Carlo sampling with this prior's logpdf
        as the target distribution to generate samples from the prior.

        Parameters
        ----------
        N : int
            Number of samples to generate.
        random_state : int, optional
            Random seed for this sampling run. If None, uses the
            instance's random_state. Default is None.

        Returns
        -------
        ndarray
            Array of shape ``(N, 1)`` containing sampled AGraph expressions.

        Raises
        ------
        ValueError
            If ``x_dim`` was not set during initialization.
        """
        if self.x_dim is None:
            raise ValueError(
                "Cannot sample without x_dim. Set x_dim during initialization "
                "to enable sampling via rvs()."
            )
        if self.operators is None:
            raise ValueError(
                "Cannot sample without operators. Set operators during initialization "
                "to enable sampling via rvs()."
            )

        seed = random_state if random_state is not None else self.random_state
        bingo_config = BingoConstructionConfig(
            max_complexity=self.max_complexity,
            terminal_probability=self.terminal_probability,
            constant_probability=self.constant_probability,
            command_probability=self.command_probability,
            node_probability=self.node_probability,
            parameter_probability=self.parameter_probability,
            prune_probability=self.prune_probability,
            fork_probability=self.fork_probability,
            repeat_mutation_probability=self.repeat_mutation_probability,
            crossover_pool_size=self.crossover_pool_size,
            mutation_prob=self.mutation_prob,
            crossover_prob=self.crossover_prob,
            exclusive=self.exclusive,
        )

        # Create generator and proposal
        generator = build_agraph_generator(self.x_dim, self.operators, bingo_config)
        prior = ImproperUniformPrior(generator)
        proposal = build_agraph_proposal(
            self.x_dim,
            self.operators,
            generator,
            bingo_config,
        )

        # Run SMC sampling with this prior's logpdf as the target
        models, _, _ = sample(
            likelihood=self._logpdf_single,
            proposal=proposal,
            prior=prior,
            max_time=self.max_time,
            max_equation_evals=self.max_equation_evals,
            seed=seed,
            checkpoint_file=self.checkpoint_file,
            multiprocess=self.multiprocess,
            kwargs={
                "num_particles": N,
                "num_mcmc_samples": self.num_mcmc_samples,
                "target_ess": self.target_ess,
            },
        )

        return np.array(models).reshape(-1, 1)
