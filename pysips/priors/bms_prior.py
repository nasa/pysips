# pylint: disable=duplicate-code
"""
Bayesian Machine Scientist (BMS) Prior.

This module implements a prior distribution that scores symbolic expressions
based on weighted operator frequency counts, following the "Bayesian Machine
Scientist" approach. The log-probability of an expression is:

.. math::

    \\log P(\\text{expr}) = -\\sum_{\\text{op}} \\left[
        n_{\\text{op}} \\cdot w_{\\text{op}}
        + n_{\\text{op}}^2 \\cdot w2_{\\text{op}}
    \\right]

where :math:`n_{\\text{op}}` is the count of operator ``op`` in the
expression, :math:`w_{\\text{op}}` is the linear weight, and
:math:`w2_{\\text{op}}` is the quadratic weight.

The prior supports SMC-based sampling via the inherited ``rvs()`` method.
Operators are automatically inferred from the keys of the ``weights`` and
``squared_weights`` dictionaries.

Example
-------
>>> from pysips.priors import BMSPrior
>>> weights = {2: 0.5, 3: 0.3}        # addition, multiplication
>>> sq_weights = {2: 0.1, 3: 0.05}
>>> prior = BMSPrior(weights, sq_weights, x_dim=4)
>>> log_p = prior.logpdf(agraphs)      # shape (N, 1)
>>> samples = prior.rvs(100)           # sample 100 expressions
"""

from collections import defaultdict
from typing import Dict, Literal, Optional

from bingo.expressions.agraph import AGraphExpression
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph import (
    VARIABLE,
    CONSTANT,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    SIN,
    COS,
    EXPONENTIAL,
    LOGARITHM,
    POWER,
    ABS,
    SQRT,
    SAFE_POWER,
    SINH,
    COSH,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    TANH,
    SQUARE,
    CUBE,
)

from .samplable_prior import SamplablePrior


TerminalTreatment = Literal["include", "exclude", "combine"]

# Placeholder default weights for the BMS prior. These should be replaced
# with well-fit values obtained via fit_bms_prior().
# Keys are bingo operator integer IDs from
# bingo.expressions.agraph.pyagraph.operators.
DEFAULT_BMS_WEIGHTS: Dict[int, float] = {
    VARIABLE: 1.0,
    CONSTANT: 1.0,
    ADDITION: 1.0,
    SUBTRACTION: 1.0,
    MULTIPLICATION: 1.0,
    DIVISION: 1.0,
    SIN: 1.0,
    COS: 1.0,
    EXPONENTIAL: 1.0,
    LOGARITHM: 1.0,
    POWER: 1.0,
    ABS: 1.0,
    SQRT: 1.0,
    SAFE_POWER: 1.0,
    SINH: 1.0,
    COSH: 1.0,
    TAN: 1.0,
    ARCSIN: 1.0,
    ARCCOS: 1.0,
    ARCTAN: 1.0,
    TANH: 1.0,
    SQUARE: 1.0,
    CUBE: 1.0,
}

DEFAULT_BMS_SQUARED_WEIGHTS: Dict[int, float] = {
    VARIABLE: 0.0,
    CONSTANT: 0.0,
    ADDITION: 0.0,
    SUBTRACTION: 0.0,
    MULTIPLICATION: 0.0,
    DIVISION: 0.0,
    SIN: 0.0,
    COS: 0.0,
    EXPONENTIAL: 0.0,
    LOGARITHM: 0.0,
    POWER: 0.0,
    ABS: 0.0,
    SQRT: 0.0,
    SAFE_POWER: 0.0,
    SINH: 0.0,
    COSH: 0.0,
    TAN: 0.0,
    ARCSIN: 0.0,
    ARCCOS: 0.0,
    ARCTAN: 0.0,
    TANH: 0.0,
    SQUARE: 0.0,
    CUBE: 0.0,
}


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals
class BMSPrior(SamplablePrior):
    """
    Bayesian Machine Scientist prior for symbolic expressions.

    Scores AGraph expressions based on weighted operator frequency counts.
    Designed for use as a log-prior in SMC/MCMC sampling pipelines.
    Supports sampling from the prior via the ``rvs()`` method.

    Parameters
    ----------
    weights : dict
        Linear operator weights mapping operator ID to weight value.
        Operators are inferred from the keys of this dictionary.
    squared_weights : dict
        Quadratic operator weights mapping operator ID to weight value.
    x_dim : int, optional
        Number of input variables (dimension of X data). Required for
        sampling via ``rvs()``. Default is None.
    tree : bool, optional
        If True, use tree traversal for counting operators (repeated
        subgraphs counted multiple times). Default is True.
    terminals : {"include", "exclude", "combine"}, optional
        How to handle terminal nodes in operator counting.
        Default is "exclude".
    num_particles : int, optional
        Number of SMC particles for sampling. Default is 50.
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
    max_complexity : int, optional
        Maximum allowable AGraph complexity (node count). Default is 24.
    terminal_probability : float, optional
        Probability of generating terminal nodes when building expressions.
        Default is 0.1.
    constant_probability : float, optional
        Probability of generating constants. If None, computed as ``1/(x_dim + 1)``.
    command_probability : float, optional
        Probability of applying command mutations. Default is 0.2.
    node_probability : float, optional
        Probability of mutating nodes. Default is 0.2.
    parameter_probability : float, optional
        Probability of mutating parameters. Default is 0.2.
    prune_probability : float, optional
        Probability of pruning mutation operations. Default is 0.2.
    fork_probability : float, optional
        Probability of fork mutation operations. Default is 0.2.
    repeat_mutation_probability : float, optional
        Probability of repeating mutation steps. Default is 0.05.
    crossover_pool_size : int, optional
        Size of the crossover pool. If None, defaults to ``num_particles`` or 50.
    mutation_prob : float, optional
        Probability of selecting the mutation proposal. Default is 0.75.
    crossover_prob : float, optional
        Probability of selecting the crossover proposal. Default is 0.25.
    exclusive : bool, optional
        If True, mutation and crossover proposals are mutually exclusive.
        Default is True.

    Examples
    --------
    >>> weights = {2: 0.5, 3: 0.3}
    >>> sq_weights = {2: 0.1, 3: 0.05}
    >>> prior = BMSPrior(weights, sq_weights, x_dim=4)
    >>> log_probs = prior.logpdf(agraphs)
    >>> samples = prior.rvs(100)  # Sample 100 expressions from prior
    """

    def __init__(
        self,
        weights: Dict[int, float],
        squared_weights: Dict[int, float],
        x_dim: Optional[int] = None,
        tree: bool = True,
        terminals: TerminalTreatment = "exclude",
        num_mcmc_samples: int = 5,
        target_ess: float = 0.8,
        max_time: Optional[float] = None,
        max_equation_evals: Optional[int] = None,
        checkpoint_file: Optional[str] = None,
        random_state: Optional[int] = None,
        multiprocess: bool = False,
        max_complexity: int = 24,
        terminal_probability: float = 0.1,
        constant_probability: Optional[float] = None,
        command_probability: float = 0.2,
        node_probability: float = 0.2,
        parameter_probability: float = 0.2,
        prune_probability: float = 0.2,
        fork_probability: float = 0.2,
        repeat_mutation_probability: float = 0.05,
        crossover_pool_size: Optional[int] = None,
        mutation_prob: float = 0.75,
        crossover_prob: float = 0.25,
        exclusive: bool = True,
    ):
        # Initialize parent class with SMC parameters
        super().__init__(
            x_dim=x_dim,
            operators=list(set(weights.keys()) | set(squared_weights.keys())),
            num_mcmc_samples=num_mcmc_samples,
            target_ess=target_ess,
            max_time=max_time,
            max_equation_evals=max_equation_evals,
            checkpoint_file=checkpoint_file,
            random_state=random_state,
            multiprocess=multiprocess,
            max_complexity=max_complexity,
            terminal_probability=terminal_probability,
            constant_probability=constant_probability,
            command_probability=command_probability,
            node_probability=node_probability,
            parameter_probability=parameter_probability,
            prune_probability=prune_probability,
            fork_probability=fork_probability,
            repeat_mutation_probability=repeat_mutation_probability,
            crossover_pool_size=crossover_pool_size,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            exclusive=exclusive,
        )

        self.tree = tree
        self.terminals = terminals

        self._weights: Dict[int, float] = defaultdict(float)
        self._weights.update(weights)

        self._squared_weights: Dict[int, float] = defaultdict(float)
        self._squared_weights.update(squared_weights)

    @property
    def weights(self) -> Dict[int, float]:
        """Linear operator weights."""
        return dict(self._weights)

    @property
    def squared_weights(self) -> Dict[int, float]:
        """Quadratic operator weights."""
        return dict(self._squared_weights)

    def _logpdf_single(self, agraph) -> float:
        """Compute log-probability for a single AGraph expression."""
        if isinstance(agraph, EvolvableExpression):
            agraph = agraph.expression
        operator_counts = agraph.get_operator_counts(
            tree=self.tree, terminals=self.terminals
        )

        energy = 0.0
        for op, count in operator_counts.items():
            energy += count * self._weights[op] + count**2 * self._squared_weights[op]

        return -energy
