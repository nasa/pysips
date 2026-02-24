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
from typing import Dict, List, Literal, Optional

import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.operator_definitions import (
    IS_ARITY_2_MAP,
    IS_TERMINAL_MAP,
    VARIABLE,
)

from .samplable_prior import SamplablePrior


TerminalTreatment = Literal["include", "exclude", "combine"]


def _get_operator_counts(
    agraph: AGraph,
    tree: bool = True,
    terminals: TerminalTreatment = "include",
) -> Dict[int, int]:
    """
    Count the occurrences of each operator in an AGraph expression.

    Parameters
    ----------
    agraph : AGraph
        The symbolic expression to analyze.
    tree : bool, optional
        If True, perform depth-first tree traversal (counts repeated
        subgraphs multiple times). If False, count unique nodes in the
        DAG. Default is True.
    terminals : {"include", "exclude", "combine"}
        How to handle terminal nodes:
        - "include": Count each terminal type separately.
        - "exclude": Don't count terminal nodes.
        - "combine": Combine all terminals into a single "Variable" category.
        Default is "include".

    Returns
    -------
    dict
        Mapping from operator ID (int) to count.
    """
    operator_counts: Dict[int, int] = defaultdict(int)

    agraph._update()
    command_array = agraph._simplified_command_array

    if tree:
        stack = [command_array[-1]]
        while stack:
            node, param1, param2 = stack.pop()

            if IS_TERMINAL_MAP[node]:
                if terminals == "exclude":
                    continue
                elif terminals == "combine":
                    operator_counts[VARIABLE] += 1
                    continue
                else:
                    operator_counts[node] += 1
                    continue

            operator_counts[node] += 1
            stack.append(command_array[param1])
            if IS_ARITY_2_MAP[node]:
                stack.append(command_array[param2])
    else:
        for node, _, _ in command_array:
            if IS_TERMINAL_MAP[node]:
                if terminals == "exclude":
                    continue
                elif terminals == "combine":
                    operator_counts[VARIABLE] += 1
                    continue
                else:
                    operator_counts[node] += 1
                    continue
            operator_counts[node] += 1

    return dict(operator_counts)


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

    def _logpdf_single(self, agraph: AGraph) -> float:
        """Compute log-probability for a single AGraph expression."""
        operator_counts = _get_operator_counts(
            agraph, tree=self.tree, terminals=self.terminals
        )

        energy = 0.0
        for op, count in operator_counts.items():
            energy += count * self._weights[op] + count**2 * self._squared_weights[op]

        return -energy
