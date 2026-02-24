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

Example
-------
>>> from pysips.priors import BMSPrior
>>> weights = {2: 0.5, 3: 0.3}        # addition, multiplication
>>> sq_weights = {2: 0.1, 3: 0.05}
>>> prior = BMSPrior(weights, sq_weights)
>>> log_p = prior.logpdf(agraphs)      # shape (N, 1)
"""

from collections import defaultdict
from typing import Dict, Literal

import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.operator_definitions import (
    IS_ARITY_2_MAP,
    IS_TERMINAL_MAP,
    VARIABLE,
)


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


class BMSPrior:
    """
    Bayesian Machine Scientist prior for symbolic expressions.

    Scores AGraph expressions based on weighted operator frequency counts.
    Designed for use as a log-prior in SMC/MCMC sampling pipelines.

    Parameters
    ----------
    weights : dict
        Linear operator weights mapping operator ID to weight value.
    squared_weights : dict
        Quadratic operator weights mapping operator ID to weight value.
    tree : bool, optional
        If True, use tree traversal for counting operators (repeated
        subgraphs counted multiple times). Default is True.
    terminals : {"include", "exclude", "combine"}, optional
        How to handle terminal nodes in operator counting.
        Default is "exclude".

    Examples
    --------
    >>> weights = {2: 0.5, 3: 0.3}
    >>> sq_weights = {2: 0.1, 3: 0.05}
    >>> prior = BMSPrior(weights, sq_weights)
    >>> log_probs = prior.logpdf(agraphs)
    """

    def __init__(
        self,
        weights: Dict[int, float],
        squared_weights: Dict[int, float],
        tree: bool = True,
        terminals: TerminalTreatment = "exclude",
    ):
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

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log-prior probability for an array of AGraph expressions.

        Parameters
        ----------
        x : array-like
            Array or list of AGraph objects. Accepted shapes include
            ``(N,)``, ``(N, 1)``

        Returns
        -------
        ndarray
            Array of shape ``(N, 1)`` with log-probability values.
        """
        log_probs = np.array([self._logpdf_single(ag) for ag in x])
        return log_probs.reshape(-1, 1)

    def _logpdf_single(self, agraph: AGraph) -> float:
        """Compute log-probability for a single AGraph expression."""
        operator_counts = _get_operator_counts(
            agraph, tree=self.tree, terminals=self.terminals
        )

        energy = 0.0
        for op, count in operator_counts.items():
            energy += count * self._weights[op] + count**2 * self._squared_weights[op]

        return -energy
