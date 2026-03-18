"""
BMS Prior Fitting via Iterative SMC Sampling.

This module provides functions for fitting BMS prior weights to match
the operator frequency distribution of a target dataset of expressions.
The fitting uses an iterative approach:

1. Sample expressions from the current prior via SMC
2. Compute operator frequency statistics from samples
3. Update weights using SGD-like rule toward target frequencies
4. Repeat

Example
-------
>>> from pysips.priors import fit_bms_prior, load_corpus
>>> agraphs = load_corpus()  # Load training equations
>>> prior, history = fit_bms_prior(
...     agraphs,
...     x_dim=4,
...     operators=[2, 3, 4, 5, 6],  # +, -, *, /, sin
...     n_iterations=20,
...     num_particles=500,
...     verbose=True,
... )
>>> print(prior.weights)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from bingo.expressions.agraph import AGraphExpression
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph import (
    IS_TERMINAL_ARRAY,
    OPERATOR_NAMES,
)

from .bms_prior import BMSPrior, TerminalTreatment


def _get_aggregate_operator_counts(
    agraphs: List[AGraphExpression],
    tree: bool = True,
    terminals: TerminalTreatment = "exclude",
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute mean operator frequencies across a collection of AGraphs.

    Parameters
    ----------
    agraphs : list of AGraphExpression
        Collection of symbolic expressions to analyze.
    tree : bool, optional
        If True, perform depth-first tree traversal. Default is True.
    terminals : {"include", "exclude", "combine"}
        How to handle terminal nodes. Default is "exclude".

    Returns
    -------
    mean_counts : dict
        Mean count of each operator per expression.
    mean_squared_counts : dict
        Mean squared count of each operator per expression.
    """
    if not agraphs:
        return {}, {}

    aggregate_counts: Dict[int, int] = defaultdict(int)
    aggregate_squared_counts: Dict[int, int] = defaultdict(int)

    for agraph in agraphs:
        expr = agraph.expression if isinstance(agraph, EvolvableExpression) else agraph
        op_counts = expr.get_operator_counts(tree=tree, terminals=terminals)
        for op, count in op_counts.items():
            aggregate_counts[op] += count
            aggregate_squared_counts[op] += count**2

    n = len(agraphs)
    mean_counts = {op: aggregate_counts[op] / n for op in aggregate_counts}
    mean_squared_counts = {
        op: aggregate_squared_counts[op] / n for op in aggregate_squared_counts
    }

    return mean_counts, mean_squared_counts


def _update_weights(
    target: Dict[int, float],
    measured: Dict[int, float],
    weights: Dict[int, float],
    learning_rate: float = 0.02,
) -> Dict[int, float]:
    """
    Update weights using a stochastic gradient-descent-like rule.

    The update rule is:
        w_op += learning_rate * random() * (measured_op - target_op) / target_op

    Parameters
    ----------
    target : dict
        Target operator frequencies (from dataset).
    measured : dict
        Measured operator frequencies (from samples).
    weights : dict
        Current weight values (modified in-place).
    learning_rate : float, optional
        Learning rate for updates. Default is 0.02.

    Returns
    -------
    dict
        Updated weights (same object as input).
    """
    for op in target.keys():
        if target[op] == 0:
            continue

        e = np.random.rand()
        current_weight = weights[op]
        delta = learning_rate * e * (measured.get(op, 0) - target[op]) / target[op]
        weights[op] = current_weight + delta

    return weights


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def fit_bms_prior(
    agraphs: List[AGraphExpression],
    x_dim: int,
    operators: Optional[List[int]] = None,
    initial_weights: Optional[Union[float, Dict[int, float]]] = None,
    initial_squared_weights: Optional[Union[float, Dict[int, float]]] = None,
    initial_weight_value: float = 1.0,
    n_iterations: int = 20,
    learning_rate: float = 0.02,
    num_particles: int = 500,
    tree: bool = True,
    terminals: TerminalTreatment = "exclude",
    squared_weights_positivity_clipping: int = 0,
    verbose: bool = False,
    **kwargs,
) -> Tuple[BMSPrior, List[Dict[str, Any]]]:
    """
    Fit BMS prior weights to match operator frequencies of target expressions.

    This function iteratively:
    1. Samples expressions from the current BMSPrior via SMC
    2. Computes operator frequency statistics from the samples
    3. Updates weights toward the target distribution using SGD-like rule

    Parameters
    ----------
    agraphs : list of AGraphExpression
        Target dataset of symbolic expressions to match.
    x_dim : int
        Number of input variables for sampled expressions.
    operators : list of int, optional
        Operator IDs to use. If None, infers from target agraphs
        (excluding terminals).
    initial_weights : float or dict, optional
        Initial linear weights. Can be a scalar (applied to all operators)
        or a dict mapping operator IDs to values. Default uses
        initial_weight_value for all.
    initial_squared_weights : float or dict, optional
        Initial quadratic weights. If None, mirrors initial_weights.
    initial_weight_value : float, optional
        Default initial weight value when not specified. Default is 1.0.
    n_iterations : int, optional
        Number of fitting iterations. Default is 20.
    learning_rate : float, optional
        Weight update learning rate. Default is 0.02.
    num_particles : int, optional
        Number of SMC particles per iteration. Default is 500.
    tree : bool, optional
        If True, use tree traversal for counting operators. Default is True.
    terminals : {"include", "exclude", "combine"}, optional
        How to handle terminal nodes in counting. Default is "exclude".
    squared_weights_positivity_clipping : int, optional
        How to handle negative squared weights after update. Default is 0 (clip squared weights to 0).
        Higher values will ensure that operator Energy is non-negative at the specified frequency level.
    verbose : bool, optional
        If True, print progress during fitting. Default is False.
    **kwargs
        Additional arguments passed to BMSPrior (e.g., num_mcmc_samples,
        target_ess, max_complexity, etc.).

    Returns
    -------
    prior : BMSPrior
        Fitted BMSPrior with updated weights.
    history : list of dict
        Fitting history, each entry containing:
        - 'iteration': int
        - 'weights': dict (copy of weights at this iteration)
        - 'squared_weights': dict
        - 'measured_freq': dict (operator frequencies from samples)
        - 'measured_sq_freq': dict (squared frequencies from samples)
        - 'error': float (total frequency error)

    Examples
    --------
    >>> from pysips.priors import fit_bms_prior, load_corpus
    >>> agraphs = load_corpus()
    >>> prior, history = fit_bms_prior(
    ...     agraphs,
    ...     x_dim=4,
    ...     operators=[2, 3, 4],  # +, -, *
    ...     n_iterations=10,
    ...     verbose=True,
    ... )
    >>> # Check final weights
    >>> print(prior.weights)
    >>> # Plot convergence
    >>> errors = [h['error'] for h in history]
    """
    # Compute target operator frequencies from training data
    target_freq, target_sq_freq = _get_aggregate_operator_counts(
        agraphs, tree=tree, terminals=terminals
    )

    # Determine operators if not provided
    if operators is None:
        operators = [op for op in target_freq.keys() if not IS_TERMINAL_ARRAY[op]]

    if verbose:
        print(f"Fitting BMS prior with {len(operators)} operators")
        print(
            "Target frequencies:",
            {OPERATOR_NAMES[op][-1]: target_freq[op] for op in operators},
        )

    # Initialize weights
    if initial_weights is None:
        default_value = initial_weight_value
    elif isinstance(initial_weights, dict):
        default_value = initial_weight_value
    else:
        default_value = float(initial_weights)

    weights: Dict[int, float] = defaultdict(lambda: default_value)

    if isinstance(initial_weights, dict):
        for op, val in initial_weights.items():
            weights[op] = val

    # Ensure all target operators have explicit weight entries
    for op in target_freq.keys():
        _ = weights[op]  # Access to create key in defaultdict

    # Initialize squared weights
    if initial_squared_weights is None:
        if initial_weights is None:
            sq_default_value = initial_weight_value
        elif isinstance(initial_weights, dict):
            sq_default_value = initial_weight_value
        else:
            sq_default_value = float(initial_weights)

        squared_weights: Dict[int, float] = defaultdict(lambda: sq_default_value)

        if isinstance(initial_weights, dict):
            for op, val in initial_weights.items():
                squared_weights[op] = val
    elif isinstance(initial_squared_weights, dict):
        squared_weights = defaultdict(lambda: initial_weight_value)
        for op, val in initial_squared_weights.items():
            squared_weights[op] = val
    else:
        sq_value = float(initial_squared_weights)
        squared_weights = defaultdict(lambda: sq_value)

    # Ensure all target operators have explicit squared weight entries
    for op in target_sq_freq.keys():
        _ = squared_weights[op]

    history: List[Dict[str, Any]] = []

    for i in range(n_iterations):
        if verbose:
            print(f"\nIteration {i + 1}/{n_iterations}")

        # Create BMSPrior with current weights
        prior = BMSPrior(
            weights=dict(weights),
            squared_weights=dict(squared_weights),
            x_dim=x_dim,
            tree=tree,
            terminals=terminals,
            **kwargs,
        )

        # Sample from the prior using SMC
        samples = prior.rvs(num_particles).flatten().tolist()

        # Compute operator statistics from samples
        measured_freq, measured_sq_freq = _get_aggregate_operator_counts(
            samples, tree=tree, terminals=terminals
        )

        # Compute error for monitoring
        total_error = sum(
            abs(measured_freq.get(op, 0) - target_freq.get(op, 0)) for op in target_freq
        )

        # Record history
        history.append(
            {
                "iteration": i + 1,
                "weights": dict(weights),
                "squared_weights": dict(squared_weights),
                "measured_freq": dict(measured_freq),
                "measured_sq_freq": dict(measured_sq_freq),
                "error": total_error,
            }
        )

        if verbose:
            print(f"  Total frequency error: {total_error:.4f}")
            print(
                "  Measured:",
                {OPERATOR_NAMES[op][-1]: measured_freq[op] for op in measured_freq},
            )

        # Update weights toward target
        _update_weights(target_freq, measured_freq, weights, learning_rate)
        _update_weights(
            target_sq_freq, measured_sq_freq, squared_weights, learning_rate
        )

        # Clip squared weights to be non-negative
        if squared_weights_positivity_clipping == 0:
            for op in squared_weights:
                squared_weights[op] = max(0.0, squared_weights[op])
        elif squared_weights_positivity_clipping > 0:
            # Ensure operator Energy is non-negative at the specified frequency level
            for op in squared_weights:
                min_sq_weight = -(weights[op] / squared_weights_positivity_clipping)
                # min_sq_weight = min(min_sq_weight, 1e-5)
                squared_weights[op] = max(min_sq_weight, squared_weights[op])

    # Create final prior with fitted weights
    final_prior = BMSPrior(
        weights=dict(weights),
        squared_weights=dict(squared_weights),
        x_dim=x_dim,
        tree=tree,
        terminals=terminals,
        **kwargs,
    )

    if verbose:
        print(
            "\nFitting complete. Final weights:",
            {OPERATOR_NAMES[op][-1]: weights[op] for op in weights},
        )
        print(
            "Final squared weights:",
            {OPERATOR_NAMES[op][-1]: squared_weights[op] for op in squared_weights},
        )

    return final_prior, history
