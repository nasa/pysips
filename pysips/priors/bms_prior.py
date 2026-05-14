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
Operators for generation/sampling can be specified explicitly via the
``operators`` parameter, or are automatically inferred from the keys of
the ``weights`` and ``squared_weights`` dictionaries if not provided.

Pre-fit weights for different corpora are stored as JSON files in
``pysips/priors/data/`` and loaded via :func:`load_bms_weights`.

Example
-------
>>> from pysips.priors import BMSPrior, load_bms_weights
>>> weights, sq_weights = load_bms_weights("benchmark")
>>> prior = BMSPrior(weights, sq_weights, x_dim=4)
>>> log_p = prior.logpdf(agraphs)      # shape (N, 1)
>>> samples = prior.rvs(100)           # sample 100 expressions
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from bingo.expressions.agraph.evolvable import EvolvableExpression

from .samplable_prior import SamplablePrior


TerminalTreatment = Literal["include", "exclude", "combine"]

BMS_WEIGHTS_DIR = Path(__file__).parent / "data"


def _bms_weights_path(corpus: str = "benchmark") -> Path:
    return BMS_WEIGHTS_DIR / f"default_bms_{corpus}.json"


def load_bms_weights(
    corpus: str = "benchmark",
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Load pre-fit BMS weights for a given corpus.

    Loads linear and quadratic operator weights from a JSON file in
    ``pysips/priors/data/default_bms_{corpus}.json``.

    Parameters
    ----------
    corpus : str, optional
        Corpus name (e.g. ``"wikipedia"``, ``"benchmark"``).
        Default is ``"benchmark"``.

    Returns
    -------
    weights : dict
        Linear operator weights mapping operator ID (int) to weight.
    squared_weights : dict
        Quadratic operator weights mapping operator ID (int) to weight.

    Raises
    ------
    FileNotFoundError
        If no pre-fit weights file exists for the given corpus.
    """
    path = _bms_weights_path(corpus)
    if not path.exists():
        available = sorted(BMS_WEIGHTS_DIR.glob("default_bms_*.json"))
        raise FileNotFoundError(
            f"No pre-fit BMS weights for corpus {corpus!r} at {path}. "
            f"Available: {[p.name for p in available]}. "
            f"Use fit_bms_prior() to fit custom weights."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    weights = {int(k): v for k, v in data["weights"].items()}
    squared_weights = {int(k): v for k, v in data["squared_weights"].items()}
    return weights, squared_weights


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
        Used for scoring expressions via ``logpdf()``.
    squared_weights : dict
        Quadratic operator weights mapping operator ID to weight value.
        Used for scoring expressions via ``logpdf()``.
    operators : list of int, optional
        List of operator IDs to use for generation/sampling. If None,
        operators are inferred from the keys of ``weights`` and
        ``squared_weights``. Default is None.
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
        operators: Optional[List[int]] = None,
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
        # Derive operators from weights if not explicitly provided
        if operators is None:
            operators = list(set(weights.keys()) | set(squared_weights.keys()))

        # Initialize parent class with SMC parameters
        super().__init__(
            x_dim=x_dim,
            operators=operators,
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
