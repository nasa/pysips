"""Fitting orchestrator for size-calibrated priors.

Provides utilities for fitting a size-calibrated prior from a corpus
of symbolic expressions:

- :func:`classify_operators` — separate operators by arity.
- :func:`compute_size_histogram` — empirical log-PMF from expressions.
- :func:`fit_size_calibrated_prior` — main entry point with auto-dispatch.
"""

import logging
from collections import Counter
from math import log
from typing import Callable, Dict, List, Optional, Tuple, Union

from bingo.expressions.agraph import AGraphExpression
from bingo.expressions.agraph.pyagraph import (
    IS_ARITY_2_ARRAY,
    IS_TERMINAL_ARRAY,
)

from pysips.priors.data.load_corpus import load_corpus
from pysips.priors.size_calibration_z_k import (
    compute_labeled_tree_counts,
    estimate_log_z_k_katz,
    estimate_log_z_k_mc,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Size function                                                       #
# ------------------------------------------------------------------ #


def _default_size_fn(agraph):
    """Tree-based node count (consistent with Katz phrase extraction)."""
    return agraph.tree_complexity


# ------------------------------------------------------------------ #
# Corpus size histogram                                               #
# ------------------------------------------------------------------ #


def compute_size_histogram(
    agraphs: List[AGraphExpression],
    size_fn: Optional[Callable[[AGraphExpression], int]] = None,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Compute a log-probability histogram of expression sizes.

    Parameters
    ----------
    agraphs : list of AGraphExpression
        Corpus of expressions.
    size_fn : callable, optional
        Maps an AGraph to an integer size.  Defaults to
        :func:`_default_size_fn` (tree node count).

    Returns
    -------
    log_hist : dict of {int: float}
        Mapping from size *k* to ``log(count_k / total)``.
    counts : dict of {int: int}
        Raw counts per size.
    """
    if size_fn is None:
        size_fn = _default_size_fn
    if not agraphs:
        raise ValueError("agraphs must be non-empty")

    raw = Counter(size_fn(ag) for ag in agraphs)
    total = sum(raw.values())
    log_hist = {k: log(c / total) for k, c in raw.items()}
    return log_hist, dict(raw)


# ------------------------------------------------------------------ #
# Operator classification                                             #
# ------------------------------------------------------------------ #


def classify_operators(operators: List[int], x_dim: int) -> Dict[str, object]:
    """Classify an operator list by arity.

    Parameters
    ----------
    operators : list of int
        Non-terminal operator IDs from the user's operator set.
    x_dim : int
        Number of input variables (determines terminal multiplicity).

    Returns
    -------
    dict
        ``n_t`` : int — number of distinct terminal labels
        (``x_dim`` variables + 1 constant).
        ``n_u`` : int — number of unary operators.
        ``n_b`` : int — number of binary operators.
        ``unary_ids`` : list of int — unary operator IDs.
        ``binary_ids`` : list of int — binary operator IDs.
    """
    unary_ids = [
        op for op in operators if not IS_ARITY_2_ARRAY[op] and not IS_TERMINAL_ARRAY[op]
    ]
    binary_ids = [op for op in operators if IS_ARITY_2_ARRAY[op]]
    n_t = x_dim + 1
    return {
        "n_t": n_t,
        "n_u": len(unary_ids),
        "n_b": len(binary_ids),
        "unary_ids": unary_ids,
        "binary_ids": binary_ids,
    }


# ------------------------------------------------------------------ #
# fit_size_calibrated_prior                                           #
# ------------------------------------------------------------------ #


def fit_size_calibrated_prior(
    base_prior,
    corpus: Union[str, List[AGraphExpression]],
    x_dim: int,
    operators: List[int],
    size_fn: Optional[Callable[[AGraphExpression], int]] = None,
    max_size: int = 100,
    n_shapes_per_size: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, object]:
    """Fit a size-calibrated prior from a corpus.

    Orchestrates corpus histogram extraction and Z_k estimation,
    auto-dispatching to the most efficient estimation method based
    on the base prior type.

    Parameters
    ----------
    base_prior : SamplablePrior, str, or None
        Base prior.  ``None`` or ``"uniform"`` means improper uniform
        (analytical Z_k).  A ``KatzPrior`` triggers the hybrid
        estimator.  Any other ``SamplablePrior`` uses MC fallback.
    corpus : str or list of AGraphExpression
        Corpus name (forwarded to :func:`load_corpus`) or list of
        pre-loaded AGraph expressions.
    x_dim : int
        Number of input variables.
    operators : list of int
        Non-terminal operator IDs.
    size_fn : callable, optional
        Size measure.  Default is tree node count.
    max_size : int, optional
        Maximum tree size for Z_k estimation.  Default 100.
    n_shapes_per_size : int, optional
        Shapes per size for hybrid/MC estimation.  Default 1000.
    random_state : int or None, optional
        Random seed.

    Returns
    -------
    dict
        ``"log_z_k"`` : dict of {int: float}
            Estimated log Z_k per size.
        ``"corpus_log_hist"`` : dict of {int: float}
            Log-PMF of corpus size distribution.
        ``"corpus_counts"`` : dict of {int: int}
            Raw counts per size.
        ``"base_prior"`` : object or None
            Resolved base prior instance.
        ``"estimation_info"`` : dict or None
            Diagnostics from Z_k estimation (MC/hybrid only).
    """
    # Resolve corpus
    if isinstance(corpus, str):
        agraphs = load_corpus(corpus)
    else:
        agraphs = corpus

    # Corpus histogram
    corpus_log_hist, corpus_counts = compute_size_histogram(agraphs, size_fn)
    logger.info(
        "Corpus: %d expressions, sizes %d..%d",
        sum(corpus_counts.values()),
        min(corpus_counts),
        max(corpus_counts),
    )

    # Resolve base prior
    if base_prior is None or base_prior == "uniform":
        resolved_prior = None
    else:
        resolved_prior = base_prior

    # Z_k estimation with auto-dispatch
    estimation_info = None
    if resolved_prior is None:
        # Uniform: analytical Z_k
        info = classify_operators(operators, x_dim)
        log_z_k = compute_labeled_tree_counts(
            info["n_t"], info["n_u"], info["n_b"], max_size
        )
        logger.info("Uniform base: analytical Z_k for %d sizes", len(log_z_k))
    elif hasattr(resolved_prior, "model"):
        # Katz: hybrid Z_k
        result = estimate_log_z_k_katz(
            resolved_prior.model,
            operators,
            x_dim,
            max_size,
            n_shapes_per_size=n_shapes_per_size,
            random_state=random_state,
        )
        log_z_k = result["log_z_k"]
        estimation_info = result
        logger.info(
            "Katz base: hybrid Z_k for %d sizes (%d shapes/size)",
            len(log_z_k),
            n_shapes_per_size,
        )
    else:
        # MC fallback
        result = estimate_log_z_k_mc(
            resolved_prior,
            x_dim,
            operators,
            max_size,
            n_shapes_per_size=n_shapes_per_size,
            random_state=random_state,
        )
        log_z_k = result["log_z_k"]
        estimation_info = result
        logger.info(
            "MC fallback: Z_k for %d sizes (%d shapes/size)",
            len(log_z_k),
            n_shapes_per_size,
        )

    return {
        "log_z_k": log_z_k,
        "corpus_log_hist": corpus_log_hist,
        "corpus_counts": corpus_counts,
        "base_prior": resolved_prior,
        "estimation_info": estimation_info,
    }
