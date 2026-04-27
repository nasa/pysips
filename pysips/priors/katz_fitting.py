"""Katz back-off prior fitting from a corpus of AGraph expressions.

A single traversal of the corpus suffices: counts of all observed
n-grams (variable length up to ``n``) are tabulated for the
left/only and right-given-left models, then Simple Good-Turing
discounts are computed in closed form during fitting.
"""

from typing import List, Optional

from bingo.expressions.agraph import AGraphExpression

from .katz_backoff import KatzBackoffModel, KatzBackoffTreeModel
from .ngram_utils import ROOT, collect_phrases, vocabulary


def fit_katz_model(
    agraphs: List[AGraphExpression],
    n: int = 2,
    extra_operators: Optional[List[int]] = None,
    k: int = 0,
    unigram_smoothing: float = 0.0,
    min_alpha: float = 0.0,
    min_backoff_mass: float = 0.0,
) -> KatzBackoffTreeModel:
    """Fit a :class:`KatzBackoffTreeModel` to a corpus of expressions.

    Extracts all phrases from the corpus via
    :func:`~pysips.priors.ngram_utils.collect_phrases` and constructs
    a left/only model and a right-given-left model, both of order
    ``n`` (matching the paper's semantics: n = phrase length).

    Parameters
    ----------
    agraphs : list of AGraphExpression
        Training corpus.
    n : int, optional
        The n-gram order (phrase length). ``n=1`` means unigrams (no
        parent conditioning), ``n=2`` conditions on parent only,
        ``n=3`` conditions on parent + grandparent. Default is 2.
    extra_operators : list of int, optional
        Extra operator IDs to add to the vocabulary. Useful when the
        generator during sampling can produce operators not present in
        the corpus. Default is ``None``.
    k : int, optional
        Back-off count threshold (paper default: 0).
    unigram_smoothing : float, optional
        Laplace pseudo-count for the unigram base case. Default 0.0
        (raw frequency, matches reference). Set ``> 0`` to keep OOV
        unigrams strictly positive.
    min_alpha : float, optional
        Floor for the back-off weight :math:`\\alpha`. Default 0.0
        (matches reference).
    min_backoff_mass : float, optional
        Floor for the reserved unseen mass :math:`\\beta`. Default 0.0
        (matches reference).

    Returns
    -------
    KatzBackoffTreeModel
        Fitted tree model.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    left_phrases, right_phrases = collect_phrases(agraphs, n)

    vocab = vocabulary(left_phrases + right_phrases, extra=extra_operators)

    # Both models use max_order = n (the phrase length). Paper semantics:
    # n=1 -> unigrams, n=2 -> bigrams (parent), n=3 -> trigrams (parent+grandparent).
    max_order = n

    left_model = KatzBackoffModel(
        max_order=max_order,
        vocabulary=vocab,
        k=k,
        unigram_smoothing=unigram_smoothing,
        min_alpha=min_alpha,
        min_backoff_mass=min_backoff_mass,
    ).fit(left_phrases)
    right_model = KatzBackoffModel(
        max_order=max_order,
        vocabulary=vocab,
        k=k,
        unigram_smoothing=unigram_smoothing,
        min_alpha=min_alpha,
        min_backoff_mass=min_backoff_mass,
    ).fit(right_phrases)
    return KatzBackoffTreeModel(left_model=left_model, right_model=right_model, n=n)


def fit_katz_prior(
    agraphs: List[AGraphExpression],
    n: int = 2,
    x_dim: Optional[int] = None,
    extra_operators: Optional[List[int]] = None,
    k: int = 0,
    unigram_smoothing: float = 0.0,
    min_alpha: float = 0.0,
    min_backoff_mass: float = 0.0,
    **prior_kwargs,
):
    """Fit a Katz model and wrap it in a :class:`KatzPrior`.

    Convenience wrapper combining :func:`fit_katz_model` with
    :class:`~pysips.priors.katz_prior.KatzPrior` construction.
    """
    from .katz_prior import KatzPrior  # pylint: disable=import-outside-toplevel

    model = fit_katz_model(
        agraphs=agraphs,
        n=n,
        extra_operators=extra_operators,
        k=k,
        unigram_smoothing=unigram_smoothing,
        min_alpha=min_alpha,
        min_backoff_mass=min_backoff_mass,
    )
    return KatzPrior(model=model, x_dim=x_dim, **prior_kwargs)


__all__ = ["fit_katz_model", "fit_katz_prior", "ROOT"]
