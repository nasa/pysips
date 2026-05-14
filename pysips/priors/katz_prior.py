# pylint: disable=duplicate-code
"""
Katz Back-off Language Model Prior.

Implements a structural prior on symbolic expression trees based on
operator n-grams, following Bartlett et al. (GECCO 2023, Sec 2.4).
The log-probability of an expression is the sum of log-probabilities
of all phrases extracted from its tree, where each phrase is evaluated
under either the "left/only" or "right-given-left" Katz back-off model.

The prior supports SMC sampling via :meth:`rvs` inherited from
:class:`~pysips.priors.samplable_prior.SamplablePrior`. Fit a prior
to a corpus via :func:`~pysips.priors.katz_fitting.fit_katz_prior`,
or use :func:`load_katz_model` to load a pre-fit model (or fit
one on the fly).

Example
-------
>>> from pysips.priors import KatzPrior, load_katz_model
>>> model = load_katz_model(n=2, corpus="benchmark")
>>> prior = KatzPrior(model, x_dim=4)
>>> log_p = prior.logpdf(agraphs)       # shape (N, 1)
>>> samples = prior.rvs(100)            # SMC sample 100 expressions
"""

import json
from pathlib import Path
from typing import List, Optional

from bingo.expressions.agraph import AGraphExpression
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph import (
    ABS,
    ADDITION,
    ARCCOS,
    ARCSIN,
    ARCTAN,
    CONSTANT,
    COS,
    COSH,
    CUBE,
    DIVISION,
    EXPONENTIAL,
    IS_TERMINAL_ARRAY,
    LOGARITHM,
    MULTIPLICATION,
    POWER,
    SAFE_POWER,
    SIN,
    SINH,
    SQRT,
    SQUARE,
    SUBTRACTION,
    TAN,
    TANH,
    VARIABLE,
)

from .katz_backoff import KatzBackoffTreeModel
from .ngram_utils import extract_phrases
from .samplable_prior import SamplablePrior

KATZ_MODEL_DIR = Path(__file__).parent / "data"


# Operators used when the prior is constructed without explicit operator
# list and the Katz model's vocabulary alone is insufficient for
# generation (e.g., missing terminals are still supported internally).
# Mirrors the set used in BMSPrior defaults.
_COMMON_OPERATORS = [
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
]


def _default_operators_from_vocab(vocab_tokens: List[int]) -> List[int]:
    """Return generation-compatible operator list (non-terminal tokens)."""
    return [op for op in vocab_tokens if not IS_TERMINAL_ARRAY[op]]


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals
class KatzPrior(SamplablePrior):
    """Prior over symbolic expressions based on a Katz back-off n-gram model.

    Scores expressions via the product of conditional probabilities
    produced by two Katz back-off models (for left/only children and
    for right-given-left children). See
    :mod:`pysips.priors.katz_backoff` for the probability model.

    Parameters
    ----------
    model : KatzBackoffTreeModel
        Fitted pair of Katz models. Use
        :func:`~pysips.priors.katz_fitting.fit_katz_prior` or
        :func:`load_katz_model` to obtain one.
    normalize : bool, optional
        If ``True``, divide the total log-probability by the number
        of phrases (left + right), yielding a per-phrase mean
        log-probability (cross-entropy).  This is the standard
        length normalization for n-gram models and prevents bias
        toward shorter expressions.  Default is ``False``.
    operators : list of int, optional
        Operator IDs to use for generation during SMC sampling. If
        ``None``, derived from the union of both models' vocabularies
        (non-terminals only).
    x_dim : int, optional
        Number of input variables; required for ``rvs()``.
    num_mcmc_samples, target_ess, max_time, max_equation_evals,
    checkpoint_file, random_state, multiprocess, max_complexity,
    terminal_probability, constant_probability, command_probability,
    node_probability, parameter_probability, prune_probability,
    fork_probability, repeat_mutation_probability, crossover_pool_size,
    mutation_prob, crossover_prob, exclusive
        See :class:`~pysips.priors.bms_prior.BMSPrior` /
        :class:`~pysips.bingo_proposal_mixin.BingoProposalMixin` for
        details; forwarded to the parent class.
    """

    def __init__(
        self,
        model: KatzBackoffTreeModel,
        normalize: bool = False,
        operators: Optional[List[int]] = None,
        x_dim: Optional[int] = None,
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
        self.model = model
        self._normalize = normalize

        if operators is None:
            vocab = list(
                set(model.left_model.vocabulary) | set(model.right_model.vocabulary)
            )
            operators = _default_operators_from_vocab(vocab)
            if not operators:
                # Fallback: use the common operator set if vocab has no
                # non-terminals (pathological edge case).
                operators = list(_COMMON_OPERATORS)

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

    @property
    def n(self) -> int:
        """Depth window of the underlying Katz models (paper's ``n``)."""
        return self.model.n

    def _logpdf_single(self, agraph) -> float:
        """Return the total log-probability for a single AGraph expression."""
        if isinstance(agraph, EvolvableExpression):
            agraph = agraph.expression
        left_phrases, right_phrases = extract_phrases(agraph, self.model.n)
        total = self.model.log_prob_phrases(left_phrases, right_phrases)
        if self._normalize:
            n_phrases = len(left_phrases) + len(right_phrases)
            if n_phrases == 0:
                return 0.0
            return total / n_phrases
        return total


def _model_path(n: int, corpus: str = "benchmark") -> Path:
    return KATZ_MODEL_DIR / f"default_katz_n{n}_{corpus}.json"


def load_katz_model(
    n: int = 2, corpus: str = "benchmark", fit_if_missing: bool = True
) -> KatzBackoffTreeModel:
    """Load a Katz model, fitting from corpus if no prefit exists.

    Attempts to load a pre-fit model from
    ``data/default_katz_n{n}_{corpus}.json``.  When no saved model is
    found and *fit_if_missing* is ``True``, the corpus is loaded via
    :func:`~pysips.priors.data.load_corpus.load_corpus`, a new model
    is fit with :func:`~pysips.priors.katz_fitting.fit_katz_model`,
    and the result is saved for future reuse.

    Parameters
    ----------
    n : int, optional
        N-gram order (phrase length). Default is 2.
    corpus : str, optional
        Corpus name recognised by :func:`load_corpus` (e.g.
        ``"wikipedia"``, ``"feynman"``, ``"benchmark"``).
        Default is ``"wikipedia"``.
    fit_if_missing : bool, optional
        If ``True`` (default), fit and save the model when no prefit
        file exists. If ``False``, raise :class:`FileNotFoundError`.

    Returns
    -------
    KatzBackoffTreeModel
        Fitted tree model.

    Raises
    ------
    FileNotFoundError
        If no pre-fit model exists and *fit_if_missing* is ``False``.
    """
    path = _model_path(n, corpus)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return KatzBackoffTreeModel.from_dict(data)

    if not fit_if_missing:
        available = sorted(KATZ_MODEL_DIR.glob("default_katz_n*.json"))
        raise FileNotFoundError(
            f"No pre-fit Katz model for n={n}, corpus={corpus!r} at "
            f"{path}. Available: {[p.name for p in available]}"
        )

    # Fit on the fly and cache for future use
    from .data.load_corpus import load_corpus  # pylint: disable=import-outside-toplevel
    from .katz_fitting import fit_katz_model  # pylint: disable=import-outside-toplevel

    print(
        f"No pre-fit Katz model for n={n}, corpus={corpus!r}. "
        f"Fitting from corpus (this may take a moment)..."
    )
    agraphs = load_corpus(corpus)
    model = fit_katz_model(agraphs, n=n)
    save_katz_model(model, path)
    print(f"Saved new Katz model to {path}")
    return model


def save_katz_model(model: KatzBackoffTreeModel, path) -> None:
    """Save a :class:`KatzBackoffTreeModel` to JSON on disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)
