"""
Katz back-off language model for operator n-grams.

Implements the Katz back-off formalism used in Bartlett, Desmond &
Ferreira (GECCO 2023, Sec 2.4, Eq. 14-16). The conditional probability
of a word given a context is

.. math::

    P(w_i | w_{i-n+1}, \\ldots, w_{i-1}) = \\begin{cases}
        d \\cdot \\frac{C(w_{i-n+1}, \\ldots, w_i)}{C(w_{i-n+1}, \\ldots, w_{i-1})}
            & \\text{if } C(\\cdot) > k \\\\
        \\alpha \\cdot P(w_i | w_{i-n+2}, \\ldots, w_{i-1})
            & \\text{otherwise}
    \\end{cases}

where :math:`d = C^*/C` is a Good-Turing discount with :math:`C^*`
estimated by Simple Good-Turing (Gale & Sampson, 1995) -- log-linear
regression through :math:`(\\log r, \\log Z_r)`. The smoothed
frequency :math:`Z_r` is computed as
:math:`N_r / (0.5 (r_{\\text{next}} - r_{\\text{prev}}))` for
interior observed counts, and :math:`N_r / (r - r_{\\text{prev}})`
for the largest observed count.

The `n` parameter follows the **paper text** (Bartlett et al. 2023):
``n=1`` produces unigrams, ``n=2`` conditions on the parent, ``n=3``
conditions on parent + grandparent. Note that the reference code at
https://github.com/DeaglanBartlett/katz uses ``n`` with an
off-by-one relative to the paper: its ``n=2`` is equivalent to this
module's ``n=3``.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import exp, log
from typing import Dict, List, Optional, Tuple

from .ngram_utils import ROOT


# Token type: operator IDs (int) plus the ROOT sentinel (kept for legacy).
Token = int
Phrase = Tuple[Token, ...]
Context = Tuple[Token, ...]


@dataclass
class KatzBackoffModel:
    """A single Katz back-off distribution over phrases of variable order.

    Stores counts for phrases (and their shorter suffixes) up to
    ``max_order``, then evaluates conditional probabilities by
    recursively backing off when the full context is unseen.

    Parameters
    ----------
    max_order : int
        Maximum phrase length (number of tokens including the terminal
        word). Must be ``>= 1``.
    vocabulary : list of int
        All tokens that may appear as phrase terminators. Required to
        compute :math:`\\alpha` normalization.
    k : int, optional
        Count threshold below which to back off (paper's ``k``).
        Default is 0 (paper default).
    unigram_smoothing : float, optional
        Pseudo-count added to every vocabulary token in the unigram
        base case. Default is 0.0 (no smoothing -- matches the
        reference implementation, which uses raw empirical
        frequencies). Set to ``> 0`` to guarantee strictly positive
        unigram probabilities (Laplace-style smoothing).
    min_backoff_mass : float, optional
        Minimum fraction of probability mass reserved for the back-off
        path per context. Default is 0.0 (no floor -- matches the
        reference implementation). Setting ``> 0`` ensures
        :math:`\\alpha > 0` even when discounts leave no mass for
        unseen tokens, useful when scoring out-of-corpus expressions
        with very small training corpora.
    min_alpha : float, optional
        Minimum value for the back-off weight :math:`\\alpha`. Default
        is 0.0 (no floor -- matches the reference implementation).

    Notes
    -----
    Call :meth:`fit` once with a list of phrases before calling
    :meth:`log_prob`. Discounts are computed during fitting via
    Simple Good-Turing with log-linear smoothing of
    frequency-of-frequency (Gale & Sampson 1995): :math:`Z_r` is
    averaged over the gap between neighbouring observed counts to
    handle "gappy" :math:`N_r` data, then a single line is fit through
    :math:`(\\log r, \\log Z_r)` and the discount
    :math:`d = (r+1) S(r+1) / (r S(r))` uses the smoothed
    :math:`S(r)` for *all* counts. No switching between raw and
    smoothed estimates is done, matching the reference code.
    """

    max_order: int
    vocabulary: List[Token]
    k: int = 0
    unigram_smoothing: float = 0.0
    min_backoff_mass: float = 0.0
    min_alpha: float = 0.0

    # Internal state populated by fit().
    _phrase_counts: Dict[int, Dict[Phrase, int]] = field(default_factory=dict)
    _context_counts: Dict[int, Dict[Context, int]] = field(default_factory=dict)
    _context_seen_words: Dict[int, Dict[Context, set]] = field(default_factory=dict)
    _discounts: Dict[int, Dict[int, float]] = field(default_factory=dict)
    _alpha_cache: Dict[Tuple[int, Context], float] = field(default_factory=dict)
    _is_fit: bool = False

    def __post_init__(self):
        if self.max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {self.max_order}")
        # Deduplicate vocab while preserving order.
        seen = set()
        clean_vocab: List[Token] = []
        for tok in self.vocabulary:
            if tok != ROOT and tok not in seen:
                clean_vocab.append(tok)
                seen.add(tok)
        self.vocabulary = clean_vocab

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(self, phrases: List[Phrase]) -> "KatzBackoffModel":
        """Build count tables from a list of (variable-length) phrases.

        For every phrase of length :math:`L`, counts are incremented at
        every order from 1 up to :math:`\\min(L, \\text{max\\_order})`
        using the rightmost tokens (so the phrase terminator is always
        included).
        """
        self._phrase_counts = {
            order: defaultdict(int) for order in range(1, self.max_order + 1)
        }
        self._context_counts = {
            order: defaultdict(int) for order in range(1, self.max_order + 1)
        }
        self._context_seen_words = {
            order: defaultdict(set) for order in range(1, self.max_order + 1)
        }

        for phrase in phrases:
            if not phrase:
                continue
            L = len(phrase)
            max_obs_order = min(L, self.max_order)
            for order in range(1, max_obs_order + 1):
                sub = phrase[L - order :]
                self._phrase_counts[order][sub] += 1
                context = sub[:-1]
                word = sub[-1]
                self._context_counts[order][context] += 1
                self._context_seen_words[order][context].add(word)

        self._compute_discounts()
        self._alpha_cache.clear()
        self._is_fit = True
        return self

    def _compute_discounts(self) -> None:
        """Pre-compute discount factors per order and count.

        Implements Simple Good-Turing (Gale & Sampson 1995) exactly as
        in the paper authors' reference code: log-linear regression of
        :math:`Z_r` against :math:`r`, then ``d = (r+1) S(r+1) /
        (r S(r))`` for every observed ``r``. No switching rule, no
        clamping.
        """
        self._discounts = {}
        for order in range(1, self.max_order + 1):
            freq_of_freq: Counter = Counter(self._phrase_counts[order].values())
            discounts: Dict[int, float] = {}

            rs = sorted(freq_of_freq.keys())
            if not rs:
                self._discounts[order] = discounts
                continue
            if len(rs) < 2:
                # Cannot fit a regression with a single point; fall back
                # to the raw count (no discounting).
                for r in rs:
                    discounts[r] = 1.0
                self._discounts[order] = discounts
                continue

            # Step 1: Z_r averages N_r over the gap to its neighbours.
            #
            # Following Gale & Sampson (and the reference code):
            #   Z_r = N_r / (0.5 * (r_next - r_prev))   for interior r
            # with r_prev = 0 prepended for the first point, and
            #   Z_r = N_r / (r_last - r_prev)           for the last point.
            n = len(rs)
            Z = [0.0] * n
            for idx, r in enumerate(rs):
                r_prev = rs[idx - 1] if idx > 0 else 0
                if idx < n - 1:
                    r_next = rs[idx + 1]
                    Z[idx] = freq_of_freq[r] / (0.5 * (r_next - r_prev))
                else:
                    # Final point: full distance back to r_prev (no halving).
                    Z[idx] = freq_of_freq[r] / (r - r_prev)

            # Step 2: log-linear regression on (log r, log Z_r).
            log_rs = [log(r) for r in rs]
            log_Zs = [log(z) for z in Z]
            sum_x = sum(log_rs)
            sum_y = sum(log_Zs)
            sum_xx = sum(x * x for x in log_rs)
            sum_xy = sum(x * y for x, y in zip(log_rs, log_Zs))
            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) < 1e-12:
                for r in rs:
                    discounts[r] = 1.0
                self._discounts[order] = discounts
                continue
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n

            def S(r: int) -> float:
                return exp(slope * log(r) + intercept)

            # Step 3: d = (r+1) S(r+1) / (r S(r)). No clipping.
            for r in rs:
                discounts[r] = (r + 1) * S(r + 1) / (r * S(r))
            self._discounts[order] = discounts

    # ------------------------------------------------------------------ #
    # Probability evaluation                                              #
    # ------------------------------------------------------------------ #

    def log_prob(self, phrase: Phrase) -> float:
        """Return log P(phrase[-1] | phrase[:-1]) under the back-off model."""
        if not self._is_fit:
            raise RuntimeError("KatzBackoffModel must be fit before use")
        if not phrase:
            raise ValueError("phrase must contain at least one token")

        if len(phrase) > self.max_order:
            phrase = phrase[-self.max_order :]
        p = self._prob(phrase)
        if p <= 0.0:
            return float("-inf")
        return log(p)

    def _prob(self, phrase: Phrase) -> float:
        order = len(phrase)
        word = phrase[-1]
        context = phrase[:-1]

        if order == 1:
            return self._unigram_prob(word)

        c_cw = self._phrase_counts[order].get(phrase, 0)
        if c_cw > self.k:
            c_c = self._context_counts[order][context]
            d = self._discounts[order].get(c_cw, 1.0)
            return d * c_cw / c_c

        # Back off: P(w|c) = alpha(c) * P(w|c[1:]).
        # Following the reference implementation: if the context was
        # never observed at this order (no prefix count), alpha=1 and
        # we drop directly to the shorter context.
        if self._context_counts[order].get(context, 0) == 0:
            return self._prob(phrase[1:])
        alpha = self._alpha(order, context)
        return alpha * self._prob(phrase[1:])

    def _alpha(self, order: int, context: Context) -> float:
        """Compute the Katz back-off weight for a given context.

        Mirrors :meth:`get_alpha` from the reference back-off code:
        :math:`\\beta = 1 - \\sum_{w \\in \\text{seen}} d(c, w) C(c, w) /
        C(c)` and
        :math:`\\alpha = \\beta / (1 - \\sum_{w \\in \\text{seen}}
        P_{bo}(w | c[1:]))`.
        """
        cache_key = (order, context)
        if cache_key in self._alpha_cache:
            return self._alpha_cache[cache_key]

        seen_words = self._context_seen_words[order].get(context, set())
        if not seen_words:
            # Context entirely unseen at this order: pass everything through.
            self._alpha_cache[cache_key] = 1.0
            return 1.0

        c_c = self._context_counts[order][context]
        seen_mass = 0.0
        for w in seen_words:
            c_cw = self._phrase_counts[order][context + (w,)]
            if c_cw > self.k:
                d = self._discounts[order].get(c_cw, 1.0)
                seen_mass += d * c_cw / c_c
        beta = 1.0 - seen_mass
        if self.min_backoff_mass > 0.0 and beta < self.min_backoff_mass:
            beta = self.min_backoff_mass

        # Denominator: 1 - sum_{seen} P_bo(w | shorter_context).
        shorter_context = context[1:]
        seen_pbo_sum = 0.0
        for w in seen_words:
            seen_pbo_sum += self._prob(shorter_context + (w,))
        denom = 1.0 - seen_pbo_sum

        if denom <= 0.0:
            # Mathematically equivalent fallback: sum P_bo over unseen.
            unseen = [w for w in self.vocabulary if w not in seen_words]
            denom = sum(self._prob(shorter_context + (w,)) for w in unseen)
            if denom <= 0.0:
                denom = 1.0

        alpha = beta / denom
        if self.min_alpha > 0.0 and alpha < self.min_alpha:
            alpha = self.min_alpha
        self._alpha_cache[cache_key] = alpha
        return alpha

    def _unigram_prob(self, word: Token) -> float:
        """Unigram probability (base case of recursion).

        Default behaviour matches the reference implementation: raw
        empirical frequency :math:`C(w) / N`. When
        ``unigram_smoothing > 0``, additive Laplace smoothing is applied
        to guarantee strictly positive probabilities for OOV tokens.
        """
        counts = self._phrase_counts[1]
        total_tokens = self._context_counts[1].get((), 0)
        smooth = self.unigram_smoothing
        c_w = counts.get((word,), 0)

        if smooth <= 0.0:
            if total_tokens <= 0:
                return 0.0
            return c_w / total_tokens

        vocab_size = len(self.vocabulary)
        denom = total_tokens + smooth * max(vocab_size, 1)
        if denom <= 0:
            return 1.0 / max(vocab_size, 1)
        if word in self.vocabulary:
            return (c_w + smooth) / denom
        return smooth / denom

    # ------------------------------------------------------------------ #
    # Serialization helpers                                               #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict:
        """Return a plain-dict representation suitable for JSON/pickle."""
        return {
            "max_order": self.max_order,
            "vocabulary": list(self.vocabulary),
            "k": self.k,
            "unigram_smoothing": self.unigram_smoothing,
            "min_backoff_mass": self.min_backoff_mass,
            "min_alpha": self.min_alpha,
            "phrase_counts": {
                str(order): {
                    ",".join(str(t) for t in phrase): cnt
                    for phrase, cnt in phrases.items()
                }
                for order, phrases in self._phrase_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KatzBackoffModel":
        """Reconstruct a fitted model from :meth:`to_dict` output."""
        model = cls(
            max_order=data["max_order"],
            vocabulary=list(data["vocabulary"]),
            k=data.get("k", 0),
            unigram_smoothing=data.get("unigram_smoothing", 0.0),
            min_backoff_mass=data.get("min_backoff_mass", 0.0),
            min_alpha=data.get("min_alpha", 0.0),
        )
        model._phrase_counts = {
            order: defaultdict(int) for order in range(1, model.max_order + 1)
        }
        model._context_counts = {
            order: defaultdict(int) for order in range(1, model.max_order + 1)
        }
        model._context_seen_words = {
            order: defaultdict(set) for order in range(1, model.max_order + 1)
        }
        for order_str, phrase_map in data["phrase_counts"].items():
            order = int(order_str)
            for phrase_key, cnt in phrase_map.items():
                phrase = tuple(int(t) for t in phrase_key.split(","))
                model._phrase_counts[order][phrase] = cnt
                context = phrase[:-1]
                word = phrase[-1]
                model._context_counts[order][context] += cnt
                model._context_seen_words[order][context].add(word)
        model._compute_discounts()
        model._is_fit = True
        return model

    def __repr__(self) -> str:
        fitted = "fit" if self._is_fit else "unfit"
        n_phrases = (
            sum(len(pc) for pc in self._phrase_counts.values()) if self._is_fit else 0
        )
        return (
            f"KatzBackoffModel(max_order={self.max_order}, "
            f"vocab_size={len(self.vocabulary)}, n_distinct_phrases={n_phrases}, "
            f"{fitted})"
        )


@dataclass
class KatzBackoffTreeModel:
    """Pair of Katz models for tree-structured expressions.

    Bundles the "left/only" model (used for the root, unary children,
    and left children of binary operators) with the "right-given-left"
    model (used for right children of binary operators). See
    :mod:`pysips.priors.ngram_utils` for details on phrase construction.

    Parameters
    ----------
    left_model : KatzBackoffModel
    right_model : KatzBackoffModel
    n : int
        Phrase length (n-gram order) following the paper text.
        ``n=1`` is unigrams (no parent conditioning), ``n=2``
        conditions on the parent, ``n=3`` on parent + grandparent.
        Both sub-models are fitted with ``max_order = n``.
    """

    left_model: KatzBackoffModel
    right_model: KatzBackoffModel
    n: int

    def log_prob_phrases(
        self,
        left_phrases: List[Phrase],
        right_phrases: List[Phrase],
    ) -> float:
        """Sum log-probabilities over all phrases from a single expression."""
        total = 0.0
        for p in left_phrases:
            total += self.left_model.log_prob(p)
        for p in right_phrases:
            total += self.right_model.log_prob(p)
        return total

    def to_dict(self) -> Dict:
        return {
            "n": self.n,
            "left_model": self.left_model.to_dict(),
            "right_model": self.right_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KatzBackoffTreeModel":
        return cls(
            left_model=KatzBackoffModel.from_dict(data["left_model"]),
            right_model=KatzBackoffModel.from_dict(data["right_model"]),
            n=data["n"],
        )
