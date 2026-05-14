"""Size-calibrated prior for symbolic expressions.

Implements the calibration formula:

.. math::

    P_{\\text{calibrated}}(T)
        = \\frac{p_{\\text{corpus}}(|T|) \\cdot P_{\\text{base}}(T)}{Z_{|T|}}

so that ``log P = log p_corpus(k) + log P_base(T) - log Z_k``.
"""

from math import inf
from typing import Callable, Dict, Optional, Union

from bingo.expressions.agraph import AGraphExpression
from bingo.expressions.agraph.evolvable import EvolvableExpression

from .samplable_prior import SamplablePrior
from .size_calibration_fitting import _default_size_fn


class SizeCalibratedPrior(SamplablePrior):
    """Prior that re-weights a base prior by corpus size distribution.

    Parameters
    ----------
    base_prior : SamplablePrior or None
        Underlying prior.  ``None`` means improper uniform (base log-prob
        is always 0).
    log_z_k : dict of {int: float}
        Pre-computed normalisation constants ``log Z_k`` per size.
    corpus_log_hist : dict of {int: float} or callable
        Log-probability of each size in the target corpus.  Either a
        dict mapping size to log-prob, or a callable ``(int) -> float``.
    floor_log_prob : float, optional
        Value returned for sizes not present in *corpus_log_hist*.
        Default is ``-inf``.
    size_fn : callable, optional
        Maps an AGraph to an integer size.  Default uses
        ``tree_complexity``.
    """

    def __init__(
        self,
        base_prior,
        log_z_k: Dict[int, float],
        corpus_log_hist: Union[Dict[int, float], Callable[[int], float]],
        floor_log_prob: float = -inf,
        size_fn: Optional[Callable[[AGraphExpression], int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_prior = base_prior
        self.log_z_k = dict(log_z_k)
        self.corpus_log_hist = corpus_log_hist
        self.floor_log_prob = floor_log_prob
        self._size_fn = size_fn if size_fn is not None else _default_size_fn

    def _logpdf_single(self, agraph: AGraphExpression) -> float:
        if isinstance(agraph, EvolvableExpression):
            agraph = agraph.expression

        k = self._size_fn(agraph)

        # Corpus term
        if callable(self.corpus_log_hist) and not isinstance(
            self.corpus_log_hist, dict
        ):
            log_corpus = self.corpus_log_hist(k)
        else:
            log_corpus = self.corpus_log_hist.get(k, self.floor_log_prob)

        # Z_k term
        if k not in self.log_z_k:
            return -inf

        log_zk = self.log_z_k[k]

        # Base prior term
        if self.base_prior is None:
            log_base = 0.0
        else:
            log_base = self.base_prior._logpdf_single(agraph)

        return log_corpus + log_base - log_zk

    def to_dict(self):
        """Serialize prior configuration to a JSON-compatible dict.

        The ``base_prior`` is **not** serialized — it must be
        reconstructed separately when loading.

        Returns
        -------
        dict
            Keys: ``log_z_k``, ``corpus_log_hist``, ``floor_log_prob``.
        """
        if callable(self.corpus_log_hist) and not isinstance(
            self.corpus_log_hist, dict
        ):
            raise TypeError(
                "Cannot serialize a callable corpus_log_hist. "
                "Convert to a dict first."
            )
        return {
            "log_z_k": {str(k): v for k, v in self.log_z_k.items()},
            "corpus_log_hist": {str(k): v for k, v in self.corpus_log_hist.items()},
            "floor_log_prob": self.floor_log_prob,
        }

    @classmethod
    def from_dict(cls, data, base_prior=None, **kwargs):
        """Reconstruct a ``SizeCalibratedPrior`` from a dict.

        Parameters
        ----------
        data : dict
            Output of :meth:`to_dict`.
        base_prior : SamplablePrior or None, optional
            Base prior (not stored in the dict).
        **kwargs
            Extra arguments forwarded to the constructor.

        Returns
        -------
        SizeCalibratedPrior
        """
        log_z_k = {int(k): v for k, v in data["log_z_k"].items()}
        corpus_log_hist = {int(k): v for k, v in data["corpus_log_hist"].items()}
        floor_log_prob = data.get("floor_log_prob", -inf)
        return cls(
            base_prior=base_prior,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_log_hist,
            floor_log_prob=floor_log_prob,
            **kwargs,
        )
