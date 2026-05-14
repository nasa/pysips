"""Tests for SizeCalibratedPrior."""

from math import log
from unittest.mock import MagicMock

import numpy as np
import pytest

from bingo.expressions.agraph import AGraphExpression


def _make_mock_agraph(tree_complexity_value):
    """Create a mock AGraphExpression with a given tree_complexity."""
    ag = MagicMock(spec=AGraphExpression)
    ag.tree_complexity = tree_complexity_value
    return ag


class TestSizeCalibratedPrior:
    """Tests for SizeCalibratedPrior."""

    def test_correct_logpdf_uniform_base_dict_corpus(self):
        """Calibrated log-prob is corpus + base - Z_k; uniform base => base=0."""
        from pysips.priors.size_calibrated_prior import SizeCalibratedPrior

        log_z_k = {3: log(10), 5: log(20)}
        corpus_log_hist = {3: log(0.6), 5: log(0.4)}

        prior = SizeCalibratedPrior(
            base_prior=None,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_log_hist,
        )

        ag = _make_mock_agraph(3)
        result = prior._logpdf_single(ag)

        # log p_corpus(3) + 0 - log Z_3
        expected = log(0.6) + 0 - log(10)
        np.testing.assert_almost_equal(result, expected)

    def test_callable_corpus_log_hist(self):
        """corpus_log_hist as a callable produces the same result."""
        from pysips.priors.size_calibrated_prior import SizeCalibratedPrior

        log_z_k = {3: log(10), 5: log(20)}
        corpus_fn = lambda k: log(0.6) if k == 3 else log(0.4)

        prior = SizeCalibratedPrior(
            base_prior=None,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_fn,
        )

        ag = _make_mock_agraph(3)
        result = prior._logpdf_single(ag)

        expected = log(0.6) - log(10)
        np.testing.assert_almost_equal(result, expected)

    def test_floor_log_prob_for_out_of_range_size(self):
        """floor_log_prob is returned when size is not in corpus_log_hist dict."""
        from pysips.priors.size_calibrated_prior import SizeCalibratedPrior

        log_z_k = {3: 0.0, 7: 0.0}
        corpus_log_hist = {3: log(0.5)}
        floor = -100.0

        prior = SizeCalibratedPrior(
            base_prior=None,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_log_hist,
            floor_log_prob=floor,
        )

        ag = _make_mock_agraph(7)  # size 7 not in corpus_log_hist
        result = prior._logpdf_single(ag)

        # corpus term = floor, base = 0, Z_k = 0
        expected = floor + 0.0 - 0.0
        np.testing.assert_almost_equal(result, expected)

    def test_default_floor_is_neg_inf(self):
        """Default floor_log_prob is -inf, giving -inf for out-of-range sizes."""
        from pysips.priors.size_calibrated_prior import SizeCalibratedPrior

        log_z_k = {3: 0.0, 7: 0.0}
        corpus_log_hist = {3: log(0.5)}

        prior = SizeCalibratedPrior(
            base_prior=None,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_log_hist,
        )

        ag = _make_mock_agraph(7)
        result = prior._logpdf_single(ag)
        assert result == -np.inf

    def test_neg_inf_when_log_z_k_missing(self):
        """Returns -inf when log_z_k has no entry for the expression size."""
        from pysips.priors.size_calibrated_prior import SizeCalibratedPrior

        log_z_k = {3: 0.0}  # only size 3
        corpus_log_hist = {3: log(0.5), 5: log(0.5)}

        prior = SizeCalibratedPrior(
            base_prior=None,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_log_hist,
        )

        ag = _make_mock_agraph(5)  # size 5 in corpus but not in log_z_k
        result = prior._logpdf_single(ag)
        assert result == -np.inf

    def test_logpdf_with_real_base_prior(self):
        """Base prior's logpdf_single is included in the calibrated score."""
        from pysips.priors.size_calibrated_prior import SizeCalibratedPrior

        base = MagicMock()
        base._logpdf_single = MagicMock(return_value=-2.5)

        log_z_k = {3: log(10)}
        corpus_log_hist = {3: log(0.6)}

        prior = SizeCalibratedPrior(
            base_prior=base,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_log_hist,
        )

        ag = _make_mock_agraph(3)
        result = prior._logpdf_single(ag)

        expected = log(0.6) + (-2.5) - log(10)
        np.testing.assert_almost_equal(result, expected)
        base._logpdf_single.assert_called_once()

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization roundtrip produces equivalent logpdf values."""
        from pysips.priors.size_calibrated_prior import SizeCalibratedPrior

        log_z_k = {3: log(10), 5: log(20)}
        corpus_log_hist = {3: log(0.6), 5: log(0.4)}
        floor = -50.0

        original = SizeCalibratedPrior(
            base_prior=None,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_log_hist,
            floor_log_prob=floor,
        )

        data = original.to_dict()
        restored = SizeCalibratedPrior.from_dict(data)

        # Check same logpdf values for multiple sizes
        for size in [3, 5, 7]:
            ag = _make_mock_agraph(size)
            np.testing.assert_almost_equal(
                original._logpdf_single(ag),
                restored._logpdf_single(ag),
            )

    def test_to_dict_from_dict_preserves_floor(self):
        """Roundtrip preserves floor_log_prob."""
        from pysips.priors.size_calibrated_prior import SizeCalibratedPrior

        original = SizeCalibratedPrior(
            base_prior=None,
            log_z_k={3: 0.0},
            corpus_log_hist={3: 0.0},
            floor_log_prob=-42.0,
        )

        data = original.to_dict()
        restored = SizeCalibratedPrior.from_dict(data)

        assert restored.floor_log_prob == -42.0
