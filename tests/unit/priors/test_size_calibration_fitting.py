"""Tests for size_calibration_fitting module."""

from math import log, exp

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph import (
    ADDITION,
    CONSTANT,
    DIVISION,
    MULTIPLICATION,
    SUBTRACTION,
    VARIABLE,
)

SIN = 15
COS = 16
EXPONENTIAL = 13
LOGARITHM = 14


# ===================================================================
# classify_operators
# ===================================================================


class TestClassifyOperators:
    """Tests for classify_operators."""

    def test_mixed_operator_set(self):
        """Standard operator set is correctly separated."""
        from pysips.priors.size_calibration_fitting import classify_operators

        ops = [ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, SIN, COS]
        result = classify_operators(ops, x_dim=2)

        assert result["n_t"] == 3  # x0, x1, constant
        assert result["n_u"] == 2  # sin, cos
        assert result["n_b"] == 4  # +, -, *, /
        assert set(result["unary_ids"]) == {SIN, COS}
        assert set(result["binary_ids"]) == {
            ADDITION,
            SUBTRACTION,
            MULTIPLICATION,
            DIVISION,
        }

    def test_binary_only(self):
        """Operator set with only binary operators."""
        from pysips.priors.size_calibration_fitting import classify_operators

        ops = [ADDITION, MULTIPLICATION]
        result = classify_operators(ops, x_dim=1)

        assert result["n_t"] == 2  # x0, constant
        assert result["n_u"] == 0
        assert result["n_b"] == 2
        assert result["unary_ids"] == []


# ===================================================================
# compute_size_histogram
# ===================================================================


class TestComputeSizeHistogram:
    """Tests for compute_size_histogram."""

    def test_valid_log_pmf(self):
        """Histogram log-probabilities sum to ~0 in log-space."""
        from pysips.priors.size_calibration_fitting import (
            compute_size_histogram,
        )
        from scipy.special import logsumexp

        # Create mock agraphs with known tree_complexity values
        agraphs = []
        for size in [3, 3, 3, 5, 5, 7]:
            ag = _make_mock_agraph(size)
            agraphs.append(ag)

        log_hist, counts = compute_size_histogram(agraphs)

        # log-probabilities should sum to ~0 (i.e., probabilities sum to 1)
        log_total = logsumexp(list(log_hist.values()))
        np.testing.assert_almost_equal(log_total, 0.0, decimal=10)

        # raw counts should be correct
        assert counts == {3: 3, 5: 2, 7: 1}

    def test_correct_log_probabilities(self):
        """Individual log-probabilities are correct."""
        from pysips.priors.size_calibration_fitting import (
            compute_size_histogram,
        )

        agraphs = [_make_mock_agraph(s) for s in [1, 1, 1, 3]]

        log_hist, _ = compute_size_histogram(agraphs)

        np.testing.assert_almost_equal(exp(log_hist[1]), 0.75)
        np.testing.assert_almost_equal(exp(log_hist[3]), 0.25)

    def test_empty_corpus_raises(self):
        """Empty corpus raises ValueError."""
        from pysips.priors.size_calibration_fitting import (
            compute_size_histogram,
        )

        with pytest.raises(ValueError, match="non-empty"):
            compute_size_histogram([])

    def test_custom_size_fn(self):
        """Custom size function is used instead of default."""
        from pysips.priors.size_calibration_fitting import (
            compute_size_histogram,
        )

        agraphs = [_make_mock_agraph(5), _make_mock_agraph(5)]

        # Custom size_fn that always returns 42
        log_hist, counts = compute_size_histogram(agraphs, size_fn=lambda ag: 42)

        assert counts == {42: 2}


# ===================================================================
# fit_size_calibrated_prior
# ===================================================================


class TestFitSizeCalibratedPrior:
    """Tests for fit_size_calibrated_prior."""

    def test_dispatches_analytical_for_uniform(self, mocker):
        """Uniform base prior triggers analytical Z_k estimation."""
        from pysips.priors.size_calibration_fitting import (
            fit_size_calibrated_prior,
        )

        mock_tree_counts = mocker.patch(
            "pysips.priors.size_calibration_fitting.compute_labeled_tree_counts",
            return_value={1: 0.0, 3: 1.0, 5: 2.0},
        )

        ops = [ADDITION, SIN]
        corpus = [_make_mock_agraph(s) for s in [3, 3, 5]]
        result = fit_size_calibrated_prior(
            base_prior=None,
            corpus=corpus,
            x_dim=1,
            operators=ops,
        )

        mock_tree_counts.assert_called_once()
        assert result["log_z_k"] == {1: 0.0, 3: 1.0, 5: 2.0}
        assert "corpus_log_hist" in result

    def test_dispatches_hybrid_for_katz(self, mocker):
        """KatzPrior triggers hybrid Z_k estimation."""
        from pysips.priors.size_calibration_fitting import (
            fit_size_calibrated_prior,
        )
        from pysips.priors.katz_prior import KatzPrior
        from unittest.mock import MagicMock

        mock_katz_z_k = mocker.patch(
            "pysips.priors.size_calibration_fitting.estimate_log_z_k_katz",
            return_value={
                "log_z_k": {1: 0.0, 3: 1.5},
                "shape_counts": {},
                "n_shapes_sampled": {},
                "low_ess_sizes": [],
            },
        )

        mock_prior = MagicMock(spec=KatzPrior)
        mock_prior.model = MagicMock()

        ops = [ADDITION, SIN]
        corpus = [_make_mock_agraph(s) for s in [3, 5]]
        result = fit_size_calibrated_prior(
            base_prior=mock_prior,
            corpus=corpus,
            x_dim=1,
            operators=ops,
        )

        mock_katz_z_k.assert_called_once()
        assert result["log_z_k"] == {1: 0.0, 3: 1.5}

    def test_dispatches_mc_for_other_priors(self, mocker):
        """Non-Katz, non-uniform prior triggers MC fallback."""
        from pysips.priors.size_calibration_fitting import (
            fit_size_calibrated_prior,
        )
        from pysips.priors.samplable_prior import SamplablePrior
        from unittest.mock import MagicMock

        mock_mc_z_k = mocker.patch(
            "pysips.priors.size_calibration_fitting.estimate_log_z_k_mc",
            return_value={
                "log_z_k": {1: 0.0, 3: 2.0},
                "shape_counts": {},
                "n_shapes_sampled": {},
                "low_ess_sizes": [],
            },
        )

        # A generic prior (not KatzPrior, not None/uniform)
        mock_prior = MagicMock(spec=SamplablePrior)
        # Ensure it doesn't have .model attribute (not Katz-like)
        del mock_prior.model

        ops = [ADDITION]
        corpus = [_make_mock_agraph(s) for s in [3]]
        result = fit_size_calibrated_prior(
            base_prior=mock_prior,
            corpus=corpus,
            x_dim=1,
            operators=ops,
        )

        mock_mc_z_k.assert_called_once()
        assert result["log_z_k"] == {1: 0.0, 3: 2.0}

    def test_accepts_corpus_string(self, mocker):
        """Corpus name string is resolved via load_corpus."""
        from pysips.priors.size_calibration_fitting import (
            fit_size_calibrated_prior,
        )

        mock_corpus = [_make_mock_agraph(s) for s in [3, 5, 7]]
        mocker.patch(
            "pysips.priors.size_calibration_fitting.load_corpus",
            return_value=mock_corpus,
        )
        mocker.patch(
            "pysips.priors.size_calibration_fitting.compute_labeled_tree_counts",
            return_value={1: 0.0, 3: 1.0},
        )

        ops = [ADDITION]
        result = fit_size_calibrated_prior(
            base_prior=None,
            corpus="benchmark",
            x_dim=1,
            operators=ops,
        )

        assert "corpus_log_hist" in result
        assert 3 in result["corpus_log_hist"]

    def test_accepts_agraph_list(self, mocker):
        """List of AGraphExpression objects works directly."""
        from pysips.priors.size_calibration_fitting import (
            fit_size_calibrated_prior,
        )

        mocker.patch(
            "pysips.priors.size_calibration_fitting.compute_labeled_tree_counts",
            return_value={1: 0.0},
        )

        corpus = [_make_mock_agraph(s) for s in [1, 3, 5]]
        result = fit_size_calibrated_prior(
            base_prior=None,
            corpus=corpus,
            x_dim=1,
            operators=[ADDITION],
        )

        assert result["corpus_counts"] == {1: 1, 3: 1, 5: 1}

    def test_uniform_string_dispatches_analytical(self, mocker):
        """base_prior='uniform' triggers analytical path."""
        from pysips.priors.size_calibration_fitting import (
            fit_size_calibrated_prior,
        )

        mock_tree_counts = mocker.patch(
            "pysips.priors.size_calibration_fitting.compute_labeled_tree_counts",
            return_value={1: 0.0},
        )

        corpus = [_make_mock_agraph(3)]
        result = fit_size_calibrated_prior(
            base_prior="uniform",
            corpus=corpus,
            x_dim=1,
            operators=[ADDITION],
        )

        mock_tree_counts.assert_called_once()

    def test_uses_logging_not_print(self, mocker):
        """Fitting uses logging module, not print()."""
        from pysips.priors.size_calibration_fitting import (
            fit_size_calibrated_prior,
        )

        mocker.patch(
            "pysips.priors.size_calibration_fitting.compute_labeled_tree_counts",
            return_value={1: 0.0},
        )
        mock_print = mocker.patch("builtins.print")

        corpus = [_make_mock_agraph(3)]
        fit_size_calibrated_prior(
            base_prior=None,
            corpus=corpus,
            x_dim=1,
            operators=[ADDITION],
        )

        mock_print.assert_not_called()


# ===================================================================
# Helpers
# ===================================================================


def _make_mock_agraph(tree_complexity_value):
    """Create a mock AGraphExpression with a given tree_complexity."""
    from unittest.mock import MagicMock
    from bingo.expressions.agraph import AGraphExpression

    ag = MagicMock(spec=AGraphExpression)
    ag.tree_complexity = tree_complexity_value
    return ag
