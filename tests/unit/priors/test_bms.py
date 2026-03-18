import numpy as np
import pytest
from unittest.mock import MagicMock

from pysips.priors import BMSPrior

SAMPLEABLEPRIOR_MODULE = "pysips.priors.samplable_prior"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# New agraph operator IDs (from bingo.expressions.agraph.pyagraph.operators)
VARIABLE = 0
CONSTANT = 1
INTEGER = 2
ADDITION = 3
SUBTRACTION = 4
MULTIPLICATION = 5


def _make_mock_agraph(operator_counts):
    """Create a mock AGraphExpression with a preset get_operator_counts result."""
    ag = MagicMock()
    ag.get_operator_counts = MagicMock(return_value=operator_counts)
    return ag


class TestBMSPrior:
    """Tests for the BMSPrior class."""

    def test_init_stores_weights(self):
        weights = {2: 0.5, 3: 0.3}
        sq_weights = {2: 0.1, 3: 0.05}
        prior = BMSPrior(weights, sq_weights, tree=False, terminals="include")

        assert prior.weights == weights
        assert prior.squared_weights == sq_weights
        assert prior.tree is False
        assert prior.terminals == "include"

    def test_default_weight_is_zero_for_missing_ops(self):
        prior = BMSPrior({2: 1.0}, {2: 0.5})
        # Access a key that wasn't provided — should default to 0.0
        assert prior._weights[99] == 0.0
        assert prior._squared_weights[99] == 0.0

    def test_logpdf_returns_correct_shape(self):
        """logpdf should return shape (N, 1)."""
        prior = BMSPrior({ADDITION: 0.5}, {ADDITION: 0.1})
        mock_ags = [_make_mock_agraph({ADDITION: 1}) for _ in range(5)]

        result = prior.logpdf(mock_ags)
        assert result.shape == (5, 1)

    def test_logpdf_single_element(self):
        """logpdf should handle a single-element array."""
        prior = BMSPrior({ADDITION: 0.5}, {ADDITION: 0.1})
        result = prior.logpdf([_make_mock_agraph({ADDITION: 1})])
        assert result.shape == (1, 1)

    def test_logpdf_2d_input(self):
        """logpdf should handle (N, 1) shaped input."""
        prior = BMSPrior({ADDITION: 0.5}, {ADDITION: 0.1})
        mock_ags = [_make_mock_agraph({ADDITION: 2}) for _ in range(3)]

        result = prior.logpdf(mock_ags)
        assert result.shape == (3, 1)

    def test_logpdf_computes_correct_value(self):
        """Verify the energy computation: -sum(n*w + n^2*w2)."""
        # Expression has 2 additions and 1 multiplication
        weights = {ADDITION: 0.5, MULTIPLICATION: 0.3}
        sq_weights = {ADDITION: 0.1, MULTIPLICATION: 0.05}
        prior = BMSPrior(weights, sq_weights)

        ag = _make_mock_agraph({ADDITION: 2, MULTIPLICATION: 1})

        # Energy = 2*0.5 + 4*0.1 + 1*0.3 + 1*0.05 = 1.0 + 0.4 + 0.3 + 0.05 = 1.75
        # logpdf = -1.75
        result = prior.logpdf([ag])
        np.testing.assert_almost_equal(result[0, 0], -1.75)

    def test_logpdf_zero_weights_give_zero_logprob(self):
        """Zero weights should yield log-probability of 0."""
        prior = BMSPrior({}, {})  # all default to 0.0
        ag = _make_mock_agraph({ADDITION: 3, MULTIPLICATION: 2})
        result = prior.logpdf([ag])
        np.testing.assert_almost_equal(result[0, 0], 0.0)

    def test_logpdf_multiple_expressions_independent(self):
        """Each expression should be scored independently."""
        weights = {ADDITION: 1.0}
        sq_weights = {ADDITION: 0.0}
        prior = BMSPrior(weights, sq_weights)

        ag1 = _make_mock_agraph({ADDITION: 1})
        ag2 = _make_mock_agraph({ADDITION: 3})

        result = prior.logpdf([ag1, ag2])

        # First: -(1*1.0) = -1.0, Second: -(3*1.0) = -3.0
        np.testing.assert_almost_equal(result[0, 0], -1.0)
        np.testing.assert_almost_equal(result[1, 0], -3.0)


class TestBMSPriorRvs:
    """Tests for BMSPrior sampling via rvs()."""

    def test_rvs_raises_without_x_dim(self):
        """rvs() should raise ValueError if x_dim was not set."""
        prior = BMSPrior({2: 0.5}, {2: 0.1})  # No x_dim

        with pytest.raises(ValueError, match="Cannot sample without x_dim"):
            prior.rvs(10)

    def test_rvs_calls_sample_with_correct_params(self, mocker):
        """rvs should call sample() with correct params and return proper shape."""
        mock_models = [MagicMock(spec=[]) for _ in range(10)]
        mock_sample = mocker.patch(
            f"{SAMPLEABLEPRIOR_MODULE}.sample", return_value=(mock_models, None, None)
        )

        prior = BMSPrior(
            weights={2: 0.5, 3: 0.3},
            squared_weights={2: 0.1, 3: 0.05},
            x_dim=4,
            num_mcmc_samples=3,
            target_ess=0.9,
        )

        result = prior.rvs(10)

        # Verify sample was called with correct params
        mock_sample.assert_called_once()
        call_kwargs = mock_sample.call_args.kwargs
        assert call_kwargs["kwargs"]["num_particles"] == 10
        assert call_kwargs["kwargs"]["num_mcmc_samples"] == 3
        assert call_kwargs["kwargs"]["target_ess"] == 0.9
        assert result.shape == (10, 1)

    @pytest.mark.parametrize(
        "instance_seed,call_seed,expected",
        [
            (42, None, 42),  # Uses instance random_state
            (42, 99, 99),  # Call parameter overrides instance
        ],
    )
    def test_rvs_random_state(self, mocker, instance_seed, call_seed, expected):
        """rvs should use random_state correctly."""
        mock_sample = mocker.patch(
            f"{SAMPLEABLEPRIOR_MODULE}.sample",
            return_value=([MagicMock(spec=[])], None, None),
        )

        prior = BMSPrior({2: 0.5}, {2: 0.1}, x_dim=3, random_state=instance_seed)
        prior.rvs(1, random_state=call_seed)

        assert mock_sample.call_args.kwargs["seed"] == expected

    def test_operator_identification_from_weights(self):
        """BMSPrior should identify operators from weights dict."""
        prior = BMSPrior({2: 0.5, 4: 0.2}, {2: 0.1, 5: 0.05}, x_dim=5)
        assert set(prior.operators) == {2, 4, 5}


class TestBMSPriorBackwardCompatibility:
    """Tests ensuring backward compatibility for BMSPrior without x_dim."""

    def test_logpdf_works_without_x_dim(self):
        """logpdf should work even if x_dim is not set."""
        # Create prior without x_dim (backward compatible)
        prior = BMSPrior({ADDITION: 0.5}, {ADDITION: 0.1})

        ag = _make_mock_agraph({ADDITION: 1})
        result = prior.logpdf([ag])
        assert result.shape == (1, 1)

    def test_init_with_minimal_args(self):
        """BMSPrior should accept just weights and squared_weights."""
        prior = BMSPrior({2: 0.5}, {2: 0.1})

        assert prior.weights == {2: 0.5}
        assert prior.squared_weights == {2: 0.1}
        assert prior.x_dim is None
        assert prior.operators == [2]
