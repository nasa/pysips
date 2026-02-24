import numpy as np
import pytest
from unittest.mock import MagicMock

from pysips.priors import BMSPrior
from pysips.priors.bms_prior import _get_operator_counts

PRIORMODULE = BMSPrior.__module__
SAMPLEABLEPRIOR_MODULE = "pysips.priors.samplable_prior"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agraph(simplified_command_array):
    """Create a mock AGraph with the given simplified command array."""
    ag = MagicMock()
    ag._simplified_command_array = np.array(simplified_command_array, dtype=int)
    ag._update = MagicMock()
    return ag


# Bingo operator IDs (subset used in tests)
VARIABLE = 0
CONSTANT = 1
ADDITION = 2
SUBTRACTION = 3
MULTIPLICATION = 4


class TestGetOperatorCounts:
    """Tests for the _get_operator_counts helper."""

    def _patch_bingo_maps(self, mocker):
        """Patch bingo operator maps for controlled testing."""
        # These match bingo's real definitions for ops 0-4
        is_terminal = {0: True, 1: True, 2: False, 3: False, 4: False}
        is_arity_2 = {0: False, 1: False, 2: True, 3: True, 4: True}
        mocker.patch("pysips.priors.bms_prior.IS_TERMINAL_MAP", is_terminal)
        mocker.patch("pysips.priors.bms_prior.IS_ARITY_2_MAP", is_arity_2)
        mocker.patch("pysips.priors.bms_prior.VARIABLE", 0)

    def test_tree_traversal_counts_repeated_subgraphs(self, mocker):
        """Tree traversal should count shared nodes multiple times."""
        self._patch_bingo_maps(mocker)

        # Expression: (X_0 + X_0)  ->  X_0 is row 0, + is row 1
        # command_array: [[VARIABLE, 0, 0], [ADDITION, 0, 0]]
        # Tree: + -> X_0 (left), X_0 (right) => VARIABLE counted twice
        ag = _make_mock_agraph(
            [
                [VARIABLE, 0, 0],  # row 0: X_0
                [ADDITION, 0, 0],  # row 1: row0 + row0
            ]
        )

        counts = _get_operator_counts(ag, tree=True, terminals="include")
        assert counts[VARIABLE] == 2
        assert counts[ADDITION] == 1

    def test_dag_traversal_counts_unique_nodes(self, mocker):
        """DAG (non-tree) traversal counts each row once."""
        self._patch_bingo_maps(mocker)

        ag = _make_mock_agraph(
            [
                [VARIABLE, 0, 0],
                [ADDITION, 0, 0],
            ]
        )

        counts = _get_operator_counts(ag, tree=False, terminals="include")
        assert counts[VARIABLE] == 1
        assert counts[ADDITION] == 1

    def test_terminals_exclude(self, mocker):
        """terminals='exclude' should omit terminal nodes."""
        self._patch_bingo_maps(mocker)

        ag = _make_mock_agraph(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ]
        )

        counts = _get_operator_counts(ag, tree=True, terminals="exclude")
        assert VARIABLE not in counts
        assert CONSTANT not in counts
        assert counts[ADDITION] == 1

    def test_terminals_combine(self, mocker):
        """terminals='combine' should lump all terminals under VARIABLE."""
        self._patch_bingo_maps(mocker)

        ag = _make_mock_agraph(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ]
        )

        counts = _get_operator_counts(ag, tree=True, terminals="combine")
        assert counts[VARIABLE] == 2  # both X_0 and Const -> VARIABLE
        assert CONSTANT not in counts
        assert counts[ADDITION] == 1


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

    def test_logpdf_returns_correct_shape(self, mocker):
        """logpdf should return shape (N, 1)."""
        mocker.patch(
            f"{PRIORMODULE}._get_operator_counts",
            return_value={2: 1},
        )

        prior = BMSPrior({2: 0.5}, {2: 0.1})
        mock_ags = [MagicMock() for _ in range(5)]

        result = prior.logpdf(mock_ags)
        assert result.shape == (5, 1)

    def test_logpdf_single_element(self, mocker):
        """logpdf should handle a single-element array."""
        mocker.patch(
            f"{PRIORMODULE}._get_operator_counts",
            return_value={2: 1},
        )

        prior = BMSPrior({2: 0.5}, {2: 0.1})
        result = prior.logpdf([MagicMock()])
        assert result.shape == (1, 1)

    def test_logpdf_2d_input(self, mocker):
        """logpdf should handle (N, 1) shaped input."""
        mocker.patch(
            f"{PRIORMODULE}._get_operator_counts",
            return_value={2: 2},
        )

        prior = BMSPrior({2: 0.5}, {2: 0.1})
        mock_ags = [MagicMock(), MagicMock(), MagicMock()]

        result = prior.logpdf(mock_ags)
        assert result.shape == (3, 1)

    def test_logpdf_computes_correct_value(self, mocker):
        """Verify the energy computation: -sum(n*w + n^2*w2)."""
        # Expression has 2 additions and 1 multiplication
        mocker.patch(
            f"{PRIORMODULE}._get_operator_counts",
            return_value={2: 2, 4: 1},
        )

        weights = {2: 0.5, 4: 0.3}
        sq_weights = {2: 0.1, 4: 0.05}
        prior = BMSPrior(weights, sq_weights)

        # Energy = 2*0.5 + 4*0.1 + 1*0.3 + 1*0.05 = 1.0 + 0.4 + 0.3 + 0.05 = 1.75
        # logpdf = -1.75
        result = prior.logpdf([MagicMock()])
        np.testing.assert_almost_equal(result[0, 0], -1.75)

    def test_logpdf_zero_weights_give_zero_logprob(self, mocker):
        """Zero weights should yield log-probability of 0."""
        mocker.patch(
            f"{PRIORMODULE}._get_operator_counts",
            return_value={2: 3, 4: 2},
        )

        prior = BMSPrior({}, {})  # all default to 0.0
        result = prior.logpdf([MagicMock()])
        np.testing.assert_almost_equal(result[0, 0], 0.0)

    def test_logpdf_multiple_expressions_independent(self, mocker):
        """Each expression should be scored independently."""
        counts_sequence = [{2: 1}, {2: 3}]
        mocker.patch(
            f"{PRIORMODULE}._get_operator_counts",
            side_effect=counts_sequence,
        )

        weights = {2: 1.0}
        sq_weights = {2: 0.0}
        prior = BMSPrior(weights, sq_weights)

        result = prior.logpdf([MagicMock(), MagicMock()])

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

    def test_logpdf_works_without_x_dim(self, mocker):
        """logpdf should work even if x_dim is not set."""
        mocker.patch(
            f"{PRIORMODULE}._get_operator_counts",
            return_value={2: 1},
        )

        # Create prior without x_dim (backward compatible)
        prior = BMSPrior({2: 0.5}, {2: 0.1})

        result = prior.logpdf([MagicMock()])
        assert result.shape == (1, 1)

    def test_init_with_minimal_args(self):
        """BMSPrior should accept just weights and squared_weights."""
        prior = BMSPrior({2: 0.5}, {2: 0.1})

        assert prior.weights == {2: 0.5}
        assert prior.squared_weights == {2: 0.1}
        assert prior.x_dim is None
        assert prior.operators == [2]
