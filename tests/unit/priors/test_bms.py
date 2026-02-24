import numpy as np
from unittest.mock import MagicMock

from pysips.priors import BMSPrior
from pysips.priors.bms_prior import _get_operator_counts


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
            "pysips.priors.bms_prior._get_operator_counts",
            return_value={2: 1},
        )

        prior = BMSPrior({2: 0.5}, {2: 0.1})
        mock_ags = [MagicMock() for _ in range(5)]

        result = prior.logpdf(mock_ags)
        assert result.shape == (5, 1)

    def test_logpdf_single_element(self, mocker):
        """logpdf should handle a single-element array."""
        mocker.patch(
            "pysips.priors.bms_prior._get_operator_counts",
            return_value={2: 1},
        )

        prior = BMSPrior({2: 0.5}, {2: 0.1})
        result = prior.logpdf([MagicMock()])
        assert result.shape == (1, 1)

    def test_logpdf_2d_input(self, mocker):
        """logpdf should handle (N, 1) shaped input."""
        mocker.patch(
            "pysips.priors.bms_prior._get_operator_counts",
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
            "pysips.priors.bms_prior._get_operator_counts",
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
            "pysips.priors.bms_prior._get_operator_counts",
            return_value={2: 3, 4: 2},
        )

        prior = BMSPrior({}, {})  # all default to 0.0
        result = prior.logpdf([MagicMock()])
        np.testing.assert_almost_equal(result[0, 0], 0.0)

    def test_logpdf_multiple_expressions_independent(self, mocker):
        """Each expression should be scored independently."""
        counts_sequence = [{2: 1}, {2: 3}]
        mocker.patch(
            "pysips.priors.bms_prior._get_operator_counts",
            side_effect=counts_sequence,
        )

        weights = {2: 1.0}
        sq_weights = {2: 0.0}
        prior = BMSPrior(weights, sq_weights)

        result = prior.logpdf([MagicMock(), MagicMock()])

        # First: -(1*1.0) = -1.0, Second: -(3*1.0) = -3.0
        np.testing.assert_almost_equal(result[0, 0], -1.0)
        np.testing.assert_almost_equal(result[1, 0], -3.0)
