import pytest
import numpy as np

from pysips.laplace_nmll import LaplaceNmll

IMPORTMODULE = LaplaceNmll.__module__


class TestLaplaceNmll:

    @pytest.fixture
    def sample_data(self):
        """Fixture to provide sample data for tests."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])
        return X, y

    @pytest.fixture
    def mock_model(self, mocker):
        """Fixture to provide a mock EvolvableExpression model."""
        model = mocker.MagicMock()
        expr = mocker.MagicMock()
        expr.constants = (1.0, 2.0)
        model.expression = expr
        return model

    def test_score_returned(self, sample_data, mock_model):
        """Test that the score from expression.score is returned."""
        X, y = sample_data
        mock_model.expression.score.return_value = 5.0

        laplace_nmll = LaplaceNmll(X, y)
        result = laplace_nmll(mock_model)

        assert result == 5.0
        mock_model.expression.fit.assert_called_once()
        mock_model.expression.score.assert_called_once()

    @pytest.mark.parametrize("opt_restarts", [1, 3, 5, 10])
    def test_number_of_restarts(self, sample_data, mock_model, opt_restarts):
        """Test that fit/score is called the correct number of times."""
        X, y = sample_data
        mock_model.expression.score.return_value = -1.0

        laplace_nmll = LaplaceNmll(X, y, opt_restarts=opt_restarts)
        laplace_nmll(mock_model)
        assert mock_model.expression.fit.call_count == opt_restarts
        assert mock_model.expression.score.call_count == opt_restarts

    def test_constants_kept_from_best_trial(self, sample_data):
        """Test that constants are kept from the trial with highest nmll."""
        X, y = sample_data

        from unittest.mock import MagicMock

        model = MagicMock()

        # Use a simple helper class to track constants state
        class FakeExpr:
            def __init__(self):
                self.constants = (1.0, 1.0)
                self._fit_count = 0

            def fit(self, X, y):
                self._fit_count += 1
                # Simulate optimizer changing constants after each fit
                self.constants = (float(self._fit_count),) * 2

            def score(self, X, y, metric=None):
                # Return scores: 10.0, 5.0, 30.0 — 3rd is best
                return [10.0, 5.0, 30.0][self._fit_count - 1]

        fake_expr = FakeExpr()
        model.expression = fake_expr

        laplace_nmll = LaplaceNmll(X, y, opt_restarts=3)
        result = laplace_nmll(model)

        assert result == 30.0
        # Best constants should be from the 3rd fit
        assert fake_expr.constants == (3.0, 3.0)

    def test_param_init_bounds(self, sample_data, mocker):
        """Test that custom param_init_bounds are stored correctly."""
        X, y = sample_data

        laplace_nmll = LaplaceNmll(X, y, param_init_bounds=[-10, 10])
        assert laplace_nmll._bounds == [-10, 10]

    def test_default_param_init_bounds(self, sample_data):
        """Test that default param_init_bounds are [-5, 5]."""
        X, y = sample_data

        laplace_nmll = LaplaceNmll(X, y)
        assert laplace_nmll._bounds == [-5, 5]
