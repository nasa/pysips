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
        """Fixture to provide a mock bingo AGraph model."""
        model = mocker.MagicMock()
        model.get_local_optimization_params.return_value = np.array([1.0, 2.0])
        return model

    def test_negative_is_applied_to_regression_output(
        self, sample_data, mock_model, mocker
    ):
        X, y = sample_data

        mock_regression = mocker.MagicMock(return_value=-5.0)
        mocker.patch(f"{IMPORTMODULE}.ExplicitRegression", return_value=mock_regression)

        mock_scipy_optimizer = mocker.MagicMock()
        mocker.patch(
            f"{IMPORTMODULE}.ScipyOptimizer",
            return_value=mock_scipy_optimizer,
        )

        laplace_nmll = LaplaceNmll(X, y)
        result = laplace_nmll(mock_model)

        assert result == 5.0

    @pytest.mark.parametrize("opt_restarts", [1, 3, 5, 10])
    def test_number_of_restarts(self, sample_data, mock_model, mocker, opt_restarts):
        """Test that the optimizer is run the correct number of times based on opt_restarts."""
        X, y = sample_data

        mock_regression = mocker.MagicMock()
        mock_regression.return_value = -1.0  # Constant return value
        mocker.patch(f"{IMPORTMODULE}.ExplicitRegression", return_value=mock_regression)

        mock_scipy_optimizer = mocker.MagicMock()
        mocker.patch(
            f"{IMPORTMODULE}.ScipyOptimizer",
            return_value=mock_scipy_optimizer,
        )

        laplace_nmll = LaplaceNmll(X, y, opt_restarts=opt_restarts)
        laplace_nmll(mock_model)
        assert mock_scipy_optimizer.call_count == opt_restarts

    def test_constants_kept_from_best_trial(self, sample_data, mock_model, mocker):
        """Test that constants are kept from optimizer trial with highest nmll."""
        X, y = sample_data

        # Return increasingly better values (-3 > -5 > -10 when negated)
        mock_regression = mocker.MagicMock()
        mock_regression.side_effect = [10.0, 5.0, 3.0]
        mocker.patch(f"{IMPORTMODULE}.ExplicitRegression", return_value=mock_regression)

        # Create different parameter sets for different optimization runs
        mock_scipy_optimizer = mocker.MagicMock()
        params_run1 = np.array([1.0, 1.0])
        params_run2 = np.array([2.0, 2.0])
        params_run3 = np.array([3.0, 3.0])  # This should be kept as the best
        mock_model.get_local_optimization_params.side_effect = [
            params_run1,
            params_run2,
            params_run3,
        ]
        mocker.patch(
            f"{IMPORTMODULE}.ScipyOptimizer",
            return_value=mock_scipy_optimizer,
        )

        laplace_nmll = LaplaceNmll(X, y, opt_restarts=3)
        result = laplace_nmll(mock_model)

        assert result == -3.0
        mock_model.set_local_optimization_params.assert_called_once_with(params_run3)

    def test_optimizer_kwargs_passed_through(self, sample_data, mocker):
        """Test that optimizer kwargs are passed to the bingo deterministic optimizer."""
        X, y = sample_data

        scipy_optimizer_spy = mocker.patch(f"{IMPORTMODULE}.ScipyOptimizer")
        mocker.patch(f"{IMPORTMODULE}.ExplicitRegression")

        custom_kwargs = {
            "param_init_bounds": [-10, 10],
            "tol": 1e-8,
            "options": {"maxiter": 500},
        }
        LaplaceNmll(X, y, **custom_kwargs)
        expected_kwargs = {"method": "lm", **custom_kwargs}
        _, actual_kwargs = scipy_optimizer_spy.call_args

        for key, value in expected_kwargs.items():
            assert key in actual_kwargs
            assert actual_kwargs[key] == value
