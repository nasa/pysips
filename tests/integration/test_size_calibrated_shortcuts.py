"""Integration tests for size-calibrated prior string shortcuts.

Tests that the built-in size-calibrated prior strings resolve end-to-end,
and that unsupported artifact combinations fail cleanly.
"""

import numpy as np
import pytest

from pysips.regressor import PysipsRegressor

PREBUILT_OPERATORS = ["+", "-", "*", "/", "sin", "cos"]
SHORTCUT_TEST_MAX_EQUATION_EVALS = 25


@pytest.fixture
def toy_data():
    """Simple 3-d toy dataset matching shipped size-calibration artifacts."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(20, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
    return X, y


class TestSizeCalibratedUniform:
    """Tests for prior='size_calibrated_uniform'."""

    def test_end_to_end_fit(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_uniform",
            operators=PREBUILT_OPERATORS,
            num_particles=5,
            max_equation_evals=SHORTCUT_TEST_MAX_EQUATION_EVALS,
            random_state=0,
            show_progress_bar=False,
        )
        reg.fit(X, y)
        assert reg.best_model_ is not None

    def test_operator_mismatch_raises(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_uniform",
            operators=["+", "*"],
            random_state=0,
        )
        with pytest.raises(FileNotFoundError, match="No pre-built z_k data"):
            reg.fit(X, y)

    def test_x_dim_mismatch_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(20, 2))
        y = X[:, 0] + X[:, 1]
        reg = PysipsRegressor(
            prior="size_calibrated_uniform",
            operators=PREBUILT_OPERATORS,
            random_state=0,
        )
        with pytest.raises(FileNotFoundError, match="No pre-built z_k data"):
            reg.fit(X, y)


class TestSizeCalibratedKatz:
    """Tests for prior='size_calibrated_katz'."""

    def test_end_to_end_fit(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_katz",
            operators=PREBUILT_OPERATORS,
            num_particles=5,
            max_equation_evals=SHORTCUT_TEST_MAX_EQUATION_EVALS,
            random_state=0,
            show_progress_bar=False,
        )
        reg.fit(X, y)
        assert reg.best_model_ is not None


class TestSizeCalibratedBMS:
    """Tests for prior='size_calibrated_bms'."""

    def test_end_to_end_fit(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_bms",
            operators=PREBUILT_OPERATORS,
            num_particles=5,
            max_equation_evals=SHORTCUT_TEST_MAX_EQUATION_EVALS,
            random_state=0,
            show_progress_bar=False,
        )
        reg.fit(X, y)
        assert reg.best_model_ is not None


class TestFloorLogProbOverride:
    """Tests that floor_log_prob is overridable via prior_params."""

    def test_floor_log_prob_passed_through(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_uniform",
            operators=PREBUILT_OPERATORS,
            prior_params={"floor_log_prob": -100.0},
            num_particles=5,
            max_equation_evals=SHORTCUT_TEST_MAX_EQUATION_EVALS,
            random_state=0,
            show_progress_bar=False,
        )
        reg.fit(X, y)
        assert reg.best_model_ is not None
