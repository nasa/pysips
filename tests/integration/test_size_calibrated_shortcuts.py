"""Integration tests for size-calibrated prior string shortcuts.

Tests that ``PysipsRegressor(prior="size_calibrated_katz")`` and
``prior="size_calibrated_uniform"`` work end-to-end, and that config
mismatches raise ``ValueError``.
"""

import numpy as np
import pytest

from pysips.regressor import PysipsRegressor

STANDARD_OPERATORS = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]


@pytest.fixture
def toy_data():
    """Simple 1-d toy dataset: y = 2*x."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(20, 1))
    y = 2 * X[:, 0]
    return X, y


class TestSizeCalibratedUniform:
    """Tests for prior='size_calibrated_uniform'."""

    def test_end_to_end_fit(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_uniform",
            operators=STANDARD_OPERATORS,
            num_particles=5,
            max_time=5,
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
        with pytest.raises(ValueError, match="Operator mismatch"):
            reg.fit(X, y)

    def test_x_dim_mismatch_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(20, 2))
        y = X[:, 0] + X[:, 1]
        reg = PysipsRegressor(
            prior="size_calibrated_uniform",
            operators=STANDARD_OPERATORS,
            random_state=0,
        )
        with pytest.raises(ValueError, match="x_dim mismatch"):
            reg.fit(X, y)


class TestSizeCalibratedKatz:
    """Tests for prior='size_calibrated_katz'."""

    def test_end_to_end_fit(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_katz",
            operators=STANDARD_OPERATORS,
            num_particles=5,
            max_time=5,
            random_state=0,
            show_progress_bar=False,
        )
        reg.fit(X, y)
        assert reg.best_model_ is not None


class TestSizeCalibratedBMS:
    """Tests for prior='size_calibrated_bms'."""

    def test_raises_not_implemented(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_bms",
            operators=STANDARD_OPERATORS,
            random_state=0,
        )
        with pytest.raises(NotImplementedError, match="fit_size_calibrated_prior"):
            reg.fit(X, y)


class TestFloorLogProbOverride:
    """Tests that floor_log_prob is overridable via prior_params."""

    def test_floor_log_prob_passed_through(self, toy_data):
        X, y = toy_data
        reg = PysipsRegressor(
            prior="size_calibrated_uniform",
            operators=STANDARD_OPERATORS,
            prior_params={"floor_log_prob": -100.0},
            num_particles=5,
            max_time=5,
            random_state=0,
            show_progress_bar=False,
        )
        reg.fit(X, y)
        assert reg.best_model_ is not None
