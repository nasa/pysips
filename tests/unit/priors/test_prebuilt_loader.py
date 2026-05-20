"""Tests for prebuilt_loader module."""

import pytest

from pysips.priors.prebuilt_loader import (
    _resolve_operator_ids,
    load_corpus_histogram,
    load_prebuilt_size_calibrated,
    load_z_k,
)

STANDARD_OPS_STR = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
STANDARD_OPS_INT = [3, 4, 5, 6, 13, 14, 15, 16]


class TestResolveOperatorIds:
    def test_string_operators(self):
        result = _resolve_operator_ids(["+", "-", "*", "/"])
        assert result == sorted([3, 4, 5, 6])

    def test_int_operators(self):
        result = _resolve_operator_ids([3, 4, 5, 6])
        assert result == [3, 4, 5, 6]

    def test_mixed(self):
        result = _resolve_operator_ids(["+", 4, "*", 6])
        assert result == sorted([3, 4, 5, 6])


class TestLoadCorpusHistogram:
    def test_loads_empirical(self):
        hist = load_corpus_histogram(STANDARD_OPS_STR, 1, "empirical")
        assert isinstance(hist, dict)
        assert all(isinstance(k, int) for k in hist)
        assert all(isinstance(v, float) for v in hist.values())
        assert len(hist) > 0

    def test_loads_parametric(self):
        hist = load_corpus_histogram(STANDARD_OPS_STR, 1, "parametric")
        assert isinstance(hist, dict)
        assert len(hist) > 0

    def test_operator_mismatch(self):
        with pytest.raises(ValueError, match="Operator mismatch"):
            load_corpus_histogram(["+", "*"], 1)

    def test_x_dim_mismatch(self):
        with pytest.raises(ValueError, match="x_dim mismatch"):
            load_corpus_histogram(STANDARD_OPS_STR, 2)


class TestLoadZK:
    def test_loads_uniform(self):
        z_k = load_z_k("uniform", STANDARD_OPS_STR, 1)
        assert isinstance(z_k, dict)
        assert all(isinstance(k, int) for k in z_k)
        assert len(z_k) > 0

    def test_loads_katz(self):
        z_k = load_z_k("katz", STANDARD_OPS_STR, 1)
        assert isinstance(z_k, dict)
        assert len(z_k) > 0

    def test_unknown_base_prior(self):
        with pytest.raises(KeyError, match="No pre-built Z_k"):
            load_z_k("unknown", STANDARD_OPS_STR, 1)


class TestLoadPrebuiltSizeCalibrated:
    def test_loads_both(self):
        hist, z_k = load_prebuilt_size_calibrated("uniform", STANDARD_OPS_STR, 1)
        assert isinstance(hist, dict)
        assert isinstance(z_k, dict)
        assert len(hist) > 0
        assert len(z_k) > 0
