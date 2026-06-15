"""Tests for prebuilt_loader module."""

import pytest

from pysips.priors.prebuilt_loader import (
    _operator_filename_tag,
    _prebuilt_filename,
    katz_model_path,
    load_corpus_histogram,
    load_prebuilt_size_calibrated,
    load_z_k,
)

STANDARD_OPS_INT = [3, 4, 5, 6]
EXTENDED_OPS_INT = [3, 4, 5, 6, 15, 16]


class TestKatzModelPath:
    def test_uses_benchmark_default(self):
        assert katz_model_path(2).name == "default_katz_n2_benchmark.json"


class TestPrebuiltFileNaming:
    def test_operator_filename_tag_is_operator_specific(self):
        assert _operator_filename_tag(STANDARD_OPS_INT) == "ops3-4-5-6"

    def test_z_k_filename_uses_operator_ids(self):
        assert (
            _prebuilt_filename(
                "z_k",
                "benchmark",
                3,
                STANDARD_OPS_INT,
                base_prior_key="katz",
            )
            == "z_k_katz_benchmark_x3_ops3-4-5-6.json"
        )

    def test_histogram_filename_is_corpus_only(self):
        # Corpus histograms describe the corpus's size distribution and
        # do not depend on x_dim or the operator set.
        assert _prebuilt_filename("histogram", "benchmark") == (
            "corpus_histogram_benchmark.json"
        )

    def test_z_k_requires_x_dim_and_operators(self):
        with pytest.raises(ValueError, match="x_dim and operator_ids"):
            _prebuilt_filename("z_k", "benchmark", base_prior_key="katz")

    def test_z_k_requires_base_prior_key(self):
        with pytest.raises(ValueError, match="base_prior_key"):
            _prebuilt_filename("z_k", "benchmark", 1, STANDARD_OPS_INT)


class TestLoadCorpusHistogram:
    def test_loads_empirical(self):
        hist = load_corpus_histogram("empirical")
        assert isinstance(hist, dict)
        assert all(isinstance(k, int) for k in hist)
        assert all(isinstance(v, float) for v in hist.values())
        assert len(hist) > 0

    def test_loads_parametric(self):
        hist = load_corpus_histogram("parametric")
        assert isinstance(hist, dict)
        assert len(hist) > 0

    def test_unknown_corpus_raises(self):
        with pytest.raises(FileNotFoundError):
            load_corpus_histogram(corpus="does_not_exist")


class TestLoadZK:
    def test_loads_uniform(self):
        z_k = load_z_k("uniform", STANDARD_OPS_INT, 3)
        assert isinstance(z_k, dict)
        assert all(isinstance(k, int) for k in z_k)
        assert len(z_k) > 0

    def test_loads_katz(self):
        z_k = load_z_k("katz", STANDARD_OPS_INT, 3)
        assert isinstance(z_k, dict)
        assert len(z_k) > 0

    def test_loads_bms(self):
        z_k = load_z_k("bms", EXTENDED_OPS_INT, 3)
        assert isinstance(z_k, dict)
        assert len(z_k) > 0

    def test_unknown_base_prior(self):
        with pytest.raises(KeyError, match="No pre-built Z_k"):
            load_z_k("unknown", STANDARD_OPS_INT, 3)

    def test_non_canonical_operator_ids_rejected(self):
        with pytest.raises(TypeError, match="canonical_operator_ids"):
            load_z_k("katz", ["+", "*"], 3)

    def test_operator_mismatch(self):
        with pytest.raises(FileNotFoundError):
            # No prebuilt Z_k for this operator subset.
            load_z_k("katz", [3, 5], 3)

    def test_x_dim_mismatch(self):
        with pytest.raises(FileNotFoundError):
            # No prebuilt Z_k for x_dim=2 with the standard operator set.
            load_z_k("katz", STANDARD_OPS_INT, 2)


class TestLoadPrebuiltSizeCalibrated:
    def test_loads_both(self):
        hist, z_k = load_prebuilt_size_calibrated("uniform", STANDARD_OPS_INT, 3)
        assert isinstance(hist, dict)
        assert isinstance(z_k, dict)
        assert len(hist) > 0
        assert len(z_k) > 0
