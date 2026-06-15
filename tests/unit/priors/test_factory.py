"""Tests for internal Prior Resolution in pysips.priors.factory."""

from unittest.mock import MagicMock

import pytest

from pysips.bingo_construction import BingoConstructionConfig
from pysips.priors.factory import build_prior


FACTORY_MODULE = "pysips.priors.factory"


def _bingo_config(**overrides):
    return BingoConstructionConfig(**overrides)


def test_build_prior_uniform_uses_shared_generator(mocker):
    mock_generator = MagicMock()
    mock_uniform_prior = MagicMock()
    mocker.patch(
        f"{FACTORY_MODULE}.build_agraph_generator", return_value=mock_generator
    )
    mock_improper = mocker.patch(
        f"{FACTORY_MODULE}.ImproperUniformPrior", return_value=mock_uniform_prior
    )

    prior = build_prior(
        "uniform",
        None,
        operators=["+", "*"],
        x_dim=2,
        bingo_config=_bingo_config(),
    )

    assert prior is mock_uniform_prior
    mock_improper.assert_called_once_with(mock_generator)


def test_build_prior_bms_uses_benchmark_default(mocker):
    mocker.patch(
        f"{FACTORY_MODULE}.load_bms_weights", return_value=({3: 1.0}, {3: 0.1})
    )
    mock_bms_prior = MagicMock()
    mock_bms_cls = mocker.patch(
        f"{FACTORY_MODULE}.BMSPrior", return_value=mock_bms_prior
    )

    prior = build_prior(
        "bms",
        None,
        operators=["+", "-"],
        x_dim=3,
        bingo_config=_bingo_config(max_complexity=30),
        num_mcmc_samples=7,
        target_ess=0.5,
        max_equation_evals=12,
        random_state=99,
    )

    assert prior is mock_bms_prior
    mock_bms_cls.assert_called_once()
    assert mock_bms_cls.call_args.args == ({3: 1.0}, {3: 0.1})
    assert mock_bms_cls.call_args.kwargs["operators"] == ["+", "-"]
    assert mock_bms_cls.call_args.kwargs["x_dim"] == 3
    assert mock_bms_cls.call_args.kwargs["max_complexity"] == 30
    assert mock_bms_cls.call_args.kwargs["num_mcmc_samples"] == 7
    assert mock_bms_cls.call_args.kwargs["target_ess"] == 0.5
    assert mock_bms_cls.call_args.kwargs["max_equation_evals"] == 12
    assert mock_bms_cls.call_args.kwargs["random_state"] == 99


def test_build_prior_plain_katz_defaults_to_strict_prebuilt(mocker):
    mock_model = MagicMock()
    mock_load = mocker.patch(f"{FACTORY_MODULE}.load_katz_model", return_value=mock_model)
    mock_katz_prior = MagicMock()
    mock_katz_cls = mocker.patch(
        f"{FACTORY_MODULE}.KatzPrior", return_value=mock_katz_prior
    )

    prior = build_prior(
        "katz",
        None,
        operators=["+", "*"],
        x_dim=4,
        bingo_config=_bingo_config(),
    )

    assert prior is mock_katz_prior
    mock_load.assert_called_once_with(n=2, corpus="benchmark", fit_if_missing=False)
    mock_katz_cls.assert_called_once()


def test_build_prior_plain_katz_allows_fit_if_missing_opt_in(mocker):
    mock_model = MagicMock()
    mock_load = mocker.patch(f"{FACTORY_MODULE}.load_katz_model", return_value=mock_model)
    mocker.patch(f"{FACTORY_MODULE}.KatzPrior", return_value=MagicMock())

    build_prior(
        "katz",
        {"fit_if_missing": True, "n": 3, "corpus": "benchmark"},
        operators=["+", "*"],
        x_dim=4,
        bingo_config=_bingo_config(),
    )

    mock_load.assert_called_once_with(n=3, corpus="benchmark", fit_if_missing=True)


def test_build_prior_size_calibrated_uniform_uses_canonical_ids(mocker):
    mock_hist = {3: -1.0}
    mock_z_k = {3: 2.0}
    mocker.patch(f"{FACTORY_MODULE}.load_corpus_histogram", return_value=mock_hist)
    mock_load_z_k = mocker.patch(f"{FACTORY_MODULE}.load_z_k", return_value=mock_z_k)
    mock_size_calibrated = MagicMock()
    mock_size_cls = mocker.patch(
        f"{FACTORY_MODULE}.SizeCalibratedPrior", return_value=mock_size_calibrated
    )

    prior = build_prior(
        "size_calibrated_uniform",
        {"floor_log_prob": -100.0},
        operators=["*", "+"],
        x_dim=3,
        bingo_config=_bingo_config(),
        num_mcmc_samples=9,
        target_ess=0.4,
        max_time=5.0,
        random_state=123,
    )

    assert prior is mock_size_calibrated
    mock_load_z_k.assert_called_once_with("uniform", [3, 5], 3, corpus="benchmark")
    assert mock_size_cls.call_args.kwargs["base_prior"] is None
    assert mock_size_cls.call_args.kwargs["floor_log_prob"] == -100.0
    assert mock_size_cls.call_args.kwargs["num_mcmc_samples"] == 9
    assert mock_size_cls.call_args.kwargs["target_ess"] == 0.4
    assert mock_size_cls.call_args.kwargs["max_time"] == 5.0
    assert mock_size_cls.call_args.kwargs["random_state"] == 123


def test_build_prior_size_calibrated_katz_forces_strict_prebuilt(mocker):
    mocker.patch(f"{FACTORY_MODULE}.load_corpus_histogram", return_value={3: -1.0})
    mocker.patch(f"{FACTORY_MODULE}.load_z_k", return_value={3: 2.0})
    mock_model = MagicMock()
    mock_load = mocker.patch(f"{FACTORY_MODULE}.load_katz_model", return_value=mock_model)
    mocker.patch(f"{FACTORY_MODULE}.KatzPrior", return_value=MagicMock())
    mocker.patch(f"{FACTORY_MODULE}.SizeCalibratedPrior", return_value=MagicMock())

    build_prior(
        "size_calibrated_katz",
        {"n": 2, "corpus": "benchmark"},
        operators=["+", "*"],
        x_dim=3,
        bingo_config=_bingo_config(),
    )

    mock_load.assert_called_once_with(n=2, corpus="benchmark", fit_if_missing=False)


def test_build_prior_size_calibrated_bms_uses_composition_path(mocker):
    mocker.patch(f"{FACTORY_MODULE}.load_corpus_histogram", return_value={3: -1.0})
    mock_load_z_k = mocker.patch(f"{FACTORY_MODULE}.load_z_k", return_value={3: 2.0})
    mocker.patch(
        f"{FACTORY_MODULE}.load_bms_weights", return_value=({3: 1.0}, {3: 0.1})
    )
    mock_bms_prior = MagicMock()
    mocker.patch(f"{FACTORY_MODULE}.BMSPrior", return_value=mock_bms_prior)
    mock_size_calibrated = MagicMock()
    mock_size_cls = mocker.patch(
        f"{FACTORY_MODULE}.SizeCalibratedPrior", return_value=mock_size_calibrated
    )

    prior = build_prior(
        "size_calibrated_bms",
        None,
        operators=["sin", "cos", "/", "*", "-", "+"],
        x_dim=3,
        bingo_config=_bingo_config(),
    )

    assert prior is mock_size_calibrated
    mock_load_z_k.assert_called_once_with(
        "bms", [3, 4, 5, 6, 15, 16], 3, corpus="benchmark"
    )
    assert mock_size_cls.call_args.kwargs["base_prior"] is mock_bms_prior


def test_build_prior_custom_prior_passthrough():
    custom_prior = MagicMock()
    custom_prior.rvs = MagicMock()
    custom_prior.logpdf = MagicMock()

    prior = build_prior(
        custom_prior,
        None,
        operators=["+", "*"],
        x_dim=2,
        bingo_config=_bingo_config(),
    )

    assert prior is custom_prior


@pytest.mark.parametrize(
    ("prior", "params", "message"),
    [
        ("invalid", None, "Unknown prior 'invalid'"),
        ("bms", {"n": 2}, "Unsupported prior_params"),
        ("katz", {"fit_if_missing": "yes"}, "fit_if_missing must be a bool"),
        (
            "size_calibrated_katz",
            {"fit_if_missing": True},
            "Unsupported prior_params",
        ),
        (
            "size_calibrated_uniform",
            {"floor_log_prob": float("inf")},
            "floor_log_prob must be a finite number or -inf",
        ),
    ],
)
def test_build_prior_rejects_invalid_configs(prior, params, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_prior(
            prior,
            params,
            operators=["+", "*"],
            x_dim=2,
            bingo_config=_bingo_config(),
        )


def test_build_prior_rejects_invalid_custom_prior():
    bad_prior = MagicMock(spec=[])
    bad_prior.logpdf = MagicMock()

    with pytest.raises(TypeError, match="Custom prior must have 'rvs' and 'logpdf' methods"):
        build_prior(
            bad_prior,
            None,
            operators=["+", "*"],
            x_dim=2,
            bingo_config=_bingo_config(),
        )
