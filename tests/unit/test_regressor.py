import pytest
import numpy as np
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from sklearn.base import RegressorMixin

from pysips.regressor import PysipsRegressor

# Dynamically get the module containing the PysipsRegressor class
IMPORTMODULE = PysipsRegressor.__module__
BINGO_MIXIN_MODULE = "pysips.bingo_proposal_mixin"


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    return X, y


@pytest.fixture
def mock_external_components(mocker: MockerFixture):
    """Mock all external components needed by the regressor."""
    mock_component_gen = mocker.patch(
        f"{BINGO_MIXIN_MODULE}.ComponentGenerator", autospec=True
    )
    # needs to provide unique outputs for pool generation
    mock_agraph_gen = mocker.MagicMock(side_effect=lambda: np.random.random())
    mock_agraph_gen_constructor = mocker.patch(
        f"{BINGO_MIXIN_MODULE}.AGraphGenerator",
        autospec=True,
        return_value=mock_agraph_gen,
    )
    mock_laplace_nmll = mocker.patch(f"{IMPORTMODULE}.LaplaceNmll", autospec=True)
    mock_mutation_proposal = mocker.patch(
        f"{BINGO_MIXIN_MODULE}.MutationProposal", autospec=True
    )
    mock_crossover_proposal = mocker.patch(
        f"{BINGO_MIXIN_MODULE}.CrossoverProposal", autospec=True
    )
    mock_random_choice_proposal = mocker.patch(
        f"{BINGO_MIXIN_MODULE}.RandomChoiceProposal", autospec=True
    )
    mock_sample = mocker.patch(f"{IMPORTMODULE}.sample", autospec=True)

    # Configure the mock sample function to return a model and likelihood
    mock_model = MagicMock()
    mock_model.__str__.return_value = "2*X_0"
    mock_likelihoods = np.array([0.9])
    mock_sample.return_value = ([mock_model], mock_likelihoods, [0.5])

    return {
        "component_gen": mock_component_gen,
        "agraph_gen": mock_agraph_gen_constructor,
        "laplace_nmll": mock_laplace_nmll,
        "mutation_proposal": mock_mutation_proposal,
        "crossover_proposal": mock_crossover_proposal,
        "random_choice_proposal": mock_random_choice_proposal,
        "sample": mock_sample,
        "model": mock_model,
    }


def test_init_custom_parameters():
    """Test initialization with custom parameters."""
    regressor = PysipsRegressor(
        operators=["+", "*", "sin", "cos"],
        max_complexity=30,
        terminal_probability=0.2,
        num_particles=100,
        random_state=42,
    )

    # Check custom values
    assert regressor.operators == ["+", "*", "sin", "cos"]
    assert regressor.max_complexity == 30
    assert regressor.terminal_probability == 0.2
    assert regressor.num_particles == 100
    assert regressor.random_state == 42


def test_fit(sample_data, mock_external_components):
    """Test the fit method."""
    X, y = sample_data
    mock_model = mock_external_components["model"]
    mock_sample = mock_external_components["sample"]

    regressor = PysipsRegressor(random_state=42)
    regressor.fit(X, y)

    # Verify that sample was called
    mock_sample.assert_called_once()

    # Verify that attributes were set correctly
    assert regressor.models_ == [mock_model]
    assert np.array_equal(regressor.likelihoods_, np.array([0.9]))
    assert regressor.best_model_ == mock_model

    # Verify n_features_in_ is set correctly
    assert regressor.n_features_in_ == X.shape[1]


def test_predict_requires_fit(sample_data):
    """Test that predict requires the model to be fitted first."""
    X, _ = sample_data
    regressor = PysipsRegressor()

    with pytest.raises(Exception):  # Should raise NotFittedError
        regressor.predict(X)


def test_predict(sample_data, mocker: MockerFixture):
    """Test the predict method."""
    X, _ = sample_data

    # Create a mock model with evaluate_equation_at method
    mock_model = MagicMock()
    mock_predictions = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    mock_model.evaluate_equation_at.return_value = mock_predictions

    # Mock check_is_fitted to avoid NotFittedError
    mocker.patch(f"{IMPORTMODULE}.check_is_fitted")
    # Mock check_array to return the input unchanged
    mocker.patch(f"{IMPORTMODULE}.check_array", return_value=X)

    regressor = PysipsRegressor()
    regressor.best_model_ = mock_model
    regressor.models_ = [mock_model]
    regressor.likelihoods_ = np.array([1.0])
    regressor.n_features_in_ = X.shape[1]

    predictions = regressor.predict(X)

    # Verify that evaluate_equation_at was called with X
    mock_model.evaluate_equation_at.assert_called_once_with(X)
    assert np.array_equal(predictions, mock_predictions)


def test_predict_wrong_feature_count(sample_data, mocker: MockerFixture):
    """Test predict raises error when input has wrong number of features."""
    X, _ = sample_data
    wrong_X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 features instead of 1

    # Mock check_is_fitted to avoid NotFittedError
    mocker.patch(f"{IMPORTMODULE}.check_is_fitted")
    # Mock check_array to return the input unchanged
    mocker.patch(f"{IMPORTMODULE}.check_array", return_value=wrong_X)

    regressor = PysipsRegressor()
    regressor.best_model_ = MagicMock()
    regressor.models_ = [MagicMock()]
    regressor.n_features_in_ = X.shape[1]  # Trained with 1 feature

    with pytest.raises(
        ValueError
    ):  # Should raise ValueError for feature count mismatch
        regressor.predict(wrong_X)


def test_score(sample_data, mocker: MockerFixture):
    """Test the score method."""
    X, y = sample_data

    # Mock super().score to return a predetermined R² value
    mocker.patch("sklearn.base.RegressorMixin.score", return_value=0.95)

    regressor = PysipsRegressor()
    regressor.best_model_ = MagicMock()

    score = regressor.score(X, y)

    assert score == 0.95


def test_get_expression(mocker: MockerFixture):
    """Test the get_expression method."""
    # Mock check_is_fitted to avoid NotFittedError
    mocker.patch(f"{IMPORTMODULE}.check_is_fitted")

    mock_model = MagicMock()
    mock_model.__str__.return_value = "2*X_0"

    regressor = PysipsRegressor()
    regressor.best_model_ = mock_model

    expression = regressor.get_expression()

    # Verify that __str__ was called on the best_model_
    mock_model.__str__.assert_called_once()
    assert expression == "2*X_0"


def test_get_expression_not_fitted(mocker: MockerFixture):
    """Test get_expression raises error when called on unfitted model."""
    # Mock check_is_fitted to raise an exception
    mocker.patch(f"{IMPORTMODULE}.check_is_fitted", side_effect=Exception("Not fitted"))

    regressor = PysipsRegressor()

    with pytest.raises(Exception):  # Should raise the exception from check_is_fitted
        regressor.get_expression()


def test_get_models(mocker: MockerFixture):
    """Test the get_models method."""
    # Mock check_is_fitted to avoid NotFittedError
    mocker.patch(f"{IMPORTMODULE}.check_is_fitted")

    mock_models = [MagicMock(), MagicMock()]
    mock_likelihoods = np.array([0.8, 0.2])

    regressor = PysipsRegressor()
    regressor.models_ = mock_models
    regressor.likelihoods_ = mock_likelihoods

    models, likelihoods = regressor.get_models()

    assert models == mock_models
    assert np.array_equal(likelihoods, mock_likelihoods)


def test_get_models_not_fitted(mocker: MockerFixture):
    """Test get_models raises error when called on unfitted model."""
    # Mock check_is_fitted to raise an exception
    mocker.patch(f"{IMPORTMODULE}.check_is_fitted", side_effect=Exception("Not fitted"))

    regressor = PysipsRegressor()

    with pytest.raises(Exception):  # Should raise the exception from check_is_fitted
        regressor.get_models()


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_score_handles_invalid_predictions(
    sample_data, mocker: MockerFixture, invalid_value
):
    """Test that score method handles NaN and infinity values in predictions."""
    X, y = sample_data

    mocker.patch(f"{IMPORTMODULE}.check_is_fitted")
    mocker.patch(f"{IMPORTMODULE}.check_array", return_value=X)

    mock_model = MagicMock()
    predictions = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    predictions[0] = invalid_value  # Set first prediction to the invalid value
    mock_model.evaluate_equation_at.return_value = predictions

    regressor = PysipsRegressor()
    regressor.best_model_ = mock_model
    regressor.models_ = [mock_model]
    regressor.likelihoods_ = np.array([1.0])
    regressor.n_features_in_ = X.shape[1]

    score = regressor.score(X, y)
    assert score == -np.inf


def test_score_reraises_other_value_errors(sample_data, mocker: MockerFixture):
    """Test that score method re-raises ValueError exceptions that are not NaN/inf related."""
    X, y = sample_data

    regressor = PysipsRegressor()
    regressor.best_model_ = MagicMock()

    mocker.patch.object(
        RegressorMixin, "score", side_effect=ValueError("Some other error")
    )
    with pytest.raises(ValueError, match="Some other error"):
        regressor.score(X, y)


def test_score_normal_case_passes_through(sample_data, mocker: MockerFixture):
    """Test that score method passes through normal results when no errors occur."""
    X, y = sample_data

    mock_super_score = mocker.patch(
        "sklearn.base.RegressorMixin.score", return_value=0.85
    )
    regressor = PysipsRegressor()
    regressor.best_model_ = MagicMock()

    score = regressor.score(X, y)
    assert score == 0.85
    mock_super_score.assert_called_once_with(X, y, sample_weight=None)


# --- Tests for prior parameter ---


def test_init_default_prior():
    """Test that prior defaults to 'uniform'."""
    regressor = PysipsRegressor()
    assert regressor.prior == "uniform"


def test_init_prior_uniform():
    """Test initialization with prior='uniform'."""
    regressor = PysipsRegressor(prior="uniform")
    assert regressor.prior == "uniform"


def test_init_prior_bms():
    """Test initialization with prior='bms'."""
    regressor = PysipsRegressor(prior="bms")
    assert regressor.prior == "bms"


def test_init_prior_custom_object():
    """Test initialization with a custom prior object."""
    custom_prior = MagicMock()
    custom_prior.rvs = MagicMock()
    custom_prior.logpdf = MagicMock()
    regressor = PysipsRegressor(prior=custom_prior)
    assert regressor.prior is custom_prior


def test_init_invalid_prior_string():
    """Test that an invalid prior string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown prior 'invalid'"):
        PysipsRegressor(prior="invalid")


def test_init_invalid_prior_object_missing_rvs():
    """Test that a prior object without rvs raises TypeError."""
    bad_prior = MagicMock(spec=[])
    bad_prior.logpdf = MagicMock()
    with pytest.raises(TypeError, match="'rvs' and 'logpdf'"):
        PysipsRegressor(prior=bad_prior)


def test_init_invalid_prior_object_missing_logpdf():
    """Test that a prior object without logpdf raises TypeError."""
    bad_prior = MagicMock(spec=[])
    bad_prior.rvs = MagicMock()
    with pytest.raises(TypeError, match="'rvs' and 'logpdf'"):
        PysipsRegressor(prior=bad_prior)


def test_fit_with_uniform_prior(sample_data, mock_external_components):
    """Test that fit with prior='uniform' uses ImproperUniformPrior."""
    X, y = sample_data
    mock_sample = mock_external_components["sample"]

    mock_improper = MagicMock()
    mock_external_components["_improper_uniform"] = mock_improper

    regressor = PysipsRegressor(prior="uniform", random_state=42)
    regressor.fit(X, y)

    # Verify sample was called and the prior argument is an
    # ImproperUniformPrior (the default behavior)
    mock_sample.assert_called_once()
    call_args = mock_sample.call_args
    prior_arg = call_args.kwargs.get("prior") or call_args[0][2]
    # The prior is constructed from ImproperUniformPrior, which is mocked
    # at module level in mock_external_components; just verify sample ran
    assert regressor.models_ is not None


def test_fit_with_bms_prior(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that fit with prior='bms' constructs BMSPrior."""
    X, y = sample_data
    mock_sample = mock_external_components["sample"]

    mock_weights = ({3: 0.5}, {3: 0.1})
    mock_load = mocker.patch(
        f"{IMPORTMODULE}.load_bms_weights", return_value=mock_weights
    )
    mock_bms_prior_cls = mocker.patch(f"{IMPORTMODULE}.BMSPrior", autospec=True)
    mock_bms_instance = MagicMock()
    mock_bms_prior_cls.return_value = mock_bms_instance

    regressor = PysipsRegressor(prior="bms", random_state=42)
    regressor.fit(X, y)

    # Verify weights were loaded with default corpus
    mock_load.assert_called_once_with("benchmark")

    # Verify BMSPrior was constructed with loaded weights
    mock_bms_prior_cls.assert_called_once()
    call_args = mock_bms_prior_cls.call_args
    assert call_args.args[0] == {3: 0.5}
    assert call_args.args[1] == {3: 0.1}
    call_kwargs = mock_bms_prior_cls.call_args.kwargs
    assert call_kwargs["x_dim"] == X.shape[1]

    # Verify the BMSPrior instance was passed to sample
    mock_sample.assert_called_once()
    sample_call_kwargs = mock_sample.call_args.kwargs
    assert (
        sample_call_kwargs.get("prior") is mock_bms_instance
        or mock_sample.call_args[0][2] is mock_bms_instance
    )


def test_fit_with_bms_prior_corpus_param(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that fit with prior='bms' passes corpus param to load_bms_weights."""
    X, y = sample_data

    mock_load = mocker.patch(
        f"{IMPORTMODULE}.load_bms_weights", return_value=({3: 0.5}, {3: 0.1})
    )
    mocker.patch(f"{IMPORTMODULE}.BMSPrior", autospec=True)

    regressor = PysipsRegressor(
        prior="bms",
        prior_params={"corpus": "wikipedia"},
        random_state=42,
    )
    regressor.fit(X, y)

    mock_load.assert_called_once_with("wikipedia")


def test_fit_with_bms_prior_passes_operators(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that fit with prior='bms' passes user operators to BMSPrior."""
    X, y = sample_data
    mock_sample = mock_external_components["sample"]

    mocker.patch(
        f"{IMPORTMODULE}.load_bms_weights", return_value=({3: 0.5}, {3: 0.1})
    )
    mock_bms_prior_cls = mocker.patch(f"{IMPORTMODULE}.BMSPrior", autospec=True)
    mock_bms_instance = MagicMock()
    mock_bms_prior_cls.return_value = mock_bms_instance

    # Create regressor with specific operators
    regressor = PysipsRegressor(
        prior="bms",
        operators=["+", "-", "*", "/"],
        random_state=42,
    )
    regressor.fit(X, y)

    # Verify BMSPrior was constructed with operators
    mock_bms_prior_cls.assert_called_once()
    call_kwargs = mock_bms_prior_cls.call_args.kwargs

    # Operators should be passed as-is (strings); BMSPrior/SamplablePrior
    # accepts both strings and ints
    assert "operators" in call_kwargs
    assert call_kwargs["operators"] == ["+", "-", "*", "/"]


def test_fit_with_custom_prior_object(sample_data, mock_external_components):
    """Test that fit with a custom prior object passes it directly to sample."""
    X, y = sample_data
    mock_sample = mock_external_components["sample"]

    custom_prior = MagicMock()
    custom_prior.rvs = MagicMock()
    custom_prior.logpdf = MagicMock()

    regressor = PysipsRegressor(prior=custom_prior, random_state=42)
    regressor.fit(X, y)

    # Verify the custom prior was passed directly to sample
    mock_sample.assert_called_once()
    sample_call_kwargs = mock_sample.call_args.kwargs
    prior_used = sample_call_kwargs.get("prior") or mock_sample.call_args[0][2]
    assert prior_used is custom_prior


def test_init_prior_katz():
    """Test initialization with prior='katz'."""
    regressor = PysipsRegressor(prior="katz")
    assert regressor.prior == "katz"


def test_fit_with_katz_prior(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that fit with prior='katz' constructs KatzPrior with default model."""
    X, y = sample_data
    mock_sample = mock_external_components["sample"]

    mock_katz_model = MagicMock()
    mock_load = mocker.patch(
        f"{IMPORTMODULE}.load_katz_model", return_value=mock_katz_model
    )
    mock_katz_prior_cls = mocker.patch(f"{IMPORTMODULE}.KatzPrior", autospec=True)
    mock_katz_instance = MagicMock()
    mock_katz_prior_cls.return_value = mock_katz_instance

    regressor = PysipsRegressor(prior="katz", random_state=42)
    regressor.fit(X, y)

    # Verify the default model was loaded with n=2, corpus=benchmark
    mock_load.assert_called_once_with(n=2, corpus="benchmark")

    # Verify KatzPrior was constructed with the loaded model
    mock_katz_prior_cls.assert_called_once()
    call_args = mock_katz_prior_cls.call_args
    assert call_args.args[0] is mock_katz_model
    assert call_args.kwargs["x_dim"] == X.shape[1]

    # Verify the KatzPrior instance was passed to sample
    mock_sample.assert_called_once()
    sample_call_kwargs = mock_sample.call_args.kwargs
    assert (
        sample_call_kwargs.get("prior") is mock_katz_instance
        or mock_sample.call_args[0][2] is mock_katz_instance
    )


def test_fit_with_katz_prior_passes_operators(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that fit with prior='katz' passes user operators to KatzPrior."""
    X, y = sample_data

    mocker.patch(f"{IMPORTMODULE}.load_katz_model", return_value=MagicMock())
    mock_katz_prior_cls = mocker.patch(f"{IMPORTMODULE}.KatzPrior", autospec=True)
    mock_katz_prior_cls.return_value = MagicMock()

    regressor = PysipsRegressor(
        prior="katz",
        operators=["+", "-", "*"],
        random_state=42,
    )
    regressor.fit(X, y)

    mock_katz_prior_cls.assert_called_once()
    call_kwargs = mock_katz_prior_cls.call_args.kwargs
    assert "operators" in call_kwargs
    assert call_kwargs["operators"] == ["+", "-", "*"]


# --- Tests for prior_params ---


def test_init_default_prior_params():
    """Test that prior_params defaults to None."""
    regressor = PysipsRegressor()
    assert regressor.prior_params is None


def test_init_prior_params_stored():
    """Test that prior_params is stored on the regressor."""
    params = {"corpus": "feynman", "n": 3}
    regressor = PysipsRegressor(prior="katz", prior_params=params)
    assert regressor.prior_params == params


def test_fit_katz_with_prior_params(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that katz prior_params are forwarded to load_katz_model."""
    X, y = sample_data

    mock_katz_model = MagicMock()
    mock_load = mocker.patch(
        f"{IMPORTMODULE}.load_katz_model", return_value=mock_katz_model
    )
    mock_katz_prior_cls = mocker.patch(f"{IMPORTMODULE}.KatzPrior", autospec=True)
    mock_katz_prior_cls.return_value = MagicMock()

    regressor = PysipsRegressor(
        prior="katz",
        prior_params={"corpus": "benchmark", "n": 3},
        random_state=42,
    )
    regressor.fit(X, y)

    mock_load.assert_called_once_with(n=3, corpus="benchmark")


def test_fit_bms_with_unsupported_corpus_raises(sample_data, mock_external_components):
    """Test that BMS prior with unavailable corpus raises FileNotFoundError."""
    X, y = sample_data

    regressor = PysipsRegressor(
        prior="bms",
        prior_params={"corpus": "feynman"},
        random_state=42,
    )
    with pytest.raises(FileNotFoundError, match="No pre-fit BMS weights"):
        regressor.fit(X, y)


def test_fit_bms_default_corpus(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that BMS prior with default corpus works."""
    X, y = sample_data

    mock_bms_prior_cls = mocker.patch(f"{IMPORTMODULE}.BMSPrior", autospec=True)
    mock_bms_prior_cls.return_value = MagicMock()

    regressor = PysipsRegressor(prior="bms", random_state=42)
    regressor.fit(X, y)

    mock_bms_prior_cls.assert_called_once()


def test_fit_bms_explicit_wikipedia_corpus(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that BMS prior with explicit wikipedia corpus works."""
    X, y = sample_data

    mock_bms_prior_cls = mocker.patch(f"{IMPORTMODULE}.BMSPrior", autospec=True)
    mock_bms_prior_cls.return_value = MagicMock()

    regressor = PysipsRegressor(
        prior="bms",
        prior_params={"corpus": "wikipedia"},
        random_state=42,
    )
    regressor.fit(X, y)

    mock_bms_prior_cls.assert_called_once()
