import pytest
import numpy as np
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from sklearn.base import RegressorMixin

from pysips.bingo_construction import BingoConstructionConfig
from pysips.regressor import PysipsRegressor

# Dynamically get the module containing the PysipsRegressor class
IMPORTMODULE = PysipsRegressor.__module__
BINGO_CONSTRUCTION_MODULE = "pysips.bingo_construction"


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
        f"{BINGO_CONSTRUCTION_MODULE}.ComponentGenerator", autospec=True
    )
    # needs to provide unique outputs for pool generation
    mock_agraph_gen = mocker.MagicMock(side_effect=lambda: np.random.random())
    mock_agraph_gen_constructor = mocker.patch(
        f"{BINGO_CONSTRUCTION_MODULE}.AGraphGenerator",
        autospec=True,
        return_value=mock_agraph_gen,
    )
    mock_laplace_nmll = mocker.patch(f"{IMPORTMODULE}.LaplaceNmll", autospec=True)
    mock_mutation_proposal = mocker.patch(
        f"{BINGO_CONSTRUCTION_MODULE}.MutationProposal", autospec=True
    )
    mock_crossover_proposal = mocker.patch(
        f"{BINGO_CONSTRUCTION_MODULE}.CrossoverProposal", autospec=True
    )
    mock_random_choice_proposal = mocker.patch(
        f"{BINGO_CONSTRUCTION_MODULE}.RandomChoiceProposal", autospec=True
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

    # Create a mock model with expression.predict method
    mock_expr = MagicMock()
    mock_predictions = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    mock_expr.predict.return_value = mock_predictions

    mock_model = MagicMock()
    mock_model.expression = mock_expr

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

    # Verify that expression.predict was called with X
    mock_expr.predict.assert_called_once_with(X)
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

    mock_expr = MagicMock()
    predictions = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    predictions[0] = invalid_value  # Set first prediction to the invalid value
    mock_expr.predict.return_value = predictions

    mock_model = MagicMock()
    mock_model.expression = mock_expr

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


def test_fit_uses_factory_returned_prior(sample_data, mock_external_components, mocker):
    """Test that fit passes the factory result through to sample()."""
    X, y = sample_data
    mock_sample = mock_external_components["sample"]

    resolved_prior = MagicMock()
    mocker.patch(f"{IMPORTMODULE}.build_prior", return_value=resolved_prior)

    regressor = PysipsRegressor(prior="uniform", random_state=42)
    regressor.fit(X, y)

    mock_sample.assert_called_once()
    sample_call_kwargs = mock_sample.call_args.kwargs
    assert sample_call_kwargs.get("prior") is resolved_prior or mock_sample.call_args[0][2] is resolved_prior


def test_init_prior_katz():
    """Test initialization with prior='katz'."""
    regressor = PysipsRegressor(prior="katz")
    assert regressor.prior == "katz"


def test_fit_delegates_prior_resolution_to_factory(
    sample_data, mock_external_components, mocker: MockerFixture
):
    """Test that fit delegates prior resolution to build_prior()."""
    X, y = sample_data
    mock_build_prior = mocker.patch(f"{IMPORTMODULE}.build_prior", return_value=MagicMock())

    regressor = PysipsRegressor(
        prior="katz",
        prior_params={"corpus": "benchmark", "n": 3},
        operators=["+", "-", "*"],
        random_state=42,
    )
    regressor.fit(X, y)

    mock_build_prior.assert_called_once()
    call_args = mock_build_prior.call_args
    assert call_args.args == ("katz", {"corpus": "benchmark", "n": 3})
    assert call_args.kwargs["operators"] == ["+", "-", "*"]
    assert call_args.kwargs["x_dim"] == X.shape[1]
    assert isinstance(call_args.kwargs["bingo_config"], BingoConstructionConfig)


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


def test_fit_invalid_prior_string_raises(sample_data, mock_external_components):
    """Test that invalid prior strings fail during fit-time resolution."""
    X, y = sample_data
    regressor = PysipsRegressor(prior="invalid", random_state=42)

    with pytest.raises(ValueError, match="Unknown prior 'invalid'"):
        regressor.fit(X, y)


def test_fit_invalid_prior_object_missing_rvs(sample_data, mock_external_components):
    """Test that a custom prior without rvs fails during fit-time resolution."""
    X, y = sample_data
    bad_prior = MagicMock(spec=[])
    bad_prior.logpdf = MagicMock()
    regressor = PysipsRegressor(prior=bad_prior, random_state=42)

    with pytest.raises(TypeError, match="'rvs' and 'logpdf'"):
        regressor.fit(X, y)


def test_fit_invalid_prior_object_missing_logpdf(sample_data, mock_external_components):
    """Test that a custom prior without logpdf fails during fit-time resolution."""
    X, y = sample_data
    bad_prior = MagicMock(spec=[])
    bad_prior.rvs = MagicMock()
    regressor = PysipsRegressor(prior=bad_prior, random_state=42)

    with pytest.raises(TypeError, match="'rvs' and 'logpdf'"):
        regressor.fit(X, y)
