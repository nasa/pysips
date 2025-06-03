import pytest
import numpy as np
from unittest.mock import MagicMock
from pytest_mock import MockerFixture

from pysips.regressor import PysipsRegressor

# Dynamically get the module containing the PysipsRegressor class
IMPORTMODULE = PysipsRegressor.__module__


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
        f"{IMPORTMODULE}.ComponentGenerator", autospec=True
    )
    # needs to provide unique outputs for pool generation
    mock_agraph_gen = mocker.MagicMock(side_effect=lambda: np.random.random())
    mock_agraph_gen_constructor = mocker.patch(
        f"{IMPORTMODULE}.AGraphGenerator", autospec=True, return_value=mock_agraph_gen
    )
    mock_laplace_nmll = mocker.patch(f"{IMPORTMODULE}.LaplaceNmll", autospec=True)
    mock_mutation_proposal = mocker.patch(
        f"{IMPORTMODULE}.MutationProposal", autospec=True
    )
    mock_crossover_proposal = mocker.patch(
        f"{IMPORTMODULE}.CrossoverProposal", autospec=True
    )
    mock_random_choice_proposal = mocker.patch(
        f"{IMPORTMODULE}.RandomChoiceProposal", autospec=True
    )
    mock_sample = mocker.patch(f"{IMPORTMODULE}.sample", autospec=True)

    # Configure the mock sample function to return a model and likelihood
    mock_model = MagicMock()
    mock_model.__str__.return_value = "2*X_0"
    mock_likelihoods = np.array([0.9])
    mock_sample.return_value = ([mock_model], mock_likelihoods)

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
