import numpy as np
import pytest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from pysips.regressor import PysipsRegressor


@pytest.fixture
def synthetic_data():
    """Fixture to create synthetic sin wave data."""
    n_pts = 21
    X = np.c_[np.linspace(0, 2 * np.pi, n_pts)]
    # Use seed 34 for consistent test data
    y = (np.sin(X) * 2 + 4).flatten() + np.random.default_rng(34).normal(0, 0.5, n_pts)
    return X, y


@pytest.fixture
def train_test_data(synthetic_data):
    """Fixture to split data into train and test sets."""
    X, y = synthetic_data
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def base_regressor():
    """Fixture for basic regressor with default settings."""
    return PysipsRegressor(
        operators=["+", "*", "sin"],
        max_complexity=24,
        num_particles=20,
        random_state=42,
    )


def test_basic_end_to_end(train_test_data, base_regressor):
    """Test basic end-to-end workflow for PysipsRegressor."""
    X_train, X_test, y_train, y_test = train_test_data

    model = base_regressor
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # print(f"Best expression: {model.get_expression()}")
    # print(f"Test MSE: {mse:.4f}")
    # print(f"R² score: {model.score(X_test, y_test):.4f}")


def test_hyperparameter_optimization(synthetic_data):
    """Test compatibility with scikit-learn's hyperparameter optimization."""
    X, y = synthetic_data

    # Create the regressor with reduced computation for faster testing
    base_model = PysipsRegressor(
        operators=["+", "*", "sin"],
        max_complexity=20,
        num_particles=10,
        num_mcmc_samples=50,
        random_state=42,
    )

    # Define hyperparameter grid
    param_grid = {
        "max_complexity": [15, 20],
        "terminal_probability": [0.1, 0.2],
        "mutation_prob": [0.6, 0.7],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=1,  # Use single job for testing
    )

    # Perform grid search
    grid_search.fit(X, y)

    # Check that grid search completed successfully
    assert hasattr(grid_search, "best_params_")

    # Check that the best estimator is fitted
    assert hasattr(grid_search.best_estimator_, "best_model_")

    # Verify it can make predictions
    y_pred = grid_search.predict(X)
    assert y_pred.shape == y.shape

    # print(f"Best parameters: {grid_search.best_params_}")
    # print(f"Best CV score: {-grid_search.best_score_:.4f}")  # Convert back to MSE
    # print(f"Best model expression: {grid_search.best_estimator_.get_expression()}")
