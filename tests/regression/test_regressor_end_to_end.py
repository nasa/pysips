import numpy as np
import pytest
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from pysips.regressor import PysipsRegressor

REGRESSOR_TEST_MAX_EQUATION_EVALS = 100
LATENT_DIM = 4


@pytest.fixture
def synthetic_data():
    """Fixture to create synthetic sin wave data."""
    n_pts = 21
    X = np.c_[np.linspace(0, 2 * np.pi, n_pts)]
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
        prior=[norm(0, 1) for _ in range(LATENT_DIM)],
        num_particles=20,
        max_equation_evals=REGRESSOR_TEST_MAX_EQUATION_EVALS,
        random_state=42,
        show_progress_bar=False,
    )


def test_basic_end_to_end(train_test_data, base_regressor):
    """Test basic end-to-end workflow for PysipsRegressor."""
    X_train, X_test, y_train, y_test = train_test_data
    base_regressor.fit(X_train, y_train)
    y_pred = base_regressor.predict(X_test)
    mean_squared_error(y_test, y_pred)


def test_hyperparameter_optimization(synthetic_data):
    """Test compatibility with scikit-learn's hyperparameter optimization."""
    X, y = synthetic_data

    base_model = PysipsRegressor(
        prior=[norm(0, 1) for _ in range(LATENT_DIM)],
        num_particles=10,
        num_mcmc_samples=10,
        max_equation_evals=REGRESSOR_TEST_MAX_EQUATION_EVALS,
        random_state=42,
        show_progress_bar=False,
    )

    param_grid = {"num_particles": [10, 20]}

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=2,
        scoring="neg_mean_squared_error",
        verbose=0,
        n_jobs=1,
    )
    grid_search.fit(X, y)

    assert hasattr(grid_search, "best_params_")
    assert hasattr(grid_search.best_estimator_, "best_model_")
    y_pred = grid_search.predict(X)
    assert y_pred.shape == y.shape
