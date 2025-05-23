import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from pysips.regressor import PysipsRegressor


def test_basic_end_to_end():

    n_pts = 21
    X = np.c_[np.linspace(0, 2 * np.pi, n_pts)]
    y = (np.sin(X) * 2 + 4).flatten() + np.random.default_rng(34).normal(0, 0.5, n_pts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = PysipsRegressor(
        operators=["+", "*", "sin"],
        max_complexity=24,
        num_particles=20,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Best expression: {model.get_expression()}")
    print(f"Test MSE: {mse:.4f}")
    print(f"R² score: {model.score(X_test, y_test):.4f}")
