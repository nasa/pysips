import threading
import time

import numpy as np
from pysips import PysipsRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])


def equation_eval(x):
    return x**2 + 3.5 * x**3


def run_pysips(max_time=None):
    X = init_x_vals(-10, 10, 100)
    y = equation_eval(X).flatten()

    regressor = PysipsRegressor(
        # the mathematical operations that can be used in equations
        operators=["+", "-", "*"],
        # a complexity limit for equations
        max_complexity=12,
        # the number of equations that will represent the model posterior
        # similar to a population size in a genetic algorithm
        # computation times increase with this value, effectiveness does too
        num_particles=150,
        # length of MCMC chains between SMC target distributions
        # computation times increase with this value
        # effectiveness also increases (but may saturate at larger values)
        num_mcmc_samples=10,
        # to control randomness
        # random_state=42,
        # setting a time limit
        max_time=max_time,
    )

    regressor.fit(X, y)

    expression = regressor.get_expression()
    y_pred = regressor.predict(X)
    print(f"Discovered expression: {expression}")
    print(f"R² score: {r2_score(y, y_pred):.4f}")
    print(f"Number of steps: {len(regressor.phis_)-1}")
    print(f"Phi sequence: {regressor.phis_}")
    return regressor.phis_, regressor.likelihoods_


def force_timeout(signum, frame):
    raise TimeoutError("Function timed out!")


def main():
    timeout_seconds = 10
    thread = threading.Thread(target=run_pysips)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print(f"Function stopped after {timeout_seconds} seconds")
    else:
        print("Function completed naturally")

    # restart and complete
    run_pysips()

    # what if I run this and it was already completed?
    run_pysips()


if __name__ == "__main__":
    import random

    random.seed(7)
    np.random.seed(7)
    main()
