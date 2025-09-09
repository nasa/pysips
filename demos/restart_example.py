import multiprocessing
import numpy as np

from pathlib import Path
from pysips import PysipsRegressor
from sklearn.metrics import r2_score


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
        random_state=42,
        # setting a time limit
        max_time=max_time,
        # set a checkpoint file to save progress in case of interruption
        checkpoint_file="example.chkpt",
    )

    regressor.fit(X, y)

    expression = regressor.get_expression()
    y_pred = regressor.predict(X)
    print(f"Discovered expression: {expression}")
    print(f"R² score: {r2_score(y, y_pred):.4f}")
    print(f"Number of steps: {len(regressor.phis_)-1}")
    print(f"Phi sequence: {regressor.phis_}")
    return regressor.phis_, regressor.likelihoods_


def main():

    # remove checkpoint file if exists
    (Path(__file__).parent / "example.chkpt").unlink(missing_ok=True)

    timeout_seconds = 20
    process = multiprocessing.Process(target=run_pysips)
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        print(f">> Killing run_pysips after {timeout_seconds} seconds.")
        process.terminate()
        process.join()
    else:
        print(">> Function completed naturally.")

    # restart and complete
    print(">> Restarting run_pysips and waiting for completion...")
    run_pysips()

    # note that additional calls will not run
    print(">> Restarting run_pysips again does nothing:")
    run_pysips()


if __name__ == "__main__":
    import random

    random.seed(7)
    np.random.seed(7)
    main()
