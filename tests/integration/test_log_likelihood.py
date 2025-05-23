import numpy as np

from pysips.laplace_nmll import LaplaceNmll
from pysips.metropolis import Metropolis
from bingo.symbolic_regression.agraph.agraph import AGraph


def test_log_likelihood_relative():
    x = np.arange(0, 100)
    model = lambda a, b: a * x + b
    data = model(5, 5) + np.random.default_rng(34).normal(0, 0.1, 100)

    models = np.c_[[AGraph(equation="1.0"), AGraph(equation="1.0 + 2.0*X_0")]]
    likelihood = LaplaceNmll(np.c_[x], data)
    mcmc = Metropolis(likelihood=likelihood, proposal=None, prior=None)

    log_likes = mcmc.evaluate_log_likelihood(models).flatten()
    assert log_likes[0] < log_likes[1]
