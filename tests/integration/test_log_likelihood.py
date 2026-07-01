import numpy as np

from pysips.laplace_nmll import LaplaceNmll
from bingo.expressions import AGraphExpression
from bingo.expressions.agraph.evolvable import EvolvableExpression


def test_log_likelihood_relative():
    """A linear model should score better than a constant on linear data."""
    x = np.arange(0, 100)
    data = 5 * x + 5 + np.random.default_rng(34).normal(0, 0.1, 100)

    nmll = LaplaceNmll(np.c_[x], data)

    constant = EvolvableExpression(AGraphExpression(equation="1.0"))
    linear = EvolvableExpression(AGraphExpression(equation="1.0 + 2.0*X_0"))

    score_constant = nmll(constant)
    score_linear = nmll(linear)

    assert score_constant < score_linear
