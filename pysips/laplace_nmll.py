"""
Laplace Approximation for Normalized Marginal Log-Likelihood Estimation.

This module provides functionality for computing the Normalized Marginal Log-Likelihood
(NMLL) using the Laplace approximation method. It integrates with the bingo symbolic
regression library to evaluate the likelihood of symbolic mathematical models given
observed data.

The Laplace approximation is a method for approximating integrals that appear in
Bayesian model selection, particularly useful for comparing different symbolic
regression models. It approximates the marginal likelihood by making a Gaussian
approximation around the maximum a posteriori (MAP) estimate of the parameters.

Key Features
------------
- Integration with bingo's symbolic regression framework
- Multiple optimization restarts to avoid local minima
- Configurable scipy-based optimization backend
- Automatic parameter bound initialization for robust optimization

Usage Example
-------------
>>> import numpy as np
>>> from bingo.symbolic_regression import AGraph
>>>
>>> # Generate sample data
>>> X = np.random.randn(100, 2)
>>> y = X[:, 0]**2 + X[:, 1] + np.random.normal(0, 0.1, 100)
>>>
>>> # Create NMLL evaluator
>>> nmll_evaluator = LaplaceNmll(X, y, opt_restarts=3)
>>>
>>> # Evaluate a symbolic model (assuming you have an AGraph model)
>>> # nmll_score = nmll_evaluator(model)

Notes
-----
The multiple restart strategy helps ensure robust optimization by avoiding
local minima in the parameter space, which is especially important for
complex symbolic expressions.
"""

from bingo.symbolic_regression.explicit_regression import (
    ExplicitTrainingData,
    ExplicitRegression,
)
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer


# pylint: disable=R0903
class LaplaceNmll:
    """Normalized Marginal Likelihood using Laplace approximation

    Parameters
    ----------
    X : 2d Numpy Array
        Array of shape [num_datapoints, num_features] representing the input features
    y : 1d Numpy Array
        Array of labels of shape [num_datapoints]
    opt_restarts : int, optional
        number of times to perform gradient based optimization, each with different
        random initialization, by default 1
    **optimizer_kwargs :
        any keyword arguments to be passed to bingo's scipy optimizer
    """

    def __init__(self, X, y, opt_restarts=1, **optimizer_kwargs):
        self._neg_nmll = self._init_neg_nmll(X, y)
        self._deterministic_optimizer = self._init_deterministic_optimizer(
            self._neg_nmll, **optimizer_kwargs
        )
        self._opt_restarts = opt_restarts

    def _init_neg_nmll(self, X, y):
        training_data = ExplicitTrainingData(X, y)
        return ExplicitRegression(
            training_data=training_data, metric="negative nmll laplace"
        )

    def _init_deterministic_optimizer(self, objective, **optimizer_kwargs):
        if "param_init_bounds" not in optimizer_kwargs:
            optimizer_kwargs["param_init_bounds"] = [-5, 5]
        return ScipyOptimizer(objective, method="lm", **optimizer_kwargs)

    def __call__(self, model):
        """calaculates NMLL using the Laplace approximation

        Parameters
        ----------
        model : AGraph
            a bingo equation using the AGraph representation
        """
        self._deterministic_optimizer(model)
        nmll = -self._neg_nmll(model)
        consts = model.get_local_optimization_params()
        for _ in range(self._opt_restarts - 1):
            self._deterministic_optimizer(model)
            trial_nmll = -self._neg_nmll(model)
            if trial_nmll > nmll:
                nmll = trial_nmll
                consts = model.get_local_optimization_params()
        model.set_local_optimization_params(consts)

        return nmll
