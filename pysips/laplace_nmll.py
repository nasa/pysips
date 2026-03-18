"""Laplace Approximation for Normalized Marginal Log-Likelihood Estimation.

This module provides functionality for computing the Normalized Marginal
Log-Likelihood (NMLL) using the Laplace approximation method.  It uses
the ``bingo.expressions`` AGraph interface where constant optimization
and scoring are built directly into the expression object.

Key Features
------------
- Built-in ``fit`` / ``score`` via ``AGraphExpression``
- Multiple optimization restarts to avoid local minima
- Automatic parameter bound initialization for robust optimization

Usage Example
-------------
>>> import numpy as np
>>>
>>> # Generate sample data
>>> X = np.random.randn(100, 2)
>>> y = X[:, 0]**2 + X[:, 1] + np.random.normal(0, 0.1, 100)
>>>
>>> # Create NMLL evaluator
>>> nmll_evaluator = LaplaceNmll(X, y, opt_restarts=3)
>>>
>>> # Evaluate a symbolic model (assuming you have an EvolvableExpression)
>>> # nmll_score = nmll_evaluator(model)

Notes
-----
The multiple restart strategy helps ensure robust optimization by avoiding
local minima in the parameter space, which is especially important for
complex symbolic expressions.
"""

import numpy as np


# pylint: disable=R0903
class LaplaceNmll:
    """Normalized Marginal Likelihood using Laplace approximation.

    Parameters
    ----------
    X : 2d numpy array
        Array of shape [num_datapoints, num_features].
    y : 1d numpy array
        Array of labels of shape [num_datapoints].
    opt_restarts : int, optional
        Number of optimization restarts (first uses original constants,
        subsequent restarts randomize). Default 1.
    param_init_bounds : list of float, optional
        [low, high] bounds for random constant initialization on restarts.
        Default [-5, 5].
    """

    def __init__(self, X, y, opt_restarts=1, param_init_bounds=None):
        self._X = np.atleast_2d(np.asarray(X, dtype=float))
        self._y = np.asarray(y, dtype=float).ravel()
        self._opt_restarts = opt_restarts
        self._bounds = param_init_bounds if param_init_bounds is not None else [-5, 5]

    def __call__(self, model):
        """Calculate NMLL using the Laplace approximation.

        Parameters
        ----------
        model : EvolvableExpression
            A bingo ``EvolvableExpression`` wrapping an ``AGraphExpression``.

        Returns
        -------
        float
            The normalized marginal log-likelihood (higher is better).
        """
        expr = model.expression

        # First attempt: fit with current (original) constants
        expr.fit(self._X, self._y)
        best_nmll = expr.score(self._X, self._y, metric="laplace_nmll")
        best_consts = expr.constants

        # Additional restarts with randomized constants
        lo, hi = self._bounds
        for _ in range(self._opt_restarts - 1):
            n_consts = len(expr.constants)
            if n_consts > 0:
                expr.constants = tuple(
                    np.random.uniform(lo, hi, size=n_consts)
                )
            expr.fit(self._X, self._y)
            trial_nmll = expr.score(self._X, self._y, metric="laplace_nmll")
            if trial_nmll > best_nmll:
                best_nmll = trial_nmll
                best_consts = expr.constants

        expr.constants = best_consts
        return best_nmll
