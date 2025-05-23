from bingo.symbolic_regression.explicit_regression import (
    ExplicitTrainingData,
    ExplicitRegression,
)
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer


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
