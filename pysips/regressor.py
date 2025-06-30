"""
PySIPS: Python package for Symbolic Inference via Posterior Sampling

This module provides a scikit-learn compatible interface for symbolic regression
using Sequential Monte Carlo (SMC) sampling with Bayesian model selection. It
combines symbolic expression generation, probabilistic proposal mechanisms, and
Laplace approximation for normalized marginal likelihood estimation to discover
mathematical expressions that best explain observed data.

The approach uses SMC to sample from a posterior distribution over symbolic
expressions, allowing for principled uncertainty quantification and model
selection in symbolic regression tasks. Unlike traditional genetic programming
approaches, this method provides probabilistic estimates of model quality and
can naturally handle model uncertainty.

Methodology
-----------
The algorithm works through the following steps:

1. **Expression Generation**: Creates initial symbolic expressions using
   configurable operators and complexity constraints

2. **Proposal Mechanisms**: Uses probabilistic combination of:
   - Mutation operations (structural changes to expressions)
   - Crossover operations (combining expressions from gene pool)

3. **Likelihood Evaluation**: Employs Laplace approximation to estimate
   normalized marginal likelihood for Bayesian model comparison

4. **SMC Sampling**: Uses Sequential Monte Carlo to sample from the
   posterior distribution over symbolic expressions

5. **Model Selection**: Chooses final model using either:
   - Mode selection (most frequently sampled expression)
   - Maximum likelihood selection (highest scoring expression)

Parameters Overview
-------------------
Expression Generation:
    - operators: Mathematical operators to include
    - max_complexity: Maximum expression graph size
    - terminal_probability: Probability of terminal node selection
    - constant_probability: Probability of constant vs variable terminals

Mutation Parameters:
    - command_probability: Probability of operation changes
    - node_probability: Probability of node replacement
    - parameter_probability: Probability of constant modification
    - prune_probability: Probability of expression pruning
    - fork_probability: Probability of expression expansion

Sampling Parameters:
    - num_particles: Population size for SMC
    - num_mcmc_samples: MCMC steps per SMC iteration
    - target_ess: Target effective sample size
    - crossover_pool_size: Size of crossover gene pool

Usage Example
-------------
>>> from pysips import PysipsRegressor
>>> import numpy as np
>>>
>>> # Generate sample data
>>> X = np.random.randn(100, 2)
>>> y = X[:, 0]**2 + 2*X[:, 1] + np.random.normal(0, 0.1, 100)
>>>
>>> # Create and fit regressor
>>> regressor = PysipsRegressor(
...     operators=['+', '*', 'pow'],
...     max_complexity=20,
...     num_particles=100,
...     model_selection='mode',
...     random_state=42
... )
>>> regressor.fit(X, y)
>>>
>>> # Make predictions
>>> y_pred = regressor.predict(X)
>>>
>>> # Get discovered expression
>>> expression = regressor.get_expression()
>>> print(f"Discovered expression: {expression}")
>>>
>>> # Get all sampled models
>>> models, likelihoods = regressor.get_models()

Applications
------------
This approach is particularly well-suited for:
- Scientific discovery where interpretability is crucial
- Problems requiring uncertainty quantification in model selection
- Cases where multiple plausible models exist and need to be ranked
- Regression tasks where symbolic relationships are preferred over black-box models
- Applications requiring principled model complexity control

Notes
-----
The method balances exploration and exploitation through:
- Probabilistic proposal selection between mutation and crossover
- Adaptive sampling that focuses on promising regions of expression space
- Multiple model selection criteria to handle different use cases

For best results, consider:
- Adjusting complexity limits based on problem difficulty
- Tuning mutation/crossover probabilities for your domain
- Using sufficient particles for good posterior approximation
- Setting appropriate number of MCMC samples for mixing
"""

from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator

from .laplace_nmll import LaplaceNmll
from .mutation_proposal import MutationProposal
from .crossover_proposal import CrossoverProposal
from .random_choice_proposal import RandomChoiceProposal
from .sampler import sample

USE_PYTHON = True
USE_SIMPLIFICATION = True
DEFAULT_OPERATORS = ["+", "*"]
DEFALT_PARAMETER_INITIALIZATION_BOUNDS = [-5, 5]


# pylint: disable=R0902,R0913,R0917,R0914
class PysipsRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for PySIPS symbolic regression.

    Parameters
    ----------
    operators : list, default=['+', '*']
        List of operators to use in symbolic expressions.

    max_complexity : int, default=24
        Maximum complexity of symbolic expressions.

    terminal_probability : float, default=0.1
        Probability of selecting a terminal during expression generation.

    constant_probability : float or None, default=None
        Probability of selecting a constant terminal. If None, will be set to 1/(x_dim + 1).

    command_probability : float, default=0.2
        Probability of command mutation.

    node_probability : float, default=0.2
        Probability of node mutation.

    parameter_probability : float, default=0.2
        Probability of parameter mutation.

    prune_probability : float, default=0.2
        Probability of pruning mutation.

    fork_probability : float, default=0.2
        Probability of fork mutation.

    repeat_mutation_probability : float, default=0.05
        Probability of repeating a mutation.

    crossover_pool_size : int, default=num_particles
        Size of the crossover pool.

    mutation_prob : float, default=0.75
        Probability of mutation (vs crossover).

    crossover_prob : float, default=0.25
        Probability of crossover (vs mutation).

    exclusive : bool, default=True
        Whether mutation and crossover are exclusive.

    num_particles : int, default=50
        Number of particles for sampling.

    num_mcmc_samples : int, default=5
        Number of MCMC samples.

    target_ess : float, default=0.8
        Target effective sample size.

    param_init_bounds : list, default=[-5, 5]
        Bounds for parameter initialization.

    opt_restarts : int, default=1
        Number of optimization restarts.

    model_selection : str, default="mode"
        The way to choose a best model from the produced distribution of
        models.  Current options are "mode" for the most frequently occuring
        model and "max_nml" for the model with maximum normalized marginal
        likelihood.

    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        operators=None,
        max_complexity=24,
        terminal_probability=0.1,
        constant_probability=None,
        command_probability=0.2,
        node_probability=0.2,
        parameter_probability=0.2,
        prune_probability=0.2,
        fork_probability=0.2,
        repeat_mutation_probability=0.05,
        crossover_pool_size=None,
        mutation_prob=0.75,
        crossover_prob=0.25,
        exclusive=True,
        num_particles=50,
        num_mcmc_samples=5,
        target_ess=0.8,
        param_init_bounds=None,
        opt_restarts=1,
        model_selection="mode",
        random_state=None,
    ):

        self.operators = operators if operators is not None else DEFAULT_OPERATORS
        self.max_complexity = max_complexity
        self.terminal_probability = terminal_probability
        self.constant_probability = constant_probability
        self.command_probability = command_probability
        self.node_probability = node_probability
        self.parameter_probability = parameter_probability
        self.prune_probability = prune_probability
        self.fork_probability = fork_probability
        self.repeat_mutation_probability = repeat_mutation_probability
        self.crossover_pool_size = (
            crossover_pool_size if crossover_pool_size is not None else num_particles
        )
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.exclusive = exclusive
        self.num_particles = num_particles
        self.num_mcmc_samples = num_mcmc_samples
        self.target_ess = target_ess
        self.param_init_bounds = (
            param_init_bounds
            if param_init_bounds is not None
            else DEFALT_PARAMETER_INITIALIZATION_BOUNDS
        )
        self.opt_restarts = opt_restarts
        self.model_selection = model_selection
        self.random_state = random_state

        # attributes set after fitting
        self.n_features_in_ = None
        self.models_ = None
        self.likelihoods_ = None
        self.best_model_ = None
        self.best_likelihood_ = None

    def _get_generator(self, x_dim):
        """Create expression generator."""
        constant_prob = self.constant_probability
        if constant_prob is None:
            constant_prob = 1 / (x_dim + 1)

        component_generator = ComponentGenerator(
            input_x_dimension=x_dim,
            terminal_probability=self.terminal_probability,
            constant_probability=constant_prob,
        )
        for comp in self.operators:
            component_generator.add_operator(comp)

        return AGraphGenerator(
            self.max_complexity,
            component_generator,
            use_python=USE_PYTHON,
            use_simplification=USE_SIMPLIFICATION,
        )

    def _get_proposal(self, x_dim, generator):
        """Create proposal operator."""
        constant_prob = self.constant_probability
        if constant_prob is None:
            constant_prob = 1 / (x_dim + 1)

        mutation = MutationProposal(
            x_dim,
            operators=self.operators,
            terminal_probability=self.terminal_probability,
            constant_probability=constant_prob,
            command_probability=self.command_probability,
            node_probability=self.node_probability,
            parameter_probability=self.parameter_probability,
            prune_probability=self.prune_probability,
            fork_probability=self.fork_probability,
            repeat_mutation_probability=self.repeat_mutation_probability,
        )

        # Generate crossover pool
        pool = set()
        while len(pool) < self.crossover_pool_size:
            pool.add(generator())
        crossover = CrossoverProposal(list(pool))

        # Create combined proposal
        return RandomChoiceProposal(
            [mutation, crossover],
            [self.mutation_prob, self.crossover_prob],
            self.exclusive,
        )

    def fit(self, X, y):
        """
        Fit the symbolic regression model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check and validate input data
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        # Set up the sampling config
        x_dim = X.shape[1]

        # Create generator, proposal, and likelihood
        generator = self._get_generator(x_dim)
        proposal = self._get_proposal(x_dim, generator)
        likelihood = LaplaceNmll(X, y)

        # Run sampling
        models, likelihoods = sample(
            likelihood,
            proposal,
            generator,
            seed=self.random_state,
            kwargs={
                "num_particles": self.num_particles,
                "num_mcmc_samples": self.num_mcmc_samples,
                "target_ess": self.target_ess,
            },
        )

        # Save the models and their likelihoods
        self.models_ = models
        self.likelihoods_ = likelihoods

        # Select the best model
        if self.model_selection == "max_nml":
            best_idx = np.argmax(likelihoods)
        elif self.model_selection == "mode":
            model_indices = {model: i for i, model in enumerate(models)}
            model_counts = Counter(model for model in self.models_)
            most_common_model = model_counts.most_common(1)[0][0]
            best_idx = model_indices[most_common_model]
        else:
            raise KeyError(
                f"model_selection method {self.model_selection} not recognized."
            )

        self.best_model_ = models[best_idx]
        self.best_likelihood_ = likelihoods[best_idx]

        return self

    def predict(self, X):
        """
        Predict using the best symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, ["best_model_", "models_"])
        X = check_array(X)

        # Ensure consistent feature count
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but BayesRRegressor was "
                f"trained with {self.n_features_in_} features."
            )

        # Use the best model for prediction
        return self.best_model_.evaluate_equation_at(X).flatten()

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) with respect to y.
        """
        # Use default implementation from scikit-learn
        return super().score(X, y, sample_weight=sample_weight)

    def get_expression(self):
        """
        Get the symbolic expression of the best model.

        Returns
        -------
        expression : str
            String representation of the best model.
        """
        check_is_fitted(self, ["best_model_"])
        return str(self.best_model_)

    def get_models(self):
        """
        Get all sampled models and their likelihoods.

        Returns
        -------
        models : list
            List of all sampled models.
        likelihoods : numpy.ndarray
            Array of corresponding likelihoods.
        """
        check_is_fitted(self, ["models_", "likelihoods_"])
        return self.models_, self.likelihoods_
