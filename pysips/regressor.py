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
        Probability of selecting a constant terminal. If None, will be set to 1/(X_dim + 1).

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
        operators=["+", "*"],
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
        param_init_bounds=[-5, 5],
        opt_restarts=1,
        model_selection="mode",
        random_state=None,
    ):

        self.operators = operators
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
        self.param_init_bounds = param_init_bounds
        self.opt_restarts = opt_restarts
        self.model_selection = model_selection
        self.random_state = random_state

    def _get_generator(self, X_dim):
        """Create expression generator."""
        USE_PYTHON = True
        USE_SIMPLIFICATION = True
        constant_prob = self.constant_probability
        if constant_prob is None:
            constant_prob = 1 / (X_dim + 1)

        component_generator = ComponentGenerator(
            input_x_dimension=X_dim,
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

    def _get_proposal(self, X_dim, generator):
        """Create proposal operator."""
        constant_prob = self.constant_probability
        if constant_prob is None:
            constant_prob = 1 / (X_dim + 1)

        mutation = MutationProposal(
            X_dim,
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
        X_dim = X.shape[1]

        # Create generator, proposal, and likelihood
        generator = self._get_generator(X_dim)
        proposal = self._get_proposal(X_dim, generator)
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
        return self.best_model_.evaluate_equation_at(X)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) with respect to y.
        """
        # Use default implementation from scikit-learn
        return super().score(X, y)

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
