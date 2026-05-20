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
    - max_time: Maximum runtime for sampling process

Checkpointing:
    - checkpoint_file: File path for saving/loading sampling progress
      If the checkpoint file exists, fitting will attempt to resume from
      the saved state and continue updating the checkpoint as it proceeds

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
...     checkpoint_file='my_run.checkpoint',
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
>>>
>>> # For hyperparameter tuning, disable progress bar
>>> regressor_quiet = PysipsRegressor(
...     operators=['+', '*'],
...     num_particles=50,
...     show_progress_bar=False,  # No progress output
...     random_state=42
... )
>>> regressor_quiet.fit(X, y)  # Silent fitting

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

Checkpointing allows for:
- Resuming interrupted long-running fits
- Incremental progress saving during extended sampling runs
- Recovery from system failures or resource limitations
"""

from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .bingo_proposal_mixin import BingoProposalMixin
from .priors import (
    ImproperUniformPrior,
    BMSPrior,
    load_bms_weights,
    KatzPrior,
    load_katz_model,
    SizeCalibratedPrior,
)
from .priors.prebuilt_loader import load_prebuilt_size_calibrated
from .laplace_nmll import LaplaceNmll
from .sampler import sample

_KNOWN_PRIOR_STRINGS = {
    "uniform",
    "bms",
    "katz",
    "size_calibrated_uniform",
    "size_calibrated_katz",
    "size_calibrated_bms",
}

DEFAULT_OPERATORS = ["+", "*"]
DEFALT_PARAMETER_INITIALIZATION_BOUNDS = [-5, 5]


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals
class PysipsRegressor(BingoProposalMixin, BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for PySIPS symbolic regression.

    This regressor uses Sequential Monte Carlo (SMC) sampling to explore
    the space of symbolic expressions and find mathematical models that
    best explain the observed data. The approach provides principled
    uncertainty quantification and supports checkpointing for long-running
    fits.

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

    checkpoint_file : str or None, default=None
        File path for saving and loading sampling progress. If the checkpoint
        file exists, fitting will attempt to resume from the saved state and
        continue updating the checkpoint as sampling proceeds. If None, no
        checkpointing is performed.

    random_state : int or None, default=None
        Random seed for reproducibility.

    max_time : float or None, default=None
        Maximum time in seconds to run the sampling process. If None,
        the sampling will run until completion without time constraints.
        Cannot be used together with max_equation_evals.

    max_equation_evals : int or None, default=None
        Maximum number of evaluations during the sampling process. If None,
        the sampling will run until completion without time constraints.
        Cannot be used together with max_time.

    prior : str or object, default="uniform"
        The prior distribution to use for sampling. Can be:

        - ``"uniform"`` : Improper uniform prior (default). Generates
          random symbolic expressions with equal probability.
        - ``"bms"`` : Bayesian Machine Scientist prior. Scores
          expressions based on weighted operator frequency counts.
          Only pre-fit weights are supported; raises an error if a
          pre-fit is not available for the requested corpus.
        - ``"katz"`` : Katz back-off n-gram prior. Scores expressions
          via operator n-gram probabilities. Pre-fit models are loaded
          when available; otherwise the model is fit from corpus on the
          fly and cached for future use.
        - ``"size_calibrated_uniform"`` : Uniform base prior with
          corpus-calibrated size distribution. Requires the standard
          operator set ``[+,-,*,/,sin,cos,exp,log]`` and ``x_dim=1``.
        - ``"size_calibrated_katz"`` : Katz base prior with
          corpus-calibrated size distribution. Same operator/x_dim
          requirements as ``"size_calibrated_uniform"``.
        - ``"size_calibrated_bms"`` : BMS base prior with
          corpus-calibrated size distribution. Currently raises
          ``NotImplementedError``; use ``fit_size_calibrated_prior()``
          for BMS-based calibration.
        - A custom prior object with ``rvs(N, random_state=None)``
          and ``logpdf(x)`` methods. The ``rvs`` method should return
          an array of shape ``(N, 1)`` and ``logpdf`` should return
          an array of shape ``(N, 1)``.

    prior_params : dict or None, default=None
        Optional parameters for configuring string-based priors.
        Ignored when *prior* is a custom object.

        For ``"katz"``:

        - ``"corpus"`` (str): Corpus name for fitting/loading, e.g.
          ``"wikipedia"``, ``"feynman"``, ``"benchmark"``.
          Default ``"wikipedia"``.
        - ``"n"`` (int): N-gram order. Default ``2``.
        - ``"normalize"`` (bool): If ``True``, divide the total
          log-probability by the number of phrases, yielding a
          per-phrase mean log-probability (cross-entropy).
          Default ``False``.

        For ``"bms"``:

        - ``"corpus"`` (str): Corpus name identifying which pre-fit
          weights to use. Currently only ``"wikipedia"`` (default) is
          available.

        For ``"size_calibrated_uniform"`` / ``"size_calibrated_katz"``:

        - ``"floor_log_prob"`` (float): Log-probability assigned to
          expression sizes not covered by the corpus histogram.
          Default ``-inf``.

    show_progress_bar : bool, default=True
        Whether to display a progress bar during fitting. When False, the
        progress bar will be hidden, which is useful for hyperparameter
        tuning or when running multiple fits in parallel.
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
        checkpoint_file=None,
        random_state=None,
        prior="uniform",
        prior_params=None,
        max_time=None,
        max_equation_evals=None,
        show_progress_bar=True,
    ):
        # Validate prior parameter
        if isinstance(prior, str) and prior not in _KNOWN_PRIOR_STRINGS:
            raise ValueError(
                f"Unknown prior '{prior}'. Expected one of "
                f"{sorted(_KNOWN_PRIOR_STRINGS)} or a prior object with "
                f"'rvs' and 'logpdf' methods."
            )
        if not isinstance(prior, str):
            if not hasattr(prior, "rvs") or not hasattr(prior, "logpdf"):
                raise TypeError("Custom prior must have 'rvs' and 'logpdf' methods.")

        # Validate that max_time and max_equation_evals are not both specified
        if max_time is not None and max_equation_evals is not None:
            raise ValueError(
                "max_time and max_equation_evals cannot both be specified. "
                "Please choose one constraint method."
            )

        # Initialize mixin with proposal/generator parameters
        super().__init__(
            max_complexity=max_complexity,
            terminal_probability=terminal_probability,
            constant_probability=constant_probability,
            command_probability=command_probability,
            node_probability=node_probability,
            parameter_probability=parameter_probability,
            prune_probability=prune_probability,
            fork_probability=fork_probability,
            repeat_mutation_probability=repeat_mutation_probability,
            crossover_pool_size=(
                crossover_pool_size
                if crossover_pool_size is not None
                else num_particles
            ),
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            exclusive=exclusive,
        )

        # Regressor-specific attributes
        self.operators = operators if operators is not None else DEFAULT_OPERATORS
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
        self.checkpoint_file = checkpoint_file
        self.random_state = random_state
        self.prior = prior
        self.prior_params = prior_params
        self.max_time = max_time
        self.max_equation_evals = max_equation_evals
        self.show_progress_bar = show_progress_bar

        # attributes set after fitting
        self.n_features_in_ = None
        self.models_ = None
        self.likelihoods_ = None
        self.phis_ = None
        self.best_model_ = None
        self.best_likelihood_ = None

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
        generator = self._get_generator(x_dim, self.operators)
        prior = self._build_prior(generator, x_dim)
        proposal = self._get_proposal(x_dim, generator, self.operators)
        likelihood = LaplaceNmll(
            X,
            y,
            opt_restarts=self.opt_restarts,
            param_init_bounds=self.param_init_bounds,
        )

        # Run sampling
        models, likelihoods, phis = sample(
            likelihood,
            proposal,
            prior,
            max_time=self.max_time,
            max_equation_evals=self.max_equation_evals,
            seed=self.random_state,
            checkpoint_file=self.checkpoint_file,
            show_progress_bar=self.show_progress_bar,
            kwargs={
                "num_particles": self.num_particles,
                "num_mcmc_samples": self.num_mcmc_samples,
                "target_ess": self.target_ess,
            },
        )

        # Save the models and their likelihoods
        self.models_ = models
        self.likelihoods_ = likelihoods
        self.phis_ = phis

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

    def _build_prior(self, generator, x_dim):
        """Resolve the prior configuration into a prior object.

        Parameters
        ----------
        generator : callable
            AGraph generator (used for the "uniform" prior).
        x_dim : int
            Number of input features.

        Returns
        -------
        prior : object
            A prior object with ``rvs`` and ``logpdf`` methods.
        """
        if isinstance(self.prior, str):
            params = self.prior_params or {}

            if self.prior == "uniform":
                return ImproperUniformPrior(generator)

            if self.prior == "bms":
                corpus = params.get("corpus", "benchmark")
                weights, squared_weights = load_bms_weights(corpus)
                print(f"Using BMS prior (corpus={corpus!r}).")
                return BMSPrior(
                    weights,
                    squared_weights,
                    operators=self.operators,
                    x_dim=x_dim,
                    max_complexity=self.max_complexity,
                )

            if self.prior == "katz":
                n = params.get("n", 2)
                corpus = params.get("corpus", "benchmark")
                normalize = params.get("normalize", False)
                print(
                    f"Using Katz prior (n={n}, corpus={corpus!r}, normalize={normalize})."
                )
                model = load_katz_model(n=n, corpus=corpus)
                return KatzPrior(
                    model,
                    normalize=normalize,
                    operators=self.operators,
                    x_dim=x_dim,
                    max_complexity=self.max_complexity,
                )

            if self.prior in (
                "size_calibrated_uniform",
                "size_calibrated_katz",
                "size_calibrated_bms",
            ):
                return self._build_size_calibrated_prior(x_dim, params)

        # Custom prior object — pass through as-is
        return self.prior

    def _build_size_calibrated_prior(self, x_dim, params):
        """Build a size-calibrated prior from pre-built data.

        Parameters
        ----------
        x_dim : int
            Number of input features.
        params : dict
            Prior parameters (from ``prior_params``).

        Returns
        -------
        SizeCalibratedPrior
        """
        from math import inf

        # Determine base prior key from the prior string
        # "size_calibrated_uniform" -> "uniform", etc.
        base_key = self.prior.replace("size_calibrated_", "")

        if base_key == "bms":
            raise NotImplementedError(
                "Pre-built size-calibrated BMS prior is not yet available. "
                "Use fit_size_calibrated_prior() to build one manually."
            )

        floor_log_prob = params.get("floor_log_prob", -inf)

        # Load pre-built data (validates operators and x_dim)
        corpus_log_hist, log_z_k = load_prebuilt_size_calibrated(
            base_key, self.operators, x_dim
        )

        # Construct the base prior
        if base_key == "uniform":
            base_prior = None
        elif base_key == "katz":
            n = params.get("n", 2)
            corpus = params.get("corpus", "benchmark")
            model = load_katz_model(n=n, corpus=corpus)
            base_prior = KatzPrior(
                model,
                operators=self.operators,
                x_dim=x_dim,
                max_complexity=self.max_complexity,
            )
        else:
            raise ValueError(f"Unsupported base prior: {base_key!r}")

        print(f"Using size-calibrated {base_key} prior.")
        return SizeCalibratedPrior(
            base_prior=base_prior,
            log_z_k=log_z_k,
            corpus_log_hist=corpus_log_hist,
            floor_log_prob=floor_log_prob,
            operators=self.operators,
            x_dim=x_dim,
            max_complexity=self.max_complexity,
        )

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
        prediction = self.best_model_.expression.predict(X)
        return prediction

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
        try:
            score = super().score(X, y, sample_weight=sample_weight)
        except ValueError as e:
            # catch error cause by NaN or inf values in prediction e.g. log(0)
            if "Input contains NaN" in str(e) or "Input contains infinity" in str(e):
                return -np.inf
            raise
        return score

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
