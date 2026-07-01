"""
PySIPS: Python package for Symbolic Inference via Posterior Sampling

This module provides a scikit-learn compatible interface for symbolic
regression using Sequential Monte Carlo (SMC) sampling in an N-dimensional
continuous latent space.

Methodology
-----------
Inference is vanilla SMC in a continuous latent space, with all
symbolic-regression specifics hidden inside a custom likelihood:

1. **Latent prior**: A scipy-style prior defines the continuous latent space
   and is passed straight through to smcpy.

2. **Decoding**: Each latent particle is decoded into a symbolic expression by
   a (pre-trained) decoder. A placeholder :class:`~pysips.decoder.DummyDecoder`
   is used until a trained decoder is available.

3. **Likelihood Evaluation**: Each decoded expression is scored with a Laplace
   approximation to the normalized marginal likelihood (NMLL) for Bayesian
   model comparison. The decoded expression objects ride through SMC in the
   ``log_like`` slot so that each final particle's equation is exactly the one
   that gated its acceptance.

4. **SMC Sampling**: smcpy's native adaptive sampler draws from the posterior,
   auto-tuning the proposal covariance from the acceptance rate.

5. **Model Selection**: The final model is chosen by either mode selection
   (most frequently sampled expression) or maximum NMLL.

Usage Example
-------------
>>> from scipy.stats import norm
>>> from pysips import PysipsRegressor
>>> import numpy as np
>>>
>>> X = np.random.randn(100, 2)
>>> y = X[:, 0] ** 2 + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)
>>>
>>> prior = [norm(0, 1) for _ in range(8)]  # 8-D latent space
>>> regressor = PysipsRegressor(
...     prior=prior,
...     num_particles=100,
...     model_selection="mode",
...     random_state=42,
... )
>>> regressor.fit(X, y)
>>> y_pred = regressor.predict(X)
>>> expression = regressor.get_expression()
>>> models, likelihoods = regressor.get_models()

Notes
-----
Checkpointing allows resuming interrupted long-running fits by saving
incremental progress to a pickle file.
"""

from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .decoder import DummyDecoder
from .likelihood import LatentLikelihood
from .sampler import sample, _infer_latent_dim

DEFALT_PARAMETER_INITIALIZATION_BOUNDS = [-5, 5]


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals
class PysipsRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for PySIPS symbolic regression.

    This regressor performs Bayesian symbolic regression by running Sequential
    Monte Carlo (SMC) sampling in an N-dimensional continuous latent space.
    Each latent particle is decoded into a symbolic expression and scored with
    a Laplace approximation to the normalized marginal likelihood, yielding a
    posterior distribution over expressions with principled uncertainty
    quantification and support for checkpointing.

    Parameters
    ----------
    prior : sequence of scipy-style distributions
        Prior over the continuous latent space, passed straight through to
        smcpy. Its length (summed over each distribution's ``dim``) defines the
        latent dimension N. Required.

    decoder : object or None, default=None
        Decoder mapping latent points to symbolic expressions. Must implement
        ``decode(latent_points)`` returning an object array of expressions and
        ``validate(n_dims)``. If None, a :class:`~pysips.decoder.DummyDecoder`
        placeholder is used.

    num_particles : int, default=50
        Number of particles for sampling.

    num_mcmc_samples : int, default=5
        Number of MCMC samples per SMC step.

    target_ess : float, default=0.8
        Target effective sample size.

    param_init_bounds : list, default=[-5, 5]
        Bounds for constant initialization during NMLL fitting.

    opt_restarts : int, default=1
        Number of optimization restarts during NMLL fitting.

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

    show_progress_bar : bool, default=True
        Whether to display a progress bar during fitting. When False, the
        progress bar will be hidden, which is useful for hyperparameter
        tuning or when running multiple fits in parallel.
    """

    def __init__(
        self,
        prior=None,
        decoder=None,
        num_particles=50,
        num_mcmc_samples=5,
        target_ess=0.8,
        param_init_bounds=None,
        opt_restarts=1,
        model_selection="mode",
        checkpoint_file=None,
        random_state=None,
        max_time=None,
        max_equation_evals=None,
        show_progress_bar=True,
    ):
        # Validate that max_time and max_equation_evals are not both specified
        if max_time is not None and max_equation_evals is not None:
            raise ValueError(
                "max_time and max_equation_evals cannot both be specified. "
                "Please choose one constraint method."
            )

        self.prior = prior
        self.decoder = decoder
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

        if self.prior is None:
            raise ValueError(
                "A latent-space prior must be provided. Pass a sequence of "
                "scipy-style distributions whose length defines the latent "
                "dimension N."
            )

        decoder = self.decoder if self.decoder is not None else DummyDecoder()

        # The prior defines the latent dimension; make sure the decoder agrees.
        n_latent = _infer_latent_dim(self.prior)
        decoder.validate(n_latent)

        # The likelihood decodes each latent point and scores it via Laplace NMLL.
        likelihood = LatentLikelihood(
            decoder,
            X,
            y,
            opt_restarts=self.opt_restarts,
            param_init_bounds=self.param_init_bounds,
        )

        # Run sampling
        models, likelihoods, phis = sample(
            likelihood,
            self.prior,
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
