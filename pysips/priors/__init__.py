"""
Prior distributions for PySIPS.

This subpackage provides prior distribution classes for use in
Sequential Monte Carlo sampling. Available priors:

- ``ImproperUniformPrior``: Generates unique random samples via a custom
  generator function (improper uniform over all valid expressions).
- ``BMSPrior``: Bayesian Machine Scientist prior that scores expressions
  based on weighted operator frequency counts.
- ``KatzPrior``: Katz back-off n-gram prior that scores expressions via
  operator n-gram probabilities over the expression tree.
- ``SamplablePrior``: Abstract base class for priors that support SMC sampling.

Fitting & loading utilities:

- ``fit_bms_prior``: Fit BMS prior weights to match a target corpus.
- ``load_bms_weights``: Load pre-fit BMS weights for a given corpus.
- ``fit_katz_model`` / ``fit_katz_prior``: Fit a Katz model from a corpus.
- ``load_katz_model``: Load a pre-fit Katz model (or fit on demand).
- ``load_corpus``: Load a named corpus of symbolic equations.
"""

from .improper_uniform_prior import ImproperUniformPrior, MAX_REPEATS
from .samplable_prior import SamplablePrior
from .bms_prior import BMSPrior, load_bms_weights
from .bms_fitting import fit_bms_prior
from .katz_backoff import KatzBackoffModel, KatzBackoffTreeModel
from .katz_prior import KatzPrior, load_katz_model, save_katz_model
from .katz_fitting import fit_katz_model, fit_katz_prior
from .data.load_corpus import load_corpus
from .size_calibration_fitting import fit_size_calibrated_prior
from .size_calibrated_prior import SizeCalibratedPrior

__all__ = [
    "ImproperUniformPrior",
    "BMSPrior",
    "SamplablePrior",
    "MAX_REPEATS",
    "load_bms_weights",
    "load_corpus",
    "fit_bms_prior",
    "KatzBackoffModel",
    "KatzBackoffTreeModel",
    "KatzPrior",
    "fit_katz_model",
    "fit_katz_prior",
    "load_katz_model",
    "save_katz_model",
    "fit_size_calibrated_prior",
    "SizeCalibratedPrior",
]
