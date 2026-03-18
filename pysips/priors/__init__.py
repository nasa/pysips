"""
Prior distributions for PySIPS.

This subpackage provides prior distribution classes for use in
Sequential Monte Carlo sampling. Available priors:

- ``ImproperUniformPrior``: Generates unique random samples via a custom
  generator function (improper uniform over all valid expressions).
- ``BMSPrior``: Bayesian Machine Scientist prior that scores expressions
  based on weighted operator frequency counts.
- ``SamplablePrior``: Abstract base class for priors that support SMC sampling.

Fitting utilities:

- ``fit_bms_prior``: Fit BMS prior weights to match a target corpus of expressions.
"""

from .improper_uniform_prior import ImproperUniformPrior, MAX_REPEATS
from .samplable_prior import SamplablePrior
from .bms_prior import BMSPrior, DEFAULT_BMS_WEIGHTS, DEFAULT_BMS_SQUARED_WEIGHTS
from .bms_fitting import fit_bms_prior
from .data.load_corpus import load_corpus

__all__ = [
    "ImproperUniformPrior",
    "BMSPrior",
    "SamplablePrior",
    "MAX_REPEATS",
    "DEFAULT_BMS_WEIGHTS",
    "DEFAULT_BMS_SQUARED_WEIGHTS",
    "load_corpus",
    "fit_bms_prior",
]
