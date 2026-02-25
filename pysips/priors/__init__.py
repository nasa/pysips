"""
Prior distributions for PySIPS.

This subpackage provides prior distribution classes for use in
Sequential Monte Carlo sampling. Available priors:

- ``ImproperUniformPrior``: Generates unique random samples via a custom
  generator function (improper uniform over all valid expressions).
- ``BMSPrior``: Bayesian Machine Scientist prior that scores expressions
  based on weighted operator frequency counts.
- ``SamplablePrior``: Abstract base class for priors that support SMC sampling.
"""

from .improper_uniform_prior import ImproperUniformPrior, MAX_REPEATS
from .samplable_prior import SamplablePrior
from .bms_prior import BMSPrior
from .data.load_corpus import load_corpus

__all__ = [
    "ImproperUniformPrior",
    "BMSPrior",
    "SamplablePrior",
    "MAX_REPEATS",
    "load_corpus",
]
