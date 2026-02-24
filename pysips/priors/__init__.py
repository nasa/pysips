"""
Prior distributions for PySIPS.

This subpackage provides prior distribution classes for use in
Sequential Monte Carlo sampling. Available priors:

- ``ImproperUniformPrior``: Generates unique random samples via a custom
  generator function (improper uniform over all valid expressions).
- ``BMSPrior``: Bayesian Machine Scientist prior that scores expressions
  based on weighted operator frequency counts.
"""

from .improper_uniform_prior import ImproperUniformPrior, MAX_REPEATS
from .bms_prior import BMSPrior

__all__ = ["ImproperUniformPrior", "BMSPrior", "MAX_REPEATS"]
