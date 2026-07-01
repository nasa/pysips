"""Custom likelihood for latent-space Sequential Monte Carlo.

This module hosts the "custom sauce" of latent-space PySIPS: the pieces that
let a vanilla SMCPy run target ``p(y | g(z)) p(z)`` while the symbolic-
regression scoring stays exactly as before.

The design carries each particle's decoded equation ``M = g(z)`` through SMC by
storing it in the ``log_like`` slot (see ``docs/adr/0002``). Three small
components make this work against native SMCPy:

``LatentLikelihood``
    The ``log_like_func`` bound to SMCPy's ``VectorMCMC``. Given a block of
    latent points it decodes them, fits constants and computes the Laplace
    NMLL for each expression, caches that score on the expression's
    ``fitness`` attribute, and returns the *object array of expressions* (not
    floats).

``EquationGeometricPath``
    A ``GeometricPath`` whose target evaluation reads the cached ``fitness``
    off each expression to recover the scalar log-likelihood before tempering.
    This is the single chokepoint through which both the MCMC accept/reject
    and the weight update consume the likelihood.

``ObjectVectorMCMC``
    A three-line ``VectorMCMC`` subclass that allocates an object-typed buffer
    for proposed log-likelihoods. This is the only place native SMCPy assumes
    a float likelihood; everything else (``np.where`` accept/reject,
    resampling, ``Particles``) is object-safe.
"""

import numpy as np

from smcpy import VectorMCMC
from smcpy.paths import GeometricPath

from .laplace_nmll import LaplaceNmll


def _expressions_to_log_likes(log_like):
    """Convert an object array of expressions into cached NMLL floats.

    Entries are bingo ``EvolvableExpression`` objects carrying their NMLL on
    ``fitness``. ``None`` entries correspond to particles whose proposal landed
    in a zero-prior region; they are given ``0.0`` so the ``-inf`` prior alone
    drives rejection (matching native SMCPy's zero-buffer behavior).
    """
    floats = np.empty(log_like.shape, dtype=float)
    flat_in = log_like.ravel()
    flat_out = floats.ravel()
    for i, expression in enumerate(flat_in):
        flat_out[i] = expression.fitness if expression is not None else 0.0
    return floats


class LatentLikelihood:
    """SMCPy ``log_like_func`` that decodes latent points and scores them.

    Parameters
    ----------
    decoder : Decoder
        Maps a batch of latent points to bingo expressions.
    X : numpy.ndarray
        Training inputs of shape ``(num_datapoints, num_features)``.
    y : numpy.ndarray
        Training targets of shape ``(num_datapoints,)``.
    opt_restarts : int, optional
        Number of constant-optimization restarts for the NMLL. Default 1.
    param_init_bounds : list of float, optional
        ``[low, high]`` bounds for random constant initialization on restarts.

    Notes
    -----
    Calling the instance returns an object array of expressions, each with its
    NMLL cached on ``fitness`` -- not a float array. The expressions ride the
    SMC ``log_like`` slot so they stay aligned with their latent points through
    accept/reject and resampling.
    """

    def __init__(self, decoder, X, y, opt_restarts=1, param_init_bounds=None):
        self._decoder = decoder
        self._nmll = LaplaceNmll(
            X, y, opt_restarts=opt_restarts, param_init_bounds=param_init_bounds
        )

    def __call__(self, latent_points):
        expressions = np.asarray(
            self._decoder.decode(latent_points), dtype=object
        ).ravel()
        for expression in expressions:
            expression.fitness = self._nmll(expression)
        return expressions


class EquationGeometricPath(GeometricPath):
    """Geometric tempering path that scores equation-valued log-likes.

    Identical to ``GeometricPath`` except that the ``log_like`` argument is an
    object array of expressions; their cached ``fitness`` (the NMLL) is read
    out as the scalar log-likelihood before tempering.
    """

    def _eval_target(self, log_like, log_prior, log_p, phi):
        return super()._eval_target(
            _expressions_to_log_likes(log_like), log_prior, log_p, phi
        )


class ObjectVectorMCMC(VectorMCMC):
    """``VectorMCMC`` that carries equation objects in the ``log_like`` slot.

    Native ``VectorMCMC`` allocates a float buffer for newly proposed
    log-likelihoods. This subclass allocates an object buffer instead so the
    decoded expressions returned by ``LatentLikelihood`` survive the MCMC step.
    No other behavior changes.
    """

    def _eval_log_like_if_prior_nonzero(self, log_priors, inputs):
        pos_rows = self._row_has_nonzero_prior_probability(log_priors)
        log_likes = np.empty((log_priors.shape[0], 1), dtype=object)
        if inputs[pos_rows].size != 0:
            log_likes[pos_rows] = self.evaluate_log_likelihood(inputs[pos_rows])
        return log_likes
