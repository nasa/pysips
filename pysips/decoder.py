"""Latent-to-equation decoders.

In the latent-space formulation of PySIPS, Sequential Monte Carlo samples
points ``z`` in an N-dimensional continuous Latent Space. A *Decoder* is the
map ``g(z)`` that turns a latent point into a symbolic expression (a bingo
``EvolvableExpression``) that can then be scored by the likelihood. Inference
only ever goes latent -> equation, so this is the only direction a Decoder
needs to support.

The real Decoder will be a pretrained neural network that is not yet
available. ``DummyDecoder`` is a deterministic placeholder that emits a fixed
valid expression for every particle, which keeps the decode -> fit -> NMLL
path exercisable end to end while the real model is being built.
"""

from abc import ABC, abstractmethod

import numpy as np

from bingo.expressions import AGraphExpression
from bingo.expressions.agraph.evolvable import EvolvableExpression


class Decoder(ABC):
    """Maps points in the latent space to symbolic expressions.

    A Decoder consumes a batch of latent points and returns one bingo
    ``EvolvableExpression`` per point. Batching matches how SMCPy hands a
    block of particles to the likelihood and lets a real neural decoder run a
    single forward pass.
    """

    @abstractmethod
    def decode(self, latent_points):
        """Decode a batch of latent points into expressions.

        Parameters
        ----------
        latent_points : numpy.ndarray
            Array of shape ``(num_points, n_dims)`` of latent coordinates.

        Returns
        -------
        numpy.ndarray
            Object array of shape ``(num_points,)`` whose entries are bingo
            ``EvolvableExpression`` objects.
        """

    @abstractmethod
    def validate(self, n_dims):
        """Check that the decoder is compatible with a latent dimension.

        Parameters
        ----------
        n_dims : int
            The latent dimension inferred from the prior.

        Raises
        ------
        ValueError
            If the decoder cannot operate on ``n_dims``-dimensional input.
        """


class DummyDecoder(Decoder):
    """Placeholder Decoder that emits a fixed expression for every point.

    This stands in for the real pretrained neural decoder. It is deterministic
    (every latent point maps to the same equation) and accepts any latent
    dimension, so it can validate the latent-space sampling machinery before
    the real decoder exists.

    Parameters
    ----------
    equation : str, optional
        The expression string emitted for every latent point. Default
        ``"X_0"``.
    """

    def __init__(self, equation="X_0"):
        self._equation = equation

    def decode(self, latent_points):
        latent_points = np.atleast_2d(latent_points)
        num_points = latent_points.shape[0]
        expressions = np.empty(num_points, dtype=object)
        for i in range(num_points):
            expressions[i] = EvolvableExpression(
                AGraphExpression(equation=self._equation)
            )
        return expressions

    def validate(self, n_dims):
        return
