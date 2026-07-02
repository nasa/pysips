"""
Latent-space decoder workflow example.

This script is a guide implementing the real neural decoder.
It demonstrates the full pipeline end-to-end using the DummyDecoder as a
stand-in, so the decode -> score -> SMC path is exercisable before the real
model exists.  The only object that needs to be swapped out is ``decoder``.
"""

import numpy as np
from smcpy import ImproperUniform

from pysips import Decoder, DummyDecoder, PysipsRegressor

# ---------------------------------------------------------------------------
# 1. Training data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 80).reshape(-1, 1)
y = X[:, 0] ** 2 + rng.normal(0, 0.2, size=X.shape[0])


# ---------------------------------------------------------------------------
# 2. Prior over the latent space
# ---------------------------------------------------------------------------
LATENT_DIM = 4
prior = [ImproperUniform() for _ in range(LATENT_DIM)]


# ---------------------------------------------------------------------------
# 3. Decoder
#
# DummyDecoder is a deterministic placeholder: every latent point maps to
# the same fixed expression.  It exists so the rest of the pipeline can be
# validated before the real model is ready.
#
# --- What you need to implement ---
#
# Subclass pysips.Decoder and implement two methods:
#
#   decode(latent_points: np.ndarray) -> np.ndarray
#       latent_points has shape (num_particles, LATENT_DIM).
#       Return an object array of shape (num_particles,) whose entries are
#       bingo EvolvableExpression objects.  One forward pass of your neural
#       network should cover the entire batch.
#
#   validate(n_dims: int) -> None
#       Called before sampling begins.  Raise ValueError if your model
#       cannot handle n_dims-dimensional input (e.g. the checkpoint was
#       trained on a different embedding size).
#
# ---------------------------------------------------------------------------
decoder = DummyDecoder(equation="X_0")


# ---------------------------------------------------------------------------
# 4. Regressor
#
# Pass the prior and decoder directly to PysipsRegressor.  Internally it
# builds the LatentLikelihood, validates the decoder against the latent
# dimension inferred from the prior, and runs SMC sampling.
#
#   num_particles    -- population size (increase for better exploration)
#   num_mcmc_samples -- MCMC chain length per SMC step
#   max_time         -- wall-time cap in seconds (use during development)
#   random_state     -- seed for reproducibility
# ---------------------------------------------------------------------------
regressor = PysipsRegressor(
    prior=prior,
    decoder=decoder,
    num_particles=100,
    num_mcmc_samples=10,
    max_time=30,
    random_state=42,
)


# ---------------------------------------------------------------------------
# 5. Fit and inspect results
# ---------------------------------------------------------------------------
regressor.fit(X, y)

expression = regressor.get_expression()
models, log_likes = regressor.get_models()

print(f"Best expression : {expression}")
print(f"Best log-like   : {regressor.best_likelihood_:.4f}")
print(f"SMC steps       : {len(regressor.phis_) - 1}")
print(f"Final phi       : {regressor.phis_[-1]:.4f}")
