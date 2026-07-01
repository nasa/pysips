# Move inference from discrete expression space to a continuous latent space

We are rebuilding PySIPS to run Sequential Monte Carlo over an N-dimensional
continuous **Latent Space** `z` and target `p(y | g(z)) p(z)`, where `M = g(z)`
is produced by a **Decoder** inside a custom likelihood. This replaces the
previous approach of sampling directly over discrete symbolic expressions.

## Why

- It lets us lean on vanilla SMCPy: a scipy-style prior over `z`, SMCPy's native
  `VectorMCMC`/Metropolis kernel with acceptance-rate covariance autotuning, and
  `AdaptiveSampler`. The custom-MCMC hacks (object-valued particles,
  `_compute_cov = False`) disappear because covariance is well-defined over
  continuous latent particles.
- All project-specific logic collapses into one place: a custom likelihood that
  decodes `z`, fits constants, and computes Laplace NMLL as before.

## Scope: this is a branch-scoped experimental trim, not a permanent deletion

This change lives on an exploratory branch whose purpose is to prove out the latent
Decoder + SMC scheme with the smallest possible surface. The discrete-expression
framework is **not being abandoned**: the long-term plan is to reintroduce the genetic
proposals and discrete priors as a *separate latent module that coexists* with the old
code. On this branch we trim aggressively (git history preserves the removed code), and
the LICENSE / NASA copyright notices are retained.

## Consequences

- **Removed on this branch:** `pysips/priors/` (BMS, Katz, size-calibrated, prebuilt
  artifacts, factory), the discrete proposals (`crossover_proposal.py`,
  `mutation_proposal.py`, `random_choice_proposal.py`), and the custom `metropolis.py`.
  `bingo_construction.py` is trimmed to a minimal AGraph helper the `DummyDecoder` uses.
- A future reader will ask "where did the Katz/BMS/size-calibrated priors go?" — they were
  removed *on this branch only* to keep the experiment lean; they remain in history and are
  slated to return in a coexisting latent module.
- The Decoder is a placeholder for now (the real pretrained neural decoder is not ready);
  a `DummyDecoder` stands in so the likelihood/NMLL path stays exercisable end to end.
