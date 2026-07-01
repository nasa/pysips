# Carry decoded equations through SMC in the `log_like` slot

To keep each particle's decoded equation `M` perfectly aligned with its latent point
`z` through MCMC accept/reject and resampling, the custom likelihood returns an **object
array of decoded equations** (each carrying its NMLL on `.fitness`) in SMCPy's `log_like`
slot, rather than a float array of log-likelihoods.

## Why

A particle's state is really the pair `(z, M)`. With a stochastic decoder, `M` cannot be
recovered by re-decoding `z` after the fact (a fresh decode samples `M ~ D(M|z)`, ignoring
the `p(y|M)` reweighting that won acceptance; and even path-consistency is lost). SMCPy has
no native per-particle auxiliary-state channel besides `params` and `log_likes`. Storing the
equation in `log_like` makes it ride `np.where(rejected, log_like, new_log_like)` and
`log_likes[resample_indices]` in lockstep with `z` — no separate bookkeeping.

This is sound because of two facts in SMCPy's source:

- `VectorMCMCKernel.__init__` sets `self._mcmc.evaluate_log_posterior = self.path.logpdf`, so
  the MCMC accept/reject **and** the weight update both funnel through `GeometricPath` — a
  single chokepoint where `log_like` is turned into a number (`log_like * phi`).
- `Particles` never does arithmetic on `log_likes` (ESS/covariance/mean use `params` and
  `weights` only), so an object-typed `log_likes` passes through untouched.

## Consequences

- A custom `GeometricPath` subclass overrides `_eval_target` to read each equation's cached
  `.fitness` (the NMLL) before multiplying by `phi`. The NMLL is computed once at decode/fit
  time and cached — never recomputed in the path (which also keeps `log_like` deterministic).
- A 3-line `VectorMCMC` subclass overrides `_eval_log_like_if_prior_nonzero` to allocate an
  object buffer (`np.empty((P, 1), dtype=object)`) instead of `np.zeros`. This is the only
  SMCPy method that assumes a float likelihood buffer. It is *not* a return to the old custom
  MCMC logic — it is a dtype fix, nothing more.
- Final outputs come directly from `final_step.log_likes`: the equations are the array, and
  their NMLL values are read from `.fitness`.
- A future reader will be surprised that `log_like` holds equation objects; this ADR records
  that it is deliberate and load-bearing.
