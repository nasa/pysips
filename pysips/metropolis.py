"""
Metropolis-Hastings MCMC Implementation for Symbolic Regression.

This module provides a specialized Metropolis-Hastings Markov Chain Monte Carlo
(MCMC) sampler designed for symbolic regression models. It extends the smcpy
VectorMCMC class to handle symbolic expressions (bingo AGraph objects) as parameters,
with custom proposal mechanisms and likelihood evaluation for equation discovery.

The implementation supports both single-process and multiprocess likelihood
evaluation, making it suitable for computationally intensive symbolic regression
tasks where model evaluation is the computational bottleneck.

Algorithm Overview
------------------
The Metropolis algorithm follows the standard accept/reject framework:

1. **Proposal Generation**: Uses a provided proposal function to generate
   new symbolic expressions from current ones

2. **Likelihood Evaluation**: Computes log-likelihood for proposed expressions
   using the provided likelihood function

3. **Accept/Reject Decision**: Accepts or rejects proposals based on the
   Metropolis criterion comparing likelihoods

4. **Chain Evolution**: Iteratively builds a Markov chain of symbolic
   expressions that converges to the target distribution

The key adaptation for symbolic regression is handling discrete, structured
parameter spaces (symbolic expressions) rather than continuous parameters.

Example Integration
-------------------
>>> from bingo.symbolic_regression import AGraph
>>>
>>> def likelihood_func(model):
...     # Evaluate model on data and return log-likelihood
...     return model.evaluate_fitness_vector(X, y)
>>>
>>> def proposal_func(model):
...     # Generate new model via mutation
...     return mutate(model)
>>>
>>> mcmc = Metropolis(
...     likelihood=likelihood_func,
...     proposal=proposal_func,
...     prior=uniform_prior,
...     multiprocess=True
... )

Implementation Notes
--------------------
- Uniform priors are assumed (evaluate_log_priors returns ones)
- Proposal updates are called after each sampling round to maintain
  an adaptive gene pool for crossover operations
- Fitness values are cached on AGraph objects to avoid redundant computation
- The implementation handles vectorized operations for efficiency
"""

from multiprocessing import Pool
import numpy as np
from smcpy import VectorMCMC


class Metropolis(VectorMCMC):
    """Class for running basic MCMC w/ the Metropolis algorithm

    Parameters
    ----------
    likelihood : callable
        Computes marginal log likelihood given a bingo AGraph
    proposal : callable
        Proposes a new AGraph conditioned on an existing AGraph; must be
        symmetric.
    """

    def __init__(self, likelihood, proposal, prior, multiprocess=False):
        super().__init__(
            model=None,
            data=None,
            priors=[prior],
            log_like_func=lambda *x: likelihood,
            log_like_args=None,
        )
        self._equ_proposal = proposal
        self.proposal = lambda x, z: np.array(
            [[self._equ_proposal(xi)] for xi in x.flatten()]
        )
        self._is_multiprocess = multiprocess

    def smc_metropolis(self, inputs, num_samples, cov=None):
        """
        Parameters
        ----------
        model : AGraph
            model at which Markov chain initiates
        num_samples : int
            number of samples in the chain; includes burnin
        """
        log_priors, log_like = self._initialize_probabilities(inputs)

        for _ in range(num_samples):

            inputs, log_like, _, _ = self._perform_mcmc_step(
                inputs, None, log_like, log_priors
            )

        self._equ_proposal.update(gene_pool=inputs.flatten())

        return inputs, log_like

    def evaluate_model(self, _=None):
        return None

    def evaluate_log_priors(self, inputs):
        return np.ones((inputs.shape[0], 1))

    def evaluate_log_likelihood(self, inputs):
        if self._is_multiprocess:
            with Pool() as p:
                log_like = p.map(self._log_like_func, inputs.flatten())
            for l, xi in zip(log_like, inputs.flatten()):
                xi.fitness = l
        else:
            log_like = [self._log_like_func(xi) for xi in inputs.flatten()]
        return np.c_[log_like]
