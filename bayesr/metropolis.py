import numpy as np

from multiprocessing import Pool
from smcpy import VectorMCMC, ImproperUniform
from tqdm import tqdm


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

        for i in range(num_samples):

            inputs, log_like, _, _ = self._perform_mcmc_step(
                inputs, None, log_like, log_priors
            )

        self._equ_proposal.update(gene_pool=inputs.flatten())

        return inputs, log_like

    def evaluate_model(self):
        return None

    def evaluate_log_priors(self, x):
        return np.ones((x.shape[0], 1))

    def evaluate_log_likelihood(self, x):
        if self._is_multiprocess:
            with Pool() as p:
                log_like = p.map(self._log_like_func, x.flatten())
            for l, x in zip(log_like, x.flatten()):
                x.fitness = l
        else:
            log_like = [self._log_like_func(xi) for xi in x.flatten()]
        return np.c_[log_like]
