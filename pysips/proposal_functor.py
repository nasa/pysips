import torch
import numpy as np
import os, sys
import requests

from bingo.symbolic_regression.agraph.agraph import AGraph
from playground.bingo_with_fb import encode, decode, get_model
from sympy import *
from pysips import PysipsRegressor


class FunctorPysipsRegressor(PysipsRegressor):
    def __init__(
        self,
        operators=None,
        max_complexity=24,
        terminal_probability=0.1,
        constant_probability=None,
        command_probability=0.2,
        node_probability=0.2,
        parameter_probability=0.2,
        prune_probability=0.2,
        fork_probability=0.2,
        repeat_mutation_probability=0.05,
        crossover_pool_size=None,
        mutation_prob=0.75,
        crossover_prob=0.25,
        exclusive=True,
        num_particles=50,
        num_mcmc_samples=5,
        target_ess=0.8,
        param_init_bounds=None,
        opt_restarts=1,
        model_selection="mode",
        checkpoint_file=None,
        random_state=None,
        max_time=None,
        max_equation_evals=None,
        xvals=None,
    ):
        super().__init__(
            operators,
            max_complexity,
            terminal_probability,
            constant_probability,
            command_probability,
            node_probability,
            parameter_probability,
            prune_probability,
            fork_probability,
            repeat_mutation_probability,
            crossover_pool_size,
            mutation_prob,
            crossover_prob,
            exclusive,
            num_particles,
            num_mcmc_samples,
            target_ess,
            param_init_bounds,
            opt_restarts,
            model_selection,
            checkpoint_file,
            random_state,
            max_time,
            max_equation_evals,
            xvals
        )


    def _get_proposal(self, x_dim, generator):
        return ProposalFunctor(self.xvals)


class ProposalFunctor:
    def __init__(self, xvals):
        self.xvals = xvals
        self.model = get_model()

    def __call__(self, equation, cov):
        x, y = encode(self.xvals, equation)

        # proposal
        np.random.multivariate_normal(np.zeros(cov.shape[0]), cov)
        return decode(self.xvals, y, self.model)

    def update(self, *args, **kwargs):
        pass
