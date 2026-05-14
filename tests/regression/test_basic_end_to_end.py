import argparse
from pathlib import Path
import numpy as np
import h5py
import pytest

from pysips.laplace_nmll import LaplaceNmll
from pysips.mutation_proposal import MutationProposal
from pysips.crossover_proposal import CrossoverProposal
from pysips.random_choice_proposal import RandomChoiceProposal
from pysips.priors import ImproperUniformPrior, BMSPrior
from pysips.sampler import sample

from bingo.expressions import ComponentGenerator, AGraphGenerator


def get_proposal(
    X_dim,
    operators,
    terminal_probability=0.1,
    constant_probability=None,
    command_probability=0.2,
    node_probability=0.2,
    parameter_probability=0.2,
    prune_probability=0.2,
    fork_probability=0.2,
    repeat_mutation_probability=0.0,
    crossover_pool_size=500,
    mutation_prob=0.5,
    crossover_prob=0.5,
    exclusuive=True,
    max_complexity=48,
    **kwargs,
):
    generator = get_generator(
        X_dim, operators, terminal_probability, constant_probability, max_complexity
    )

    mutation = MutationProposal(
        X_dim,
        operators=operators,
        terminal_probability=terminal_probability,
        constant_probability=constant_probability,
        command_probability=command_probability,
        node_probability=node_probability,
        parameter_probability=parameter_probability,
        prune_probability=prune_probability,
        fork_probability=fork_probability,
        repeat_mutation_probability=repeat_mutation_probability,
    )

    pool = set()
    while len(pool) < crossover_pool_size:
        pool.add(generator())
    crossover = CrossoverProposal(list(pool), agraph_size=max_complexity)

    proposal = RandomChoiceProposal(
        [mutation, crossover], [mutation_prob, crossover_prob], exclusuive
    )

    return proposal


def get_generator(
    X_dim,
    operators,
    terminal_probability=0.1,
    constant_probability=None,
    max_complexity=48,
    **kwargs,
):
    component_generator = ComponentGenerator(
        input_x_dimension=X_dim,
        terminal_probability=terminal_probability,
        constant_probability=constant_probability,
    )
    for comp in operators:
        component_generator.add_operator(comp)
    generator = AGraphGenerator(
        max_complexity,
        max_complexity,
        component_generator,
    )

    return generator


@pytest.fixture
def test_data():
    n_pts = 21
    X = np.c_[np.linspace(0, 2 * np.pi, n_pts)]
    y = (np.sin(X) * 2 + 4).flatten() + np.random.default_rng(34).normal(0, 0.5, n_pts)
    return X, y


@pytest.fixture
def config(test_data):
    X, y = test_data
    return {
        "X_dim": X.shape[1],
        "constant_probability": 1 / (X.shape[1] + 1),
        "operators": ["+", "*"],
        "param_init_bounds": [-5, 5],
        "opt_restarts": 1,
        "terminal_probability": 0.1,
        "command_probability": 0.2,
        "node_probability": 0.2,
        "parameter_probability": 0.2,
        "prune_probability": 0.2,
        "fork_probability": 0.2,
        "repeat_mutation_probability": 0.05,
        "crossover_pool_size": 50,
        "mutation_prob": 0.75,
        "crossover_prob": 0.25,
        "exclusuive": True,
        "max_complexity": 24,
        "num_particles": 50,
        "num_mcmc_samples": 5,
        "target_ess": 0.8,
    }


@pytest.fixture
def generator(config):
    return get_generator(**config)


@pytest.fixture
def proposal(config):
    return get_proposal(**config)


@pytest.fixture
def likelihood(test_data):
    X, y = test_data
    return LaplaceNmll(X, y)


def test_basic_end_to_end_improper_uniform_prior(
    generator, likelihood, proposal, config
):
    prior = ImproperUniformPrior(generator)
    models, likelihoods, phis = sample(
        likelihood,
        proposal,
        prior,
        seed=34,
        kwargs={
            "num_particles": config["num_particles"],
            "num_mcmc_samples": config["num_mcmc_samples"],
            "target_ess": config["target_ess"],
        },
    )


def test_basic_end_to_end_bms_prior(likelihood, proposal, test_data, config):
    X, _ = test_data
    prior = BMSPrior(
        weights={2: 1.0, 4: 0.5},
        squared_weights={2: 1.0, 4: 0.25},
        x_dim=X.shape[1],
    )
    models, likelihoods, phis = sample(
        likelihood,
        proposal,
        prior,
        seed=34,
        kwargs={
            "num_particles": config["num_particles"],
            "num_mcmc_samples": config["num_mcmc_samples"],
            "target_ess": config["target_ess"],
        },
    )
