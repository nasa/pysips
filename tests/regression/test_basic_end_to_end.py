import argparse
from pathlib import Path
import numpy as np
import h5py

from pysips.laplace_nmll import LaplaceNmll
from pysips.mutation_proposal import MutationProposal
from pysips.crossover_proposal import CrossoverProposal
from pysips.random_choice_proposal import RandomChoiceProposal
from pysips.sampler import sample

from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator


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
    crossover = CrossoverProposal(list(pool))

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
    USE_PYTHON = True
    USE_SIMPLIFICATION = True
    component_generator = ComponentGenerator(
        input_x_dimension=X_dim,
        terminal_probability=terminal_probability,
        constant_probability=constant_probability,
    )
    for comp in operators:
        component_generator.add_operator(comp)
    generator = AGraphGenerator(
        max_complexity,
        component_generator,
        use_python=USE_PYTHON,
        use_simplification=USE_SIMPLIFICATION,
    )

    return generator


def test_basic_end_to_end():

    n_pts = 21
    X = np.c_[np.linspace(0, 2 * np.pi, n_pts)]
    y = (np.sin(X) * 2 + 4).flatten() + np.random.default_rng(34).normal(0, 0.5, n_pts)

    config = {
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

    likelihood = LaplaceNmll(X, y)
    generator = get_generator(**config)
    proposal = get_proposal(**config)
    models, likelihoods = sample(
        likelihood,
        proposal,
        generator,
        seed=34,
        kwargs={
            "num_particles": config["num_particles"],
            "num_mcmc_samples": config["num_mcmc_samples"],
            "target_ess": config["target_ess"],
        },
    )
