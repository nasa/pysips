"""Shared bingo expression generator and proposal construction.

This module centralizes the construction of bingo AGraph generators and
proposal operators so Prior Resolution and sampling setup can share one
internal path.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from bingo.expressions import ComponentGenerator, AGraphGenerator

from .crossover_proposal import CrossoverProposal
from .mutation_proposal import MutationProposal
from .random_choice_proposal import RandomChoiceProposal


@dataclass(frozen=True)
class BingoConstructionConfig:
    """Immutable bingo expression-generation and proposal settings."""

    max_complexity: int = 24
    terminal_probability: float = 0.1
    constant_probability: Optional[float] = None
    command_probability: float = 0.2
    node_probability: float = 0.2
    parameter_probability: float = 0.2
    prune_probability: float = 0.2
    fork_probability: float = 0.2
    repeat_mutation_probability: float = 0.05
    crossover_pool_size: Optional[int] = None
    mutation_prob: float = 0.75
    crossover_prob: float = 0.25
    exclusive: bool = True


def _resolve_constant_probability(
    x_dim: int, constant_probability: Optional[float]
) -> float:
    if constant_probability is None:
        return 1 / (x_dim + 1)
    return constant_probability


def build_agraph_generator(
    x_dim: int,
    operators: Sequence[int],
    bingo_config: BingoConstructionConfig,
):
    """Create a configured bingo AGraph generator."""
    constant_probability = _resolve_constant_probability(
        x_dim, bingo_config.constant_probability
    )
    component_generator = ComponentGenerator(
        input_x_dimension=x_dim,
        terminal_probability=bingo_config.terminal_probability,
        constant_probability=constant_probability,
    )
    for operator in operators:
        component_generator.add_operator(operator)

    return AGraphGenerator(
        bingo_config.max_complexity,
        bingo_config.max_complexity,
        component_generator,
    )


def build_agraph_proposal(
    x_dim: int,
    operators: Sequence[int],
    generator,
    bingo_config: BingoConstructionConfig,
    *,
    default_crossover_pool_size: int = 50,
):
    """Create a combined mutation/crossover proposal operator."""
    constant_probability = _resolve_constant_probability(
        x_dim, bingo_config.constant_probability
    )
    mutation = MutationProposal(
        x_dim,
        operators=operators,
        terminal_probability=bingo_config.terminal_probability,
        constant_probability=constant_probability,
        command_probability=bingo_config.command_probability,
        node_probability=bingo_config.node_probability,
        parameter_probability=bingo_config.parameter_probability,
        prune_probability=bingo_config.prune_probability,
        fork_probability=bingo_config.fork_probability,
        repeat_mutation_probability=bingo_config.repeat_mutation_probability,
    )

    pool_size = bingo_config.crossover_pool_size
    if pool_size is None:
        pool_size = default_crossover_pool_size

    pool = set()
    consecutive_failures = 0
    while len(pool) < pool_size:
        previous_size = len(pool)
        pool.add(generator())
        if len(pool) == previous_size:
            consecutive_failures += 1
            if consecutive_failures >= 100:
                break
        else:
            consecutive_failures = 0

    crossover = CrossoverProposal(
        list(pool),
        agraph_size=bingo_config.max_complexity,
    )
    return RandomChoiceProposal(
        [mutation, crossover],
        [bingo_config.mutation_prob, bingo_config.crossover_prob],
        bingo_config.exclusive,
    )
