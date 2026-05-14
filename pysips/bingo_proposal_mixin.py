"""
Bingo Proposal Mixin for SMC Sampling.

This module provides a mixin class that encapsulates the shared logic for
creating AGraph generators and proposal operators used in Sequential Monte
Carlo sampling. It is designed to be inherited by classes that need SMC
sampling capabilities with bingo symbolic expressions.

The mixin provides:
- Generator creation via ComponentGenerator and AGraphGenerator
- Proposal creation combining MutationProposal and CrossoverProposal
- Shared parameter definitions for expression generation and mutation

Example
-------
>>> class MyClass(BingoProposalMixin):
...     def __init__(self):
...         super().__init__()  # Sets all default parameters
...
...     def sample_expressions(self, x_dim, operators):
...         generator = self._get_generator(x_dim, operators)
...         proposal = self._get_proposal(x_dim, generator, operators)
...         # Use generator and proposal for SMC sampling...
"""

from typing import List, Optional

from bingo.expressions import ComponentGenerator, AGraphGenerator

from .mutation_proposal import MutationProposal
from .crossover_proposal import CrossoverProposal
from .random_choice_proposal import RandomChoiceProposal


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-few-public-methods
class BingoProposalMixin:
    """
    Mixin providing bingo generator and proposal creation for SMC sampling.

    This mixin encapsulates the shared logic for creating AGraph expression
    generators and proposal operators. Classes that need to sample symbolic
    expressions via SMC can inherit from this mixin to avoid code duplication.

    Parameters
    ----------
    max_complexity : int, optional
        Maximum AGraph complexity (number of nodes). Default is 24.
    terminal_probability : float, optional
        Probability of generating terminal nodes. Default is 0.1.
    constant_probability : float, optional
        Probability of generating constants. If None, computed as 1/(x_dim+1).
    command_probability : float, optional
        Mutation probability for command changes. Default is 0.2.
    node_probability : float, optional
        Mutation probability for node changes. Default is 0.2.
    parameter_probability : float, optional
        Mutation probability for parameter changes. Default is 0.2.
    prune_probability : float, optional
        Mutation probability for pruning. Default is 0.2.
    fork_probability : float, optional
        Mutation probability for forking. Default is 0.2.
    repeat_mutation_probability : float, optional
        Probability of applying multiple mutations. Default is 0.05.
    crossover_pool_size : int, optional
        Size of crossover pool. If None, defaults to num_particles or 50.
    mutation_prob : float, optional
        Probability of using mutation proposal. Default is 0.75.
    crossover_prob : float, optional
        Probability of using crossover proposal. Default is 0.25.
    exclusive : bool, optional
        If True, mutation and crossover are exclusive. Default is True.

    Notes
    -----
    This mixin expects `x_dim` and `operators` to be provided when calling
    `_get_generator()` and `_get_proposal()`, allowing flexibility in how
    these values are determined (e.g., from data dimensions, weight keys).
    """

    def __init__(
        self,
        max_complexity: int = 24,
        terminal_probability: float = 0.1,
        constant_probability: Optional[float] = None,
        command_probability: float = 0.2,
        node_probability: float = 0.2,
        parameter_probability: float = 0.2,
        prune_probability: float = 0.2,
        fork_probability: float = 0.2,
        repeat_mutation_probability: float = 0.05,
        crossover_pool_size: Optional[int] = None,
        mutation_prob: float = 0.75,
        crossover_prob: float = 0.25,
        exclusive: bool = True,
        **kwargs,
    ):
        # Pass remaining kwargs to next class in MRO (for cooperative inheritance)
        super().__init__(**kwargs)

        self.max_complexity = max_complexity
        self.terminal_probability = terminal_probability
        self.constant_probability = constant_probability
        self.command_probability = command_probability
        self.node_probability = node_probability
        self.parameter_probability = parameter_probability
        self.prune_probability = prune_probability
        self.fork_probability = fork_probability
        self.repeat_mutation_probability = repeat_mutation_probability
        self.crossover_pool_size = crossover_pool_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.exclusive = exclusive

    def _get_generator(self, x_dim: int, operators: List[int]):
        """
        Create an AGraph expression generator.

        Parameters
        ----------
        x_dim : int
            Number of input variables (dimension of X data).
        operators : list of int
            Operator IDs to use in expression generation.

        Returns
        -------
        AGraphGenerator
            Configured generator for creating random AGraph expressions.
        """
        constant_prob = self.constant_probability
        if constant_prob is None:
            constant_prob = 1 / (x_dim + 1)

        component_generator = ComponentGenerator(
            input_x_dimension=x_dim,
            terminal_probability=self.terminal_probability,
            constant_probability=constant_prob,
        )
        for op in operators:
            component_generator.add_operator(op)

        return AGraphGenerator(
            self.max_complexity,
            self.max_complexity,
            component_generator,
        )

    def _get_proposal(
        self,
        x_dim: int,
        generator: AGraphGenerator,
        operators: List[int],
    ) -> RandomChoiceProposal:
        """
        Create a combined mutation/crossover proposal operator.

        Parameters
        ----------
        x_dim : int
            Number of input variables (dimension of X data).
        generator : AGraphGenerator
            Expression generator used to create crossover pool.
        operators : list of int
            Operator IDs to use in mutation operations.

        Returns
        -------
        RandomChoiceProposal
            Combined proposal that randomly selects between mutation
            and crossover operations.
        """
        constant_prob = self.constant_probability
        if constant_prob is None:
            constant_prob = 1 / (x_dim + 1)

        mutation = MutationProposal(
            x_dim,
            operators=operators,
            terminal_probability=self.terminal_probability,
            constant_probability=constant_prob,
            command_probability=self.command_probability,
            node_probability=self.node_probability,
            parameter_probability=self.parameter_probability,
            prune_probability=self.prune_probability,
            fork_probability=self.fork_probability,
            repeat_mutation_probability=self.repeat_mutation_probability,
        )

        # Determine crossover pool size
        pool_size = self.crossover_pool_size
        if pool_size is None:
            # Default to num_particles if available, otherwise 50
            pool_size = getattr(self, "num_particles", 50)

        # Generate crossover pool
        pool = set()
        consecutive_failures = 0
        while len(pool) < pool_size:
            prev_size = len(pool)
            pool.add(generator())
            if len(pool) == prev_size:
                consecutive_failures += 1
                if consecutive_failures >= 100:
                    # Generator cannot produce more unique models; use what we have.
                    break
            else:
                consecutive_failures = 0
        crossover = CrossoverProposal(
            list(pool),
            agraph_size=self.max_complexity,
        )

        # Create combined proposal
        return RandomChoiceProposal(
            [mutation, crossover],
            [self.mutation_prob, self.crossover_prob],
            self.exclusive,
        )
