"""
Mutation-Based Proposal Generator for Symbolic Regression Models.

This module provides a proposal mechanism for symbolic regression that uses
bingo's AGraph mutation operations to generate new candidate models from existing
ones. It is designed to work within Markov Chain Monte Carlo (MCMC) sampling
frameworks where new model proposals are needed at each step.

The module implements a configurable mutation strategy that can perform various
types of structural changes to symbolic mathematical expressions, including
adding/removing nodes, changing operations, modifying parameters, and pruning
or expanding expression trees.

Key Features
------------
- Multiple mutation types: command, node, parameter, prune, and fork mutations
- Configurable probabilities for each mutation type
- Repeat mutation capability for more dramatic changes
- Ensures non-identical proposals (prevents proposing the same model)
- Seeded random number generation for reproducible results
- Integration with bingo's ComponentGenerator for operator management

Mutation Types
--------------
Command Mutation
    Changes the operation at a node (e.g., '+' to '*')
Node Mutation
    Replaces a node with a new randomly generated subtree
Parameter Mutation
    Modifies the numeric constants in the expression
Prune Mutation
    Removes a portion of the expression tree
Fork Mutation
    Adds a new branch to the expression tree
Repeat Mutation
    Recursively applies additional mutations with specified probability

Usage Example
-------------
>>> # Create a mutation proposal generator
>>> proposal = MutationProposal(
...     X_dim=3,  # 3 input features
...     operators=["+", "subtract", "multiply", "divide"],
...     terminal_probability=0.2,
...     command_probability=0.3,
...     node_probability=0.2,
...     seed=42
... )
>>>
>>> # Use in MCMC sampling (assuming you have a model)
>>> # new_model = proposal(current_model)

Notes
-----
The proposal generator ensures that new proposals are always different from
the input model by repeatedly applying mutations until a change occurs. This
prevents MCMC chains from getting stuck with identical consecutive states.

The update() method is provided for compatibility with adaptive MCMC frameworks
but currently performs no operations, as the mutation probabilities are fixed
at initialization.
"""

import numpy as np
from bingo.symbolic_regression import (
    ComponentGenerator,
    AGraphMutation,
)


class MutationProposal:
    """Proposal functor that performs bingo's Agraph mutation

    Parameters
    ----------
    x_dim : int
        dimension of input data (number of features in dataset)
    operators : list of str
        list of equation primatives to allow, e.g. ["+", "subtraction", "pow"]
    terminal_probability : float, optional
        [0.0-1.0] probability that a new node will be a terminal, by default 0.1
    constant_probability : float, optional
        [0.0-1.0] probability that a new terminal will be a constant, by default
        weighted the same as a single feature of the input data
    command_probability : float, optional
        probability of command mutation, by default 0.2
    node_probability : float, optional
        probability of node mutation, by default 0.2
    parameter_probability : float, optional
        probability of parameter mutation, by default 0.2
    prune_probability : float, optional
        probability of pruning (removing a portion of the equation), by default 0.2
    fork_probability : float, optional
        probability of forking (adding an additional branch to the equation),
        by default 0.2
    repeat_mutation_probability : float, optional
        probability of a repeated mutation (applied recursively). default 0.0
    seed : int, optional
        random seed used to control repeatability
    """

    # pylint: disable=R0913,R0917
    def __init__(
        self,
        x_dim,
        operators,
        terminal_probability=0.1,
        constant_probability=None,
        command_probability=0.2,
        node_probability=0.2,
        parameter_probability=0.2,
        prune_probability=0.2,
        fork_probability=0.2,
        repeat_mutation_probability=0.0,
        seed=None,
    ):
        self._rng = np.random.default_rng(seed)

        component_generator = ComponentGenerator(
            input_x_dimension=x_dim,
            terminal_probability=terminal_probability,
            constant_probability=constant_probability,
        )
        for comp in operators:
            component_generator.add_operator(comp)

        self._mutation = AGraphMutation(
            component_generator,
            command_probability,
            node_probability,
            parameter_probability,
            prune_probability,
            fork_probability,
        )
        self._repeat_mutation_prob = repeat_mutation_probability

    def _do_mutation(self, model):
        new_model = self._mutation(model)
        while self._rng.random() < self._repeat_mutation_prob:
            new_model = self._mutation(new_model)
        return new_model

    def __call__(self, model):
        """
        Apply mutation to generate a new symbolic expression model.

        This method takes a symbolic regression model (AGraph) as input and returns
        a new model created by applying one or more mutation operations. The method
        guarantees that the returned model is different from the input model by
        repeating mutations if necessary.

        Parameters
        ----------
        model : AGraph
            The input symbolic regression model to be mutated. This should be a
            bingo AGraph instance representing a mathematical expression.

        Returns
        -------
        AGraph
            A new symbolic regression model created by applying mutation(s) to
            the input model. Guaranteed to be different from the input model.

        Mutation Process
        ---------------
        1. **Initial Mutation**: Applies the configured mutation operation to the model
        2. **Repeat Mutations**: May apply additional mutations based on repeat_mutation_probability
        3. **Difference Check**: Ensures the new model differs from the original one
        4. **Repeated Attempts**: If the mutation produces an identical model, tries again

        Notes
        -----
        - The mutation type applied is selected probabilistically based on the
        probabilities specified during initialization (command_probability,
        node_probability, etc.)
        - The repeat mutation feature allows for more dramatic changes by applying
        multiple mutations in sequence with probability repeat_mutation_probability
        - This method will always return a different model, never the same as the input

        See Also
        --------
        AGraphMutation : Bingo's mutation implementation used internally
        """
        new_model = self._do_mutation(model)
        while new_model == model:
            new_model = self._do_mutation(model)
        return new_model

    def update(self, *args, **kwargs):
        """
        Update method for compatibility with adaptive MCMC frameworks.

        This method is provided to maintain API compatibility with other proposal
        mechanisms that support adaptive behavior. In the current implementation,
        the method is a no-op as the mutation proposal does not adapt its behavior
        based on sampling history.

        Parameters
        ----------
        *args : tuple
            Positional arguments (not used in the current implementation).
        **kwargs : dict
            Keyword arguments (not used in the current implementation).

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        Future versions might implement adaptive behavior such as:
        - Adjusting mutation probabilities based on acceptance rates
        - Learning which mutation types are more effective for a given problem

        In composite proposal mechanisms that combine multiple proposal types
        (such as RandomChoiceProposal), this method will be called as part
        of the update process, but currently has no effect on this proposal.
        """
        # pass
