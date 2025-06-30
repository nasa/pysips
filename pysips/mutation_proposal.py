import numpy as np
from bingo.symbolic_regression import (
    ComponentGenerator,
    AGraphMutation,
)


class MutationProposal:
    """Proposal functor that performs bingo's Agraph mutation

    Parameters
    ----------
    X_dim : int
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

    def __init__(
        self,
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
        seed=None,
    ):
        self._rng = np.random.default_rng(seed)

        component_generator = ComponentGenerator(
            input_x_dimension=X_dim,
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
        new_model = self._do_mutation(model)
        while new_model == model:
            new_model = self._do_mutation(model)
        return new_model

    def update(self, *args, **kwargs):
        pass
