"""
Composite Proposal Generator with Probabilistic Selection.

This module provides a meta-proposal mechanism that probabilistically selects
and applies one or more proposal operators from a collection of available
proposals. It supports both exclusive selection (choosing exactly one proposal)
and non-exclusive selection (choosing multiple proposals to apply sequentially).

This approach allows for flexible proposal strategies in MCMC sampling or
evolutionary algorithms by combining different types of modifications (e.g.,
mutation, crossover, local optimization) with configurable probabilities.

Selection Modes
---------------
Exclusive Mode (default)
    Selects exactly one proposal based on the provided probabilities using
    weighted random selection. The probabilities are automatically normalized
    to sum to the cumulative total.

Non-Exclusive Mode
    Each proposal is independently selected based on its probability. If no
    proposals are selected in a round, the process repeats until at least one
    is chosen. Selected proposals are applied sequentially in random order.

Usage Examples
--------------
Exclusive selection (choose one proposal type):
>>> from mutation import MutationProposal
>>> from crossover import CrossoverProposal
>>>
>>> mutation = MutationProposal(X_dim=3, operators=["+", "*"])
>>> crossover = CrossoverProposal(gene_pool)
>>>
>>> # 70% mutation, 30% crossover
>>> proposal = RandomChoiceProposal(
...     [mutation, crossover],
...     [0.7, 0.3],
...     exclusive=True
... )

Non-exclusive selection (can apply multiple proposals):
>>> # Each proposal has independent 40% chance of being applied
>>> proposal = RandomChoiceProposal(
...     [mutation, crossover, local_optimizer],
...     [0.4, 0.4, 0.2],
...     exclusive=False
... )

Integration Notes
-----------------
The update() method automatically propagates parameter updates to all
constituent proposals, making this class compatible with adaptive sampling
frameworks that modify proposal parameters during execution.

All constituent proposals must implement:
- __call__(model) method for applying the proposal
- update(*args, **kwargs) method for parameter updates (optional)
"""

from bisect import bisect_left
import numpy as np


class RandomChoiceProposal:
    """Randomly choose a proposal to use

    Parameters
    ----------
    proposals : list of proposals
        options for the proposal
    probabilities : list of float
        probabilties of choosing each proposal
    exclusive : bool, optional
        whether the proposals are mutually exclusive or if they can all be
        performed at once, by default True
    seed : int, optional
        random seed used to control repeatability
    """

    def __init__(self, proposals, probabilities, exclusive=True, seed=None):

        self._proposals = proposals
        self._probabilities = probabilities
        self._cum_probabilities = np.cumsum(probabilities)
        self._exclusive = exclusive
        self._rng = np.random.default_rng(seed)

    def _select_proposals(self):
        active_proposals = []

        if self._exclusive:
            rand = self._rng.random() * self._cum_probabilities[-1]
            active_proposals.append(
                self._proposals[bisect_left(self._cum_probabilities, rand)]
            )
            return active_proposals

        while len(active_proposals) == 0:
            for prop, p in zip(self._proposals, self._probabilities):
                if self._rng.random() < p:
                    active_proposals.append(prop)
        self._rng.shuffle(active_proposals)
        return active_proposals

    def __call__(self, model):
        """
        Apply randomly selected proposal(s) to generate a new model.

        This method implements the core functionality of the composite proposal
        generator. It selects one or more proposals based on the configured
        probabilities and selection mode, then applies them sequentially to
        transform the input model.

        Parameters
        ----------
        model : object
            The input model to be transformed. This should be compatible with
            all constituent proposal operators (typically an AGraph for symbolic
            regression or similar structured representation).

        Returns
        -------
        object
            A new model resulting from applying the selected proposal(s).
            The type matches the input model type.

        Process Overview
        ----------------
        1. **Selection Phase**: Randomly selects active proposals based on:
        - Exclusive mode: Exactly one proposal via weighted selection
        - Non-exclusive mode: Zero or more proposals via independent trials

        2. **Application Phase**: Applies selected proposals sequentially:
        - First proposal transforms the original model
        - Subsequent proposals transform the result of previous applications
        - Order is randomized in non-exclusive mode to avoid bias

        Notes
        -----
        - In non-exclusive mode, if no proposals are initially selected, the
        selection process repeats until at least one proposal is chosen
        - Sequential application means later proposals operate on the results
        of earlier ones, potentially creating compound transformations
        """
        active_proposals = self._select_proposals()
        new_model = active_proposals[0](model)
        for prop in active_proposals[1:]:
            new_model = prop(new_model)
        return new_model

    def update(self, *args, **kwargs):
        """
        Propagate parameter updates to all constituent proposals.

        This method forwards update calls to all constituent proposal operators,
        enabling the composite proposal to participate in adaptive sampling
        schemes where proposal parameters are modified during the sampling process.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to each constituent proposal's
            update method. Common examples include new gene pools, population
            statistics, or adaptation parameters.
        **kwargs : dict
            Keyword arguments to be passed to each constituent proposal's
            update method. May include parameters like learning rates,
            temperature schedules, or other adaptive parameters.

        Returns
        -------
        None
            This method modifies the constituent proposals in-place and does
            not return any values.
        """
        for p in self._proposals:
            p.update(*args, **kwargs)
