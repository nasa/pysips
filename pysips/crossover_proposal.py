"""
Crossover-Based Proposal Generator for Symbolic Regression Models.

This module provides a crossover-based proposal mechanism for symbolic regression
that creates new candidate models by combining genetic material from existing
models. It implements genetic programming crossover operations using bingo's
AGraphCrossover functionality within an MCMC or evolutionary algorithm framework.

The crossover operation mimics biological reproduction by exchanging subtrees
between two parent expressions to create offspring that inherit characteristics
from both parents. This approach can effectively explore the space of symbolic
expressions by combining successful components from different models.

Key Features
------------
- Random partner selection from a configurable gene pool
- Stochastic child selection (50/50 probability between two crossover offspring)
- Avoids self-crossover by ensuring different parent selection
- Updateable gene pool for adaptive sampling strategies
- Seeded random number generation for reproducible results

Crossover Mechanism
-------------------
The crossover operation works by:
1. Selecting a random crossover point in each parent expression tree
2. Swapping the subtrees at those points between the two parents
3. Producing two offspring that combine features from both parents
4. Randomly selecting one of the two offspring as the proposal

This process allows successful expression fragments to be preserved and
recombined in novel ways, potentially discovering better solutions through
the exploration of hybrid models.

Usage Example
-------------
>>> # Assume you have a collection of symbolic models
>>> gene_pool = [model1, model2, model3, model4]  # List of AGraph models
>>>
>>> # Create crossover proposal generator
>>> crossover = CrossoverProposal(gene_pool, seed=42)
>>>
>>> # Use in MCMC or evolutionary sampling
>>> current_model = model1
>>> new_proposal = crossover(current_model)
>>>
>>> # Update gene pool as better models are discovered
>>> updated_pool = [best_model1, best_model2, new_good_model]
>>> crossover.update(updated_pool)

Integration Notes
-----------------
The update() method allows for dynamic gene pool management, enabling
adaptive strategies where successful models from the sampling process
can be added to influence future proposals.
"""

import numpy as np
from bingo.symbolic_regression import (
    AGraphCrossover,
)


class CrossoverProposal:
    """A proposal operator that performs crossover between AGraph models.

    This class implements a callable object that creates new models by performing
    crossover operations between an input model and randomly selected partners
    from a gene pool. It utilizes bingo's AGraphCrossover mechanism and randomly
    selects one of the two children produced by each crossover operation.

    Parameters
    ----------
    gene_pool : list of AGraph
        A collection of AGraph models that will be used as potential partners
        during crossover operations
    seed : int, optional
        Random seed for the internal random number generator, used to control
        repeatability of operations
    """

    def __init__(self, gene_pool, seed=None):
        self._crossover = AGraphCrossover()
        self._gene_pool = gene_pool
        self._rng = np.random.default_rng(seed)

    def _select_other_parent(self, model):
        ind = self._rng.integers(0, len(self._gene_pool))
        while self._gene_pool[ind] == model:
            ind = self._rng.integers(0, len(self._gene_pool))
        return self._gene_pool[ind]

    def _do_crossover(self, model, other_parent):
        child_1, child_2 = self._crossover(model, other_parent)
        if self._rng.random() < 0.5:
            return child_1
        return child_2

    def __call__(self, model):
        """Perform crossover between the input model and a randomly selected one from the gene pool.

        This method randomly selects a parent from the gene pool, performs crossover between
        the input model and the selected parent, and returns one of the two resulting children
        with equal probability.

        Parameters
        ----------
        model : AGraph
            The model to be used as the first parent in the crossover operation

        Returns
        -------
        AGraph
            A new model resulting from crossover between the input model and a
            randomly selected model from the gene pool
        """
        other_parent = self._select_other_parent(model)
        new_model = self._do_crossover(model, other_parent)
        return new_model

    def update(self, gene_pool, *_, **__):
        """Update the gene pool used for selecting crossover partners.

        Parameters
        ----------
        gene_pool : iterable of AGraph
            The new collection of AGraph models to use as the gene pool
        *_ : tuple
            Additional positional arguments (ignored)
        **__ : dict
            Additional keyword arguments (ignored)

        Notes
        -----
        This method allows for updating the gene pool while maintaining the same
        crossover behavior. The additional parameters are included for compatibility
        with other proposal update interfaces but are not used.
        """
        self._gene_pool = list(gene_pool)
