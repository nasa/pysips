import numpy as np
from bingo.symbolic_regression import (
    AGraphCrossover,
)


class CrossoverProposal:
    """Proposal functor that performs bingo's Agraph crossover

    Parameters
    ----------
    gene_pool : list of AGraph
        A group of AGraphs that will be used for as parters during crossover
    seed : int, optional
        random seed used to control repeatability
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
        other_parent = self._select_other_parent(model)
        new_model = self._do_crossover(model, other_parent)
        return new_model

    def update(self, gene_pool, *args, **kwargs):
        self._gene_pool = list(gene_pool)
