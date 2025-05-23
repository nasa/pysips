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
        active_proposals = self._select_proposals()
        new_model = active_proposals[0](model)
        for prop in active_proposals[1:]:
            new_model = prop(new_model)
        return new_model

    def update(self, *args, **kwargs):
        for p in self._proposals:
            p.update(*args, **kwargs)
