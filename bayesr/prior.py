import warnings
import numpy as np

from smcpy import ImproperUniform

MAX_REPEATS = 100


class Prior(ImproperUniform):

    def __init__(self, generator):
        super().__init__()
        self._generator = generator

    def rvs(self, N, random_state=None):
        pool = set()
        pool_size = 0
        attempts = 0
        already_warned = False
        while len(pool) < N:
            pool.add(self._generator())

            if not already_warned:
                if len(pool) == pool_size:
                    attempts += 1
                else:
                    pool_size = len(pool)
                    attempts = 0

                if attempts >= MAX_REPEATS:
                    warnings.warn(
                        f"Generator called {MAX_REPEATS} times in a row without finding a "
                        "new unique model. This may indicate an issue with the generator "
                        "or insufficient unique models available."
                    )
                    already_warned = True

        return np.c_[list(pool)]
