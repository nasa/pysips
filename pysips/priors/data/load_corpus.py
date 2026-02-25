from typing import List, Optional
from pathlib import Path
from bingo.symbolic_regression import AGraph
from sympy import sympify

WIKIPEDIA_CORPUS_PATH = Path(__file__).parent / "wikipedia_named_equations.txt"


def load_corpus(corpus_name: str, max_samples: Optional[int] = None) -> List[AGraph]:
    """
    Load a corpus of equations from a specified dataset.

    Parameters
    ----------
    corpus_name : str
        The name of the corpus to load. Supported values include:
        - "wikipedia": Load equations from the Wikipedia named equations dataset.
    max_samples : Optional[int], default=None
        If specified, limits the number of equations loaded from the corpus.

    Returns
    -------
    List[AGaph]
        A list of equations from the specified corpus.
    """
    if corpus_name == "wikipedia":
        equations = []
        with open(WIKIPEDIA_CORPUS_PATH, "r") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                equations.append(sympify(line.strip()))
        return equations
    else:
        raise ValueError(f"Unsupported corpus name: {corpus_name}")
