from typing import List, Optional
from pathlib import Path

from bingo.expressions.agraph import AGraphExpression
from sympy import sympify

WIKIPEDIA_CORPUS_PATH = Path(__file__).parent / "wikipedia_named_equations.txt"


def load_corpus(corpus_name: str, max_samples: Optional[int] = None) -> List[AGraphExpression]:
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
    List[AGraphExpression]
        A list of AGraph equations from the specified corpus.

    Examples
    --------
    >>> equations = load_corpus("wikipedia", max_samples=100)
    >>> len(equations)
    100
    """
    if corpus_name == "wikipedia":
        equations = []
        with open(WIKIPEDIA_CORPUS_PATH, "r") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                try:
                    expr = sympify(line.strip())
                    agraph = AGraphExpression(equation=expr)
                    equations.append(agraph)
                except Exception:
                    # Skip equations that can't be converted
                    continue
        return equations
    else:
        raise ValueError(f"Unsupported corpus name: {corpus_name}")
