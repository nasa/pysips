from typing import List, Optional, Union
from pathlib import Path

from bingo.expressions.agraph import AGraphExpression
from sympy import sympify

_DATA_DIR = Path(__file__).parent

_CORPUS_PATHS = {
    "wikipedia": _DATA_DIR / "wikipedia_named_equations.txt",
    "feynman": _DATA_DIR / "feynman_equations.txt",
    "bonus": _DATA_DIR / "bonus_equations.txt",
}


def load_corpus(
    corpus_name: Union[str, List[str]],
    max_samples: Optional[int] = None,
) -> List[AGraphExpression]:
    """
    Load a corpus of equations from one or more datasets.

    Parameters
    ----------
    corpus_name : str or List[str]
        The name(s) of the corpus to load. May be a single string or a
        list of strings to combine multiple corpora. Supported values:

        - ``"wikipedia"``: Wikipedia named-equations corpus (~4000 eqs).
        - ``"feynman"``: 100 equations from the Feynman Lectures on Physics.
        - ``"bonus"``: 20 "bonus" problems from Goldstein, Jackson,
          Weinberg, and Schwartz.

        Pass a list to combine, e.g. ``["feynman", "bonus"]`` for all
        120 FSReD equations, or ``["wikipedia", "feynman", "bonus"]``
        for everything.
    max_samples : Optional[int], default=None
        If specified, limits the total number of equations loaded.

    Returns
    -------
    List[AGraphExpression]
        A list of AGraph equations from the specified corpus.

    Examples
    --------
    >>> equations = load_corpus("wikipedia", max_samples=100)
    >>> len(equations)
    100
    >>> feynman_all = load_corpus(["feynman", "bonus"])
    >>> len(feynman_all)
    120
    """
    if isinstance(corpus_name, str):
        corpus_names = [corpus_name]
    else:
        corpus_names = list(corpus_name)

    paths = []
    for name in corpus_names:
        if name not in _CORPUS_PATHS:
            raise ValueError(
                f"Unsupported corpus name: {name!r}. "
                f"Supported: {sorted(_CORPUS_PATHS)}"
            )
        paths.append(_CORPUS_PATHS[name])

    equations: List[AGraphExpression] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples is not None and len(equations) >= max_samples:
                    return equations
                line = line.strip()
                if not line:
                    continue
                try:
                    expr = sympify(line)
                    agraph = AGraphExpression(equation=expr)
                    equations.append(agraph)
                except Exception:
                    # Skip equations that can't be converted
                    continue
    return equations
