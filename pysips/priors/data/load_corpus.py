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

_BENCHMARK_PATH = _DATA_DIR / "benchmark_feynman_equations.txt"

_SPECIAL_CORPORA = {"all_unique", "benchmark"}

_ALL_NAMES = set(_CORPUS_PATHS) | _SPECIAL_CORPORA


def _load_equations_from_path(path: Path) -> List[AGraphExpression]:
    equations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                expr = sympify(line)
                agraph = AGraphExpression(equation=expr)
                equations.append(agraph)
            except Exception:
                continue
    return equations


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
        - ``"all_unique"``: All corpora combined with duplicates removed.
        - ``"benchmark"``: All unique equations minus the benchmark set.
          Useful as a training corpus for benchmark evaluation.

        Pass a list to combine, e.g. ``["feynman", "bonus"]`` for all
        120 FSReD equations, or ``["wikipedia", "feynman", "bonus"]``
        for everything. The special corpora ``"all_unique"`` and
        ``"benchmark"`` cannot be combined with other names.
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

    for name in corpus_names:
        if name not in _ALL_NAMES:
            raise ValueError(
                f"Unsupported corpus name: {name!r}. "
                f"Supported: {sorted(_ALL_NAMES)}"
            )

    special = _SPECIAL_CORPORA.intersection(corpus_names)
    if special:
        if len(corpus_names) > 1:
            raise ValueError(
                f"Special corpus {special.pop()!r} cannot be combined "
                f"with other corpora."
            )
        all_equations = []
        for path in _CORPUS_PATHS.values():
            all_equations.extend(_load_equations_from_path(path))
        equations = list(set(all_equations))

        if corpus_names[0] == "benchmark":
            benchmark_eqs = set(_load_equations_from_path(_BENCHMARK_PATH))
            equations = list(set(equations) - benchmark_eqs)

        if max_samples is not None:
            equations = equations[:max_samples]
        return equations

    paths = [_CORPUS_PATHS[name] for name in corpus_names]

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
                    continue
    return equations
