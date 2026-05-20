"""Loader for pre-built size-calibrated prior data files.

Loads and validates corpus histogram and Z_k JSON files shipped in
``pysips/priors/data/``.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from bingo.expressions.agraph.component_generator import ComponentGenerator

_DATA_DIR = Path(__file__).parent / "data"

# Standard pre-built config
STANDARD_OPERATORS = [3, 4, 5, 6, 15, 16, 13, 14]
STANDARD_X_DIM = 1

# File names for the standard config
_HISTOGRAM_FILE = "corpus_histogram_benchmark_x1_ops8.json"
_Z_K_FILES = {
    "uniform": "z_k_uniform_benchmark_x1_ops8.json",
    "katz": "z_k_katz_benchmark_x1_ops8.json",
}


def _resolve_operator_ids(operators: list) -> List[int]:
    """Convert a list of operator strings/ints to integer IDs.

    Parameters
    ----------
    operators : list
        Operator names (``"+"``) or integer IDs.

    Returns
    -------
    list of int
        Sorted operator IDs.
    """
    ids = []
    for op in operators:
        if isinstance(op, int):
            ids.append(op)
        else:
            ids.append(ComponentGenerator._operator_from_string(op))
    return sorted(ids)


def _validate_config(
    loaded_operators: List[int],
    loaded_x_dim: int,
    user_operators: List[int],
    user_x_dim: int,
) -> None:
    """Raise ``ValueError`` if user config doesn't match pre-built data."""
    if sorted(user_operators) != sorted(loaded_operators):
        raise ValueError(
            f"Operator mismatch: pre-built data uses operators "
            f"{sorted(loaded_operators)} but the regressor has "
            f"{sorted(user_operators)}. Use "
            f"fit_size_calibrated_prior() for custom operator sets."
        )
    if user_x_dim != loaded_x_dim:
        raise ValueError(
            f"x_dim mismatch: pre-built data uses x_dim="
            f"{loaded_x_dim} but the regressor has x_dim="
            f"{user_x_dim}. Use fit_size_calibrated_prior() "
            f"for custom x_dim values."
        )


def load_corpus_histogram(
    user_operators: list,
    user_x_dim: int,
    histogram_type: str = "empirical",
) -> Dict[int, float]:
    """Load a pre-built corpus histogram.

    Parameters
    ----------
    user_operators : list
        Operator names or IDs from the regressor.
    user_x_dim : int
        Number of input features.
    histogram_type : str
        ``"empirical"`` or ``"parametric"``.

    Returns
    -------
    dict of {int: float}
        Log-probability per size.

    Raises
    ------
    ValueError
        If the user's config does not match the pre-built data.
    """
    path = _DATA_DIR / _HISTOGRAM_FILE
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    user_ids = _resolve_operator_ids(user_operators)
    _validate_config(meta["operators"], meta["x_dim"], user_ids, user_x_dim)

    if histogram_type == "parametric":
        raw = data["parametric"]["evaluated"]
    else:
        raw = data["empirical"]

    return {int(k): v for k, v in raw.items()}


def load_z_k(
    base_prior_key: str,
    user_operators: list,
    user_x_dim: int,
) -> Dict[int, float]:
    """Load a pre-built Z_k table.

    Parameters
    ----------
    base_prior_key : str
        ``"uniform"`` or ``"katz"``.
    user_operators : list
        Operator names or IDs from the regressor.
    user_x_dim : int
        Number of input features.

    Returns
    -------
    dict of {int: float}
        Log Z_k per size.

    Raises
    ------
    ValueError
        If the user's config does not match the pre-built data.
    KeyError
        If *base_prior_key* is not recognised.
    """
    if base_prior_key not in _Z_K_FILES:
        raise KeyError(
            f"No pre-built Z_k for base prior {base_prior_key!r}. "
            f"Available: {sorted(_Z_K_FILES)}."
        )

    path = _DATA_DIR / _Z_K_FILES[base_prior_key]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    user_ids = _resolve_operator_ids(user_operators)
    _validate_config(meta["operators"], meta["x_dim"], user_ids, user_x_dim)

    return {int(k): v for k, v in data["log_z_k"].items()}


def load_prebuilt_size_calibrated(
    base_prior_key: str,
    user_operators: list,
    user_x_dim: int,
    histogram_type: str = "empirical",
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Load both corpus histogram and Z_k for a size-calibrated prior.

    Parameters
    ----------
    base_prior_key : str
        ``"uniform"`` or ``"katz"``.
    user_operators : list
        Operator names or IDs from the regressor.
    user_x_dim : int
        Number of input features.
    histogram_type : str
        ``"empirical"`` or ``"parametric"``.

    Returns
    -------
    corpus_log_hist : dict of {int: float}
    log_z_k : dict of {int: float}
    """
    corpus_log_hist = load_corpus_histogram(user_operators, user_x_dim, histogram_type)
    log_z_k = load_z_k(base_prior_key, user_operators, user_x_dim)
    return corpus_log_hist, log_z_k
