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
STANDARD_CORPUS = "benchmark"


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


def _operator_filename_tag(operator_ids: List[int]) -> str:
    """Build an operator-specific filename tag."""
    return "ops" + "-".join(str(op_id) for op_id in sorted(operator_ids))


def _prebuilt_filename(
    kind: str,
    corpus: str,
    x_dim: int | None = None,
    operator_ids: List[int] | None = None,
    *,
    base_prior_key: str | None = None,
) -> str:
    """Build the expected prebuilt file name.

    Histograms describe the corpus's size distribution and are keyed by
    ``corpus`` alone. Z_k filenames depend on ``corpus``, ``x_dim`` and
    ``operator_ids``.
    """
    if kind == "histogram":
        return f"corpus_histogram_{corpus}.json"
    if kind == "z_k":
        if base_prior_key is None:
            raise ValueError("base_prior_key is required for z_k filenames")
        if x_dim is None or operator_ids is None:
            raise ValueError(
                "x_dim and operator_ids are required for z_k filenames"
            )
        operator_tag = _operator_filename_tag(operator_ids)
        return f"z_k_{base_prior_key}_{corpus}_x{x_dim}_{operator_tag}.json"
    raise ValueError(f"Unknown prebuilt kind: {kind!r}")


def _load_prebuilt_json(
    kind: str,
    corpus: str,
    x_dim: int | None = None,
    operator_ids: List[int] | None = None,
    *,
    base_prior_key: str | None = None,
) -> Dict:
    """Load the prebuilt JSON for the requested config."""
    filename = _prebuilt_filename(
        kind,
        corpus,
        x_dim,
        operator_ids,
        base_prior_key=base_prior_key,
    )
    path = _DATA_DIR / filename
    if not path.exists():
        if kind == "histogram":
            raise FileNotFoundError(
                f"No pre-built histogram for corpus={corpus!r}. "
                f"Expected: {filename}."
            )
        raise FileNotFoundError(
            f"No pre-built {kind} data for corpus={corpus!r}, "
            f"x_dim={x_dim}, operators={sorted(operator_ids or [])}. "
            f"Expected: {filename}."
        )
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_corpus_histogram(
    histogram_type: str = "empirical",
    corpus: str = STANDARD_CORPUS,
) -> Dict[int, float]:
    """Load a pre-built corpus histogram.

    The corpus size histogram is purely a property of the corpus — it
    does not depend on the user's operator set or ``x_dim``.

    Parameters
    ----------
    histogram_type : str
        ``"empirical"`` or ``"parametric"``.
    corpus : str
        Corpus name.

    Returns
    -------
    dict of {int: float}
        Log-probability per size.

    Raises
    ------
    FileNotFoundError
        If no histogram file is available for the requested corpus.
    ValueError
        If the loaded file's metadata reports a different corpus.
    """
    data = _load_prebuilt_json("histogram", corpus)

    loaded_corpus = data.get("metadata", {}).get("corpus", corpus)
    if loaded_corpus != corpus:
        raise ValueError(
            f"Corpus mismatch: pre-built histogram is for corpus="
            f"{loaded_corpus!r} but the regressor requested "
            f"corpus={corpus!r}."
        )

    if histogram_type == "parametric":
        raw = data["parametric"]["evaluated"]
    else:
        raw = data["empirical"]

    return {int(k): v for k, v in raw.items()}


def load_z_k(
    base_prior_key: str,
    user_operators: list,
    user_x_dim: int,
    corpus: str = STANDARD_CORPUS,
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
    if base_prior_key not in {"uniform", "katz"}:
        raise KeyError(
            f"No pre-built Z_k for base prior {base_prior_key!r}. "
            f"Available: {['katz', 'uniform']}."
        )

    user_ids = _resolve_operator_ids(user_operators)
    data = _load_prebuilt_json(
        "z_k",
        corpus,
        user_x_dim,
        user_ids,
        base_prior_key=base_prior_key,
    )

    meta = data["metadata"]
    loaded_operators = sorted(meta["operators"])
    loaded_x_dim = int(meta["x_dim"])
    loaded_corpus = meta["corpus"]

    if loaded_corpus != corpus:
        raise ValueError(
            f"Corpus mismatch: pre-built data uses corpus={loaded_corpus!r} "
            f"but the regressor requested corpus={corpus!r}."
        )
    if user_ids != loaded_operators:
        raise ValueError(
            f"Operator mismatch: pre-built data uses operators "
            f"{loaded_operators} but the regressor has "
            f"{user_ids}. Use "
            f"fit_size_calibrated_prior() for custom operator sets."
        )
    if user_x_dim != loaded_x_dim:
        raise ValueError(
            f"x_dim mismatch: pre-built data uses x_dim="
            f"{loaded_x_dim} but the regressor has x_dim="
            f"{user_x_dim}. Use fit_size_calibrated_prior() "
            f"for custom x_dim values."
        )

    return {int(k): v for k, v in data["log_z_k"].items()}


def load_prebuilt_size_calibrated(
    base_prior_key: str,
    user_operators: list,
    user_x_dim: int,
    histogram_type: str = "empirical",
    corpus: str = STANDARD_CORPUS,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Load both corpus histogram and Z_k for a size-calibrated prior.

    Parameters
    ----------
    base_prior_key : str
        ``"uniform"`` or ``"katz"``.
    user_operators : list
        Operator names or IDs from the regressor (used for Z_k).
    user_x_dim : int
        Number of input features (used for Z_k).
    histogram_type : str
        ``"empirical"`` or ``"parametric"``.

    Returns
    -------
    corpus_log_hist : dict of {int: float}
    log_z_k : dict of {int: float}
    """
    corpus_log_hist = load_corpus_histogram(histogram_type, corpus)
    log_z_k = load_z_k(base_prior_key, user_operators, user_x_dim, corpus)
    return corpus_log_hist, log_z_k
