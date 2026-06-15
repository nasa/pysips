"""Loader for pre-built size-calibrated prior data files.

Loads and validates corpus histogram and Z_k JSON files shipped in
``pysips/priors/data/``.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

_DATA_DIR = Path(__file__).parent / "data"

# Default corpus for shipped prebuilt artifacts.
STANDARD_CORPUS = "benchmark"
_KATZ_MODEL_DIR = _DATA_DIR


def katz_model_path(n: int, corpus: str = STANDARD_CORPUS) -> Path:
    """Return the expected on-disk path for a prebuilt Katz model."""
    return _KATZ_MODEL_DIR / f"default_katz_n{n}_{corpus}.json"


def _validate_canonical_operator_ids(canonical_operator_ids: list) -> List[int]:
    """Validate canonical operator IDs passed in by Prior Resolution."""
    if not all(isinstance(operator_id, int) for operator_id in canonical_operator_ids):
        raise TypeError("canonical_operator_ids must be a list of bingo operator IDs.")
    return sorted(canonical_operator_ids)


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
    artifact_family: str,
    canonical_operator_ids: list,
    resolved_x_dim: int,
    corpus: str = STANDARD_CORPUS,
) -> Dict[int, float]:
    """Load a pre-built Z_k table.

    Parameters
    ----------
    artifact_family : str
        ``"uniform"``, ``"katz"``, or ``"bms"``.
    canonical_operator_ids : list of int
        Sorted bingo operator IDs supplied by Prior Resolution.
    resolved_x_dim : int
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
    if artifact_family not in {"uniform", "katz", "bms"}:
        raise KeyError(
            f"No pre-built Z_k for artifact family {artifact_family!r}. "
            f"Available: {['bms', 'katz', 'uniform']}."
        )

    operator_ids = _validate_canonical_operator_ids(canonical_operator_ids)
    data = _load_prebuilt_json(
        "z_k",
        corpus,
        resolved_x_dim,
        operator_ids,
        base_prior_key=artifact_family,
    )

    meta = data["metadata"]
    loaded_operators = sorted(meta["operators"])
    loaded_x_dim = int(meta["x_dim"])
    loaded_corpus = meta["corpus"]
    loaded_family = meta.get("base_prior", artifact_family)

    if loaded_corpus != corpus:
        raise ValueError(
            f"Corpus mismatch: pre-built data uses corpus={loaded_corpus!r} "
            f"but Prior Resolution requested corpus={corpus!r}."
        )
    if loaded_family != artifact_family:
        raise ValueError(
            f"Artifact-family mismatch: pre-built data uses {loaded_family!r} "
            f"but Prior Resolution requested {artifact_family!r}."
        )
    if operator_ids != loaded_operators:
        raise ValueError(
            f"Operator mismatch: pre-built data uses operators "
            f"{loaded_operators} but the regressor has "
            f"{operator_ids}. Use "
            f"fit_size_calibrated_prior() for custom operator sets."
        )
    if resolved_x_dim != loaded_x_dim:
        raise ValueError(
            f"x_dim mismatch: pre-built data uses x_dim="
            f"{loaded_x_dim} but Prior Resolution has x_dim="
            f"{resolved_x_dim}. Use fit_size_calibrated_prior() "
            f"for custom x_dim values."
        )

    return {int(k): v for k, v in data["log_z_k"].items()}


def load_prebuilt_size_calibrated(
    artifact_family: str,
    canonical_operator_ids: list,
    resolved_x_dim: int,
    histogram_type: str = "empirical",
    corpus: str = STANDARD_CORPUS,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Load both corpus histogram and Z_k for a size-calibrated prior.

    Parameters
    ----------
    artifact_family : str
        ``"uniform"``, ``"katz"``, or ``"bms"``.
    canonical_operator_ids : list of int
        Canonical Operator IDs (used for Z_k).
    resolved_x_dim : int
        Number of input features (used for Z_k).
    histogram_type : str
        ``"empirical"`` or ``"parametric"``.

    Returns
    -------
    corpus_log_hist : dict of {int: float}
    log_z_k : dict of {int: float}
    """
    corpus_log_hist = load_corpus_histogram(histogram_type, corpus)
    log_z_k = load_z_k(artifact_family, canonical_operator_ids, resolved_x_dim, corpus)
    return corpus_log_hist, log_z_k
