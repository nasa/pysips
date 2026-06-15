"""Internal Prior Resolution for built-in and custom priors."""

from dataclasses import dataclass
import logging
from math import inf, isfinite, isinf
from numbers import Real
from typing import Any, Mapping, Optional, Sequence

from bingo.expressions.agraph.component_generator import ComponentGenerator

from ..bingo_construction import BingoConstructionConfig, build_agraph_generator
from .bms_prior import BMSPrior, load_bms_weights
from .improper_uniform_prior import ImproperUniformPrior
from .katz_prior import KatzPrior, load_katz_model
from .prebuilt_loader import load_corpus_histogram, load_z_k
from .size_calibrated_prior import SizeCalibratedPrior

_LOGGER = logging.getLogger("pysips.priors.resolution")
_KNOWN_PRIOR_STRINGS = {
    "uniform",
    "bms",
    "katz",
    "size_calibrated_uniform",
    "size_calibrated_katz",
    "size_calibrated_bms",
}


@dataclass(frozen=True)
class PriorSpec:
    """Normalized internal representation of a built-in prior request."""

    base_prior_key: str
    artifact_family: str
    size_calibrated: bool
    corpus: str = "benchmark"
    n: Optional[int] = None
    normalize: Optional[bool] = None
    fit_if_missing: Optional[bool] = None
    floor_log_prob: Optional[float] = None


@dataclass(frozen=True)
class ResolutionContext:
    """Runtime context for resolving a concrete Prior."""

    original_operators: tuple[Any, ...]
    canonical_operator_ids: tuple[int, ...]
    x_dim: int
    bingo_config: BingoConstructionConfig
    num_mcmc_samples: int
    target_ess: float
    max_time: Optional[float]
    max_equation_evals: Optional[int]
    random_state: Optional[int]


def build_prior(
    prior,
    prior_params,
    *,
    operators: Sequence[Any],
    x_dim: int,
    bingo_config: BingoConstructionConfig,
    num_mcmc_samples: int = 5,
    target_ess: float = 0.8,
    max_time: Optional[float] = None,
    max_equation_evals: Optional[int] = None,
    random_state: Optional[int] = None,
):
    """Resolve a user-facing prior configuration into a concrete Prior."""
    params = _normalize_prior_params(prior_params)
    context = ResolutionContext(
        original_operators=tuple(operators),
        canonical_operator_ids=tuple(_canonicalize_operator_ids(operators)),
        x_dim=x_dim,
        bingo_config=bingo_config,
        num_mcmc_samples=num_mcmc_samples,
        target_ess=target_ess,
        max_time=max_time,
        max_equation_evals=max_equation_evals,
        random_state=random_state,
    )
    try:
        if isinstance(prior, str):
            spec = _normalize_builtin_prior(prior, params)
            resolved_prior = _build_builtin_prior(spec, context)
            _LOGGER.info("resolved_prior", extra=_log_fields(prior, params, context, spec))
            return resolved_prior

        _validate_custom_prior(prior)
        _LOGGER.info("resolved_prior", extra=_log_fields(prior, params, context, None))
        return prior
    except (TypeError, ValueError) as exc:
        _LOGGER.warning(
            "prior_resolution_failed",
            extra=_log_fields(prior, params, context, None, exc),
        )
        raise
    except Exception as exc:
        _LOGGER.error(
            "prior_resolution_failed",
            extra=_log_fields(prior, params, context, None, exc),
            exc_info=True,
        )
        raise


def _normalize_prior_params(prior_params) -> dict[str, Any]:
    if prior_params is None:
        return {}
    if not isinstance(prior_params, Mapping):
        raise TypeError("prior_params must be a mapping or None.")
    return dict(prior_params)


def _validate_custom_prior(prior) -> None:
    if not hasattr(prior, "rvs") or not hasattr(prior, "logpdf"):
        raise TypeError("Custom prior must have 'rvs' and 'logpdf' methods.")


def _normalize_builtin_prior(prior: str, params: Mapping[str, Any]) -> PriorSpec:
    if prior not in _KNOWN_PRIOR_STRINGS:
        raise ValueError(
            f"Unknown prior '{prior}'. Expected one of "
            f"{sorted(_KNOWN_PRIOR_STRINGS)} or a prior object with "
            f"'rvs' and 'logpdf' methods."
        )

    size_calibrated = prior.startswith("size_calibrated_")
    base_prior_key = prior.removeprefix("size_calibrated_")
    allowed_keys = {"corpus"}
    if base_prior_key == "katz":
        allowed_keys |= {"n", "normalize"}
        if not size_calibrated:
            allowed_keys.add("fit_if_missing")
    if size_calibrated:
        allowed_keys.add("floor_log_prob")

    unknown_keys = sorted(set(params) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unsupported prior_params for {prior!r}: {unknown_keys}")

    corpus = _validate_corpus(params.get("corpus", "benchmark"))
    n = None
    normalize = None
    fit_if_missing = None
    floor_log_prob = None

    if base_prior_key == "katz":
        n = _validate_positive_int(params.get("n", 2), "n")
        normalize = _validate_bool(params.get("normalize", False), "normalize")
        if size_calibrated:
            if "fit_if_missing" in params:
                raise ValueError(
                    "fit_if_missing is only supported for the plain 'katz' prior."
                )
        else:
            fit_if_missing = _validate_bool(
                params.get("fit_if_missing", False), "fit_if_missing"
            )
    elif "fit_if_missing" in params:
        raise ValueError("fit_if_missing is only supported for the plain 'katz' prior.")

    if size_calibrated:
        floor_log_prob = _validate_floor_log_prob(params.get("floor_log_prob", -inf))

    return PriorSpec(
        base_prior_key=base_prior_key,
        artifact_family=base_prior_key,
        size_calibrated=size_calibrated,
        corpus=corpus,
        n=n,
        normalize=normalize,
        fit_if_missing=fit_if_missing,
        floor_log_prob=floor_log_prob,
    )


def _validate_corpus(corpus: Any) -> str:
    if not isinstance(corpus, str) or not corpus.strip():
        raise ValueError("corpus must be a non-empty string.")
    return corpus


def _validate_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _validate_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool.")
    return value


def _validate_floor_log_prob(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("floor_log_prob must be a finite number or -inf.")
    value = float(value)
    if value == -inf:
        return value
    if not isfinite(value) or isinf(value):
        raise ValueError("floor_log_prob must be a finite number or -inf.")
    return value


def _build_builtin_prior(spec: PriorSpec, context: ResolutionContext):
    if spec.size_calibrated:
        return _build_size_calibrated_prior(spec, context)
    if spec.base_prior_key == "uniform":
        return _build_uniform_prior(context)
    if spec.base_prior_key == "bms":
        return _build_bms_prior(spec, context)
    if spec.base_prior_key == "katz":
        return _build_katz_prior(spec, context)
    raise ValueError(f"Unsupported base prior: {spec.base_prior_key!r}")


def _build_uniform_prior(context: ResolutionContext):
    generator = build_agraph_generator(
        context.x_dim,
        context.original_operators,
        context.bingo_config,
    )
    return ImproperUniformPrior(generator)


def _build_bms_prior(spec: PriorSpec, context: ResolutionContext):
    weights, squared_weights = load_bms_weights(spec.corpus)
    return BMSPrior(
        weights,
        squared_weights,
        operators=list(context.original_operators),
        x_dim=context.x_dim,
        **_samplable_prior_kwargs(context),
        **_bingo_prior_kwargs(context),
    )


def _build_katz_prior(spec: PriorSpec, context: ResolutionContext, *, fit_if_missing=None):
    model = load_katz_model(
        n=spec.n or 2,
        corpus=spec.corpus,
        fit_if_missing=(
            spec.fit_if_missing if fit_if_missing is None else fit_if_missing
        ),
    )
    return KatzPrior(
        model,
        normalize=bool(spec.normalize),
        operators=list(context.original_operators),
        x_dim=context.x_dim,
        **_samplable_prior_kwargs(context),
        **_bingo_prior_kwargs(context),
    )


def _build_size_calibrated_prior(spec: PriorSpec, context: ResolutionContext):
    corpus_log_hist = load_corpus_histogram(corpus=spec.corpus)
    log_z_k = load_z_k(
        spec.artifact_family,
        list(context.canonical_operator_ids),
        context.x_dim,
        corpus=spec.corpus,
    )
    if spec.base_prior_key == "uniform":
        base_prior = None
    elif spec.base_prior_key == "katz":
        base_prior = _build_katz_prior(spec, context, fit_if_missing=False)
    elif spec.base_prior_key == "bms":
        base_prior = _build_bms_prior(spec, context)
    else:
        raise ValueError(f"Unsupported base prior: {spec.base_prior_key!r}")

    return SizeCalibratedPrior(
        base_prior=base_prior,
        log_z_k=log_z_k,
        corpus_log_hist=corpus_log_hist,
        floor_log_prob=spec.floor_log_prob if spec.floor_log_prob is not None else -inf,
        operators=list(context.original_operators),
        x_dim=context.x_dim,
        **_samplable_prior_kwargs(context),
        **_bingo_prior_kwargs(context),
    )


def _samplable_prior_kwargs(context: ResolutionContext) -> dict[str, Any]:
    kwargs = {
        "num_mcmc_samples": context.num_mcmc_samples,
        "target_ess": context.target_ess,
    }
    if context.max_time is not None:
        kwargs["max_time"] = context.max_time
    if context.max_equation_evals is not None:
        kwargs["max_equation_evals"] = context.max_equation_evals
    if context.random_state is not None:
        kwargs["random_state"] = context.random_state
    return kwargs


def _bingo_prior_kwargs(context: ResolutionContext) -> dict[str, Any]:
    config = context.bingo_config
    return {
        "max_complexity": config.max_complexity,
        "terminal_probability": config.terminal_probability,
        "constant_probability": config.constant_probability,
        "command_probability": config.command_probability,
        "node_probability": config.node_probability,
        "parameter_probability": config.parameter_probability,
        "prune_probability": config.prune_probability,
        "fork_probability": config.fork_probability,
        "repeat_mutation_probability": config.repeat_mutation_probability,
        "crossover_pool_size": config.crossover_pool_size,
        "mutation_prob": config.mutation_prob,
        "crossover_prob": config.crossover_prob,
        "exclusive": config.exclusive,
    }


def _canonicalize_operator_ids(operators: Sequence[Any]) -> list[int]:
    canonical_ids = []
    for operator in operators:
        if isinstance(operator, int):
            canonical_ids.append(operator)
        else:
            canonical_ids.append(ComponentGenerator._operator_from_string(operator))
    return sorted(canonical_ids)


def _log_fields(
    prior,
    params: Mapping[str, Any],
    context: ResolutionContext,
    spec: Optional[PriorSpec],
    exc: Optional[Exception] = None,
) -> dict[str, Any]:
    fields = {
        "requested_prior": prior if isinstance(prior, str) else None,
        "base_prior": spec.base_prior_key if spec is not None else None,
        "size_calibrated": spec.size_calibrated if spec is not None else None,
        "corpus": spec.corpus if spec is not None else None,
        "raw_prior_param_keys": sorted(params),
        "original_operators": list(context.original_operators),
        "canonical_operator_ids": list(context.canonical_operator_ids),
        "x_dim": context.x_dim,
        "katz_n": spec.n if spec is not None else None,
        "katz_normalize": spec.normalize if spec is not None else None,
        "katz_fit_if_missing": spec.fit_if_missing if spec is not None else None,
        "floor_log_prob": spec.floor_log_prob if spec is not None else None,
        "custom_prior_class": None,
        "custom_prior_module": None,
        "failure_type": type(exc).__name__ if exc is not None else None,
    }
    if not isinstance(prior, str):
        fields["custom_prior_class"] = type(prior).__name__
        fields["custom_prior_module"] = type(prior).__module__
    return fields
