# PySIPS

PySIPS is a symbolic regression library centered on sampling and scoring symbolic expressions. This context captures the project-specific language around expression priors and how they are packaged for reuse.

## Language

**Prior**:
A probability model over symbolic expressions used to sample candidates and score them during regression.
_Avoid_: regularizer, heuristic

**Base Prior**:
The underlying prior used before any size calibration is applied. In this codebase, examples include the uniform, BMS, and Katz priors.
_Avoid_: raw prior, inner prior

**Size-Calibrated Prior**:
A prior formed by combining a Base Prior with a corpus-derived size distribution so expression size follows a target corpus while preserving base-prior preferences within each size.
_Avoid_: adjusted prior, normalized prior

**Prebuilt Prior Artifact**:
Versioned data shipped with the library and loaded at runtime to support prior construction, such as corpus histograms, `Z_k` tables, fitted Katz models, or fitted BMS weights.
_Avoid_: cache, blob, payload

**Prior Resolution**:
The process of turning a user-facing prior configuration into a concrete Prior by applying defaults, validating combinations, loading any needed Prebuilt Prior Artifacts, and composing the final Prior.
_Avoid_: parsing, wiring

**Corpus**:
The single named source of prior knowledge used during Prior Resolution. For a built-in prior family, the same Corpus governs all associated Prebuilt Prior Artifacts needed to construct that Prior.
_Avoid_: dataset, source corpus, calibration corpus

**Canonical Operator IDs**:
The normalized operator representation used during Prior Resolution after user-facing operator names or mixed forms have been converted into sorted bingo operator IDs. Canonical Operator IDs are the operator form used for artifact lookup and prior-resolution logging.
_Avoid_: raw operators, user operators

**Prior Spec**:
The normalized internal representation of a user-facing prior request. A Prior Spec captures prior-family semantics and validated prior parameters, but not runtime-specific resolution inputs.
_Avoid_: prior config, raw prior params

**Resolution Context**:
The runtime-specific information used to resolve a Prior Spec into a concrete Prior, including operator information, input dimensionality, and bingo construction settings.
_Avoid_: environment, config blob

**Bingo Construction Config**:
The internal, immutable set of bingo expression-generation and proposal-construction settings used during Prior Resolution and sampling setup.
_Avoid_: runtime kwargs, bingo params dict

## Example Dialogue

Dev: For this run, are we using a Prior directly or constructing one from Prebuilt Prior Artifacts?

Domain Expert: We are constructing a Size-Calibrated Prior from Prebuilt Prior Artifacts.

Dev: Which Base Prior is underneath it?

Domain Expert: Katz. The corpus size distribution comes from the benchmark corpus, but the within-size scoring still comes from the Katz Base Prior.
