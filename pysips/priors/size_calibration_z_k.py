"""Z_k estimation for size-calibrated priors.

Provides functions to compute or estimate the per-size partition
function Z_k under different base priors:

- **Analytical** (uniform base): exact tree-count recurrence.
- **Hybrid** (Katz base): shape MC + vectorized tree DP.
- **MC fallback** (arbitrary base): pure Monte Carlo.

Also includes tree-shape utilities for counting, sampling, and
enumerating unlabeled tree shapes.
"""

import warnings
from math import log
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

from bingo.expressions.agraph import AGraphExpression
from bingo.expressions.agraph.pyagraph import (
    CONSTANT,
    IS_ARITY_2_ARRAY,
    IS_TERMINAL_ARRAY,
    VARIABLE,
)


# ------------------------------------------------------------------ #
# Internal operator classification                                    #
# ------------------------------------------------------------------ #


def _classify_operators(operators, x_dim):
    """Classify operators by arity (internal helper)."""
    unary_ids = [
        op for op in operators
        if not IS_ARITY_2_ARRAY[op] and not IS_TERMINAL_ARRAY[op]
    ]
    binary_ids = [op for op in operators if IS_ARITY_2_ARRAY[op]]
    n_t = x_dim + 1
    return {
        "n_t": n_t,
        "n_u": len(unary_ids),
        "n_b": len(binary_ids),
        "unary_ids": unary_ids,
        "binary_ids": binary_ids,
    }


# ------------------------------------------------------------------ #
# Analytical Z_k for uniform base                                    #
# ------------------------------------------------------------------ #


def compute_labeled_tree_counts(
    n_t: int, n_u: int, n_b: int, max_size: int
) -> Dict[int, float]:
    """Exact count of labeled expression trees per tree-size.

    Uses the generating-function recurrence for

        T(x) = n_t x + n_u x T(x) + n_b x T(x)^2

    where *n_t*, *n_u*, *n_b* are the numbers of terminal labels,
    unary operators, and binary operators respectively.

    Parameters
    ----------
    n_t, n_u, n_b : int
        Terminal label count, unary operator count, binary operator
        count.
    max_size : int
        Maximum tree size to compute (inclusive).

    Returns
    -------
    dict of {int: float}
        Mapping from tree size *k* to ``log(a_k)`` where *a_k* is
        the number of distinct labeled trees of size *k*.  Sizes with
        zero count are omitted.
    """
    if max_size < 1:
        return {}

    a = [0.0] * (max_size + 1)
    a[1] = float(n_t)

    for k in range(2, max_size + 1):
        val = n_u * a[k - 1]
        for i in range(1, k - 1):
            val += n_b * a[i] * a[k - 1 - i]
        a[k] = val

    return {k: log(a[k]) for k in range(1, max_size + 1) if a[k] > 0}


# ------------------------------------------------------------------ #
# Shape tree data structure                                           #
# ------------------------------------------------------------------ #


class ShapeNode:
    """Lightweight tree-shape node.

    Attributes
    ----------
    kind : str
        One of ``"leaf"``, ``"unary"``, ``"binary"``.
    children : tuple of ShapeNode
        Child nodes (empty for leaf, 1 for unary, 2 for binary).
    """

    __slots__ = ("kind", "children", "_hash")

    def __init__(self, kind: str, children: tuple = ()):
        self.kind = kind
        self.children = children
        self._hash = hash((kind, tuple(hash(c) for c in children)))

    def __eq__(self, other):
        if not isinstance(other, ShapeNode):
            return NotImplemented
        return self.kind == other.kind and self.children == other.children

    def __hash__(self):
        return self._hash


# ------------------------------------------------------------------ #
# Shape counting                                                      #
# ------------------------------------------------------------------ #


def compute_shape_counts(
    has_unary: bool, has_binary: bool, max_size: int
) -> Dict[int, int]:
    """Count unlabeled tree shapes per size.

    Uses the recurrence for
        S(x) = x + I_u x S(x) + I_b x S(x)^2

    Parameters
    ----------
    has_unary, has_binary : bool
        Whether the operator set contains any unary / binary operators.
    max_size : int
        Maximum tree size (inclusive).

    Returns
    -------
    dict of {int: int}
        Mapping from size *k* to the number of distinct tree shapes.
    """
    if max_size < 1:
        return {}

    I_u = int(has_unary)
    I_b = int(has_binary)

    s = [0] * (max_size + 1)
    s[1] = 1

    for k in range(2, max_size + 1):
        val = I_u * s[k - 1]
        for i in range(1, k - 1):
            val += I_b * s[i] * s[k - 1 - i]
        s[k] = val

    return {k: s[k] for k in range(1, max_size + 1) if s[k] > 0}


# ------------------------------------------------------------------ #
# Shape sampling and enumeration                                      #
# ------------------------------------------------------------------ #


def sample_shape(
    k: int,
    shape_counts: Dict[int, int],
    has_unary: bool,
    has_binary: bool,
    rng: np.random.Generator,
) -> ShapeNode:
    """Sample a tree shape of size *k* uniformly at random.

    Uses recursive decomposition weighted by shape counts.

    Parameters
    ----------
    k : int
        Target tree size (number of nodes).
    shape_counts : dict of {int: int}
        Pre-computed shape counts (from :func:`compute_shape_counts`).
    has_unary, has_binary : bool
        Whether unary / binary operators exist.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ShapeNode
        Root of the sampled tree shape.
    """
    if k == 1:
        return ShapeNode("leaf")

    weights = []
    choices = []

    if has_unary:
        w_unary = shape_counts.get(k - 1, 0)
        if w_unary > 0:
            weights.append(w_unary)
            choices.append("unary")

    if has_binary:
        for i in range(1, k - 1):
            s_i = shape_counts.get(i, 0)
            s_rest = shape_counts.get(k - 1 - i, 0)
            w = s_i * s_rest
            if w > 0:
                weights.append(w)
                choices.append(("binary", i, k - 1 - i))

    total = sum(weights)
    probs = [w / total for w in weights]
    idx = rng.choice(len(choices), p=probs)
    choice = choices[idx]

    if choice == "unary":
        child = sample_shape(k - 1, shape_counts, has_unary, has_binary, rng)
        return ShapeNode("unary", (child,))

    _, left_size, right_size = choice
    left = sample_shape(left_size, shape_counts, has_unary, has_binary, rng)
    right = sample_shape(right_size, shape_counts, has_unary, has_binary, rng)
    return ShapeNode("binary", (left, right))


def enumerate_shapes(
    k: int,
    shape_counts: Dict[int, int],
    has_unary: bool,
    has_binary: bool,
) -> List[ShapeNode]:
    """Enumerate all distinct tree shapes of size *k*.

    Parameters
    ----------
    k : int
        Target tree size (number of nodes).
    shape_counts : dict of {int: int}
        Pre-computed shape counts (from :func:`compute_shape_counts`).
    has_unary, has_binary : bool
        Whether unary / binary operators exist.

    Returns
    -------
    list of ShapeNode
        All distinct tree shapes of the given size.
    """
    if k == 1:
        return [ShapeNode("leaf")]

    result: List[ShapeNode] = []
    if has_unary and shape_counts.get(k - 1, 0) > 0:
        for child in enumerate_shapes(k - 1, shape_counts, has_unary,
                                      has_binary):
            result.append(ShapeNode("unary", (child,)))
    if has_binary:
        for i in range(1, k - 1):
            if (shape_counts.get(i, 0) > 0
                    and shape_counts.get(k - 1 - i, 0) > 0):
                for left in enumerate_shapes(i, shape_counts, has_unary,
                                             has_binary):
                    for right in enumerate_shapes(k - 1 - i, shape_counts,
                                                  has_unary, has_binary):
                        result.append(ShapeNode("binary", (left, right)))
    return result


# ------------------------------------------------------------------ #
# Vectorized Katz DP (precomputed bigram matrices)                    #
# ------------------------------------------------------------------ #


class _KatzDPContext:
    """Precomputed bigram log-probability matrices for vectorized DP.

    Instead of calling ``katz_model.log_prob`` per operator pair inside
    the tree DP (millions of Python-level calls), we precompute all
    pairwise log-probabilities once into numpy arrays.  The DP then
    uses array slicing + ``logsumexp`` instead of Python loops.

    The operator vocabulary is:

        all_ops = [VARIABLE, CONSTANT] + unary_ops + binary_ops

    Every DP result is a vector of length ``n_ops`` (with ``-inf``
    for operator kinds that cannot appear at a given node).
    """

    __slots__ = (
        "all_ops", "n_ops", "op_to_idx",
        "leaf_idx", "unary_idx", "binary_idx",
        "leaf_log_m", "log_left", "log_right", "log_root",
    )

    def __init__(self, katz_model, x_dim, unary_ops, binary_ops):
        all_ops = [VARIABLE, CONSTANT] + list(unary_ops) + list(binary_ops)
        self.all_ops = all_ops
        self.n_ops = len(all_ops)
        self.op_to_idx = {op: i for i, op in enumerate(all_ops)}

        self.leaf_idx = np.array(
            [self.op_to_idx[VARIABLE], self.op_to_idx[CONSTANT]]
        )
        self.unary_idx = np.array(
            [self.op_to_idx[op] for op in unary_ops]
        )
        self.binary_idx = np.array(
            [self.op_to_idx[op] for op in binary_ops]
        )

        self.leaf_log_m = np.full(self.n_ops, float("-inf"))
        self.leaf_log_m[self.op_to_idx[VARIABLE]] = (
            log(x_dim) if x_dim > 0 else float("-inf")
        )
        self.leaf_log_m[self.op_to_idx[CONSTANT]] = 0.0

        left_model = katz_model.left_model
        right_model = katz_model.right_model
        n = self.n_ops

        self.log_left = np.full((n, n), float("-inf"))
        self.log_right = np.full((n, n), float("-inf"))
        self.log_root = np.full(n, float("-inf"))

        for i, op_i in enumerate(all_ops):
            self.log_root[i] = left_model.log_prob((op_i,))
            for j, op_j in enumerate(all_ops):
                self.log_left[i, j] = left_model.log_prob((op_i, op_j))
                self.log_right[i, j] = right_model.log_prob((op_i, op_j))


def _compute_marginal_vec(
    shape: ShapeNode,
    ctx: _KatzDPContext,
    cache: Optional[Dict] = None,
) -> np.ndarray:
    """Vectorized DP: log-marginal vector of shape ``(n_ops,)``.

    Entry *i* is the log of the total Katz probability mass from this
    subtree when the current node is labeled ``ctx.all_ops[i]``.
    Entries for operator kinds that cannot appear at this node are
    ``-inf``.

    Parameters
    ----------
    shape : ShapeNode
    ctx : _KatzDPContext
        Precomputed matrices.
    cache : dict, optional
        Shared memoisation dict (keyed on ShapeNode).

    Returns
    -------
    np.ndarray of shape ``(n_ops,)``
    """
    if cache is not None and shape in cache:
        return cache[shape]

    n = ctx.n_ops
    result = np.full(n, float("-inf"))

    if shape.kind == "leaf":
        result[:] = ctx.leaf_log_m

    elif shape.kind == "unary":
        child_m = _compute_marginal_vec(shape.children[0], ctx, cache)
        if len(ctx.unary_idx) > 0:
            result[ctx.unary_idx] = logsumexp(
                ctx.log_left[ctx.unary_idx] + child_m, axis=1
            )

    else:  # binary
        left_m = _compute_marginal_vec(shape.children[0], ctx, cache)
        right_m = _compute_marginal_vec(shape.children[1], ctx, cache)
        if len(ctx.binary_idx) > 0:
            g = logsumexp(ctx.log_right + right_m, axis=1)
            result[ctx.binary_idx] = logsumexp(
                ctx.log_left[ctx.binary_idx] + left_m + g, axis=1
            )

    if cache is not None:
        cache[shape] = result
    return result


def _shape_log_marginal_fast(
    shape: ShapeNode,
    ctx: _KatzDPContext,
    cache: Optional[Dict] = None,
) -> float:
    """Fast log M(S) using precomputed matrices."""
    root_m = _compute_marginal_vec(shape, ctx, cache)
    return float(logsumexp(ctx.log_root + root_m))


def estimate_log_z_k_katz(
    katz_model,
    operators: List[int],
    x_dim: int,
    max_size: int,
    n_shapes_per_size: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, object]:
    """Estimate log Z_k for each tree-size k under the Katz prior.

    Uses the hybrid approach: Monte Carlo over tree *shapes* combined
    with exact tree dynamic programming over operator labelings.

    Parameters
    ----------
    katz_model : KatzBackoffTreeModel
        Fitted Katz back-off model.
    operators : list of int
        Non-terminal operator IDs (user's operator set).
    x_dim : int
        Number of input variables.
    max_size : int
        Maximum tree size to estimate.
    n_shapes_per_size : int, optional
        Number of shapes to sample per size class.  Default 1000.
    random_state : int or None, optional
        Random seed.

    Returns
    -------
    dict
        ``"log_z_k"`` : dict of {int: float}
            Estimated log Z_k per size.
        ``"shape_counts"`` : dict of {int: int}
            Number of tree shapes per size.
        ``"n_shapes_sampled"`` : dict of {int: int}
            Actual number of shapes sampled per size (may be less than
            *n_shapes_per_size* if total shapes < requested).
        ``"low_ess_sizes"`` : list of int
            Sizes where fewer than 30 shapes were sampled.
    """
    info = _classify_operators(operators, x_dim)
    unary_ops = info["unary_ids"]
    binary_ops = info["binary_ids"]
    has_unary = info["n_u"] > 0
    has_binary = info["n_b"] > 0

    ctx = _KatzDPContext(katz_model, x_dim, unary_ops, binary_ops)

    shape_cts = compute_shape_counts(has_unary, has_binary, max_size)
    rng = np.random.default_rng(random_state)

    log_z_k: Dict[int, float] = {}
    n_sampled: Dict[int, int] = {}
    low_ess: List[int] = []
    dp_cache: Dict = {}

    for k in range(1, max_size + 1):
        n_shapes_k = shape_cts.get(k, 0)
        if n_shapes_k == 0:
            continue

        if n_shapes_k <= n_shapes_per_size:
            all_shapes = enumerate_shapes(k, shape_cts, has_unary,
                                          has_binary)
            n_sampled[k] = n_shapes_k
            log_m_values = np.array(
                [_shape_log_marginal_fast(s, ctx, dp_cache)
                 for s in all_shapes]
            )
            log_z_k[k] = float(logsumexp(log_m_values))
        else:
            m = n_shapes_per_size
            n_sampled[k] = m

            if m < 30:
                low_ess.append(k)

            log_m_values = np.full(m, float("-inf"))
            for i in range(m):
                shape = sample_shape(k, shape_cts, has_unary, has_binary,
                                     rng)
                log_m_values[i] = _shape_log_marginal_fast(shape, ctx,
                                                           dp_cache)

            log_z_k[k] = (log(n_shapes_k)
                          + float(logsumexp(log_m_values))
                          - log(m))

    if low_ess:
        warnings.warn(
            f"Sizes with fewer than 30 sampled shapes (low ESS): "
            f"{low_ess}",
            stacklevel=2,
        )

    return {
        "log_z_k": log_z_k,
        "shape_counts": shape_cts,
        "n_shapes_sampled": n_sampled,
        "low_ess_sizes": low_ess,
        "dp_cache_size": len(dp_cache),
    }


# ------------------------------------------------------------------ #
# Shape-based MC fallback                                             #
# ------------------------------------------------------------------ #


def _label_shape(
    shape: ShapeNode,
    x_dim: int,
    terminal_ops: List[int],
    unary_ops: List[int],
    binary_ops: List[int],
    rng: np.random.Generator,
) -> Tuple[AGraphExpression, float]:
    """Randomly label a shape to produce a concrete AGraph expression.

    Each node is assigned a uniformly random operator of the correct
    arity.  The resulting AGraph is returned together with the log of
    the number of possible labelings (needed for the importance weight).

    Parameters
    ----------
    shape : ShapeNode
        Tree shape to label.
    x_dim : int
        Number of input variables.
    terminal_ops : list of int
        Valid terminal operator IDs (variables + constant).
    unary_ops : list of int
        Valid unary operator IDs.
    binary_ops : list of int
        Valid binary operator IDs.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    agraph : AGraphExpression
        Concrete expression with the given shape.
    log_n_labelings : float
        Log of the total number of valid labelings for this shape
        (i.e. ``n_t^leaves * n_u^unary * n_b^binary``).
    """
    n_t = len(terminal_ops)
    n_u = len(unary_ops)
    n_b = len(binary_ops)

    stack_order: List[Tuple[ShapeNode, List[int]]] = []

    def _traverse(node: ShapeNode) -> int:
        child_indices = []
        for child in node.children:
            child_indices.append(_traverse(child))
        idx = len(stack_order)
        stack_order.append((node, child_indices))
        return idx

    _traverse(shape)
    n_nodes = len(stack_order)

    cmd = np.zeros((n_nodes, 3), dtype=np.uint8)
    n_leaves = 0
    n_unary = 0
    n_binary = 0
    n_constants = 0

    for i, (node, child_idx) in enumerate(stack_order):
        if node.kind == "leaf":
            n_leaves += 1
            op = terminal_ops[rng.integers(n_t)]
            if op == VARIABLE:
                cmd[i] = [VARIABLE, rng.integers(x_dim), 0]
            else:
                cmd[i] = [CONSTANT, n_constants, 0]
                n_constants += 1
        elif node.kind == "unary":
            n_unary += 1
            op = unary_ops[rng.integers(n_u)]
            cmd[i] = [op, child_idx[0], child_idx[0]]
        else:  # binary
            n_binary += 1
            op = binary_ops[rng.integers(n_b)]
            cmd[i] = [op, child_idx[0], child_idx[1]]

    agraph = AGraphExpression()
    agraph.raw_command_array = cmd
    if n_constants > 0:
        agraph.raw_constants = tuple(1.0 for _ in range(n_constants))

    log_n_labelings = (
        n_leaves * (log(n_t) if n_t > 0 else 0.0)
        + n_unary * (log(n_u) if n_u > 0 else 0.0)
        + n_binary * (log(n_b) if n_b > 0 else 0.0)
    )

    return agraph, log_n_labelings


def estimate_log_z_k_mc(
    base_prior,
    x_dim: int,
    operators: List[int],
    max_size: int,
    n_shapes_per_size: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, object]:
    """Estimate log Z_k via shape-based Monte Carlo sampling.

    For each size *k*, samples *m* tree shapes uniformly with
    replacement, randomly labels each shape with one concrete operator
    assignment, evaluates ``P_base`` on the resulting expression, and
    weights by the number of possible labelings.

    The estimator is:

    .. math::

        \\log \\hat{Z}_k = \\log N_{\\text{shapes}}(k)
            + \\operatorname{logsumexp}_i
              \\bigl[\\log N_{\\text{labelings}}(S_i)
                     + \\log P_{\\text{base}}(T_i)\\bigr]
            - \\log m

    When ``base_prior`` is ``None`` (uniform), ``log P_base = 0`` and
    the estimator reduces to the uniform shape-MC estimator.

    Parameters
    ----------
    base_prior : SamplablePrior or None
        Base prior.  ``None`` means improper uniform.
    x_dim : int
        Number of input variables.
    operators : list of int
        Operator IDs.
    max_size : int
        Maximum tree size to estimate.
    n_shapes_per_size : int, optional
        Number of shapes to sample per size.  Default 1000.
    random_state : int or None, optional
        Random seed.

    Returns
    -------
    dict
        ``"log_z_k"`` : dict of {int: float}
        ``"shape_counts"`` : dict of {int: int}
        ``"n_shapes_sampled"`` : dict of {int: int}
        ``"low_ess_sizes"`` : list of int
    """
    info = _classify_operators(operators, x_dim)
    terminal_ops = [VARIABLE] * x_dim + [CONSTANT]
    unary_ops = info["unary_ids"]
    binary_ops = info["binary_ids"]
    has_unary = info["n_u"] > 0
    has_binary = info["n_b"] > 0

    shape_cts = compute_shape_counts(has_unary, has_binary, max_size)
    rng = np.random.default_rng(random_state)

    log_z_k: Dict[int, float] = {}
    n_sampled: Dict[int, int] = {}
    low_ess: List[int] = []

    for k in range(1, max_size + 1):
        n_shapes_k = shape_cts.get(k, 0)
        if n_shapes_k == 0:
            continue

        m = n_shapes_per_size
        n_sampled[k] = m
        if m < 30:
            low_ess.append(k)

        log_values = np.full(m, float("-inf"))
        for i in range(m):
            shape = sample_shape(k, shape_cts, has_unary, has_binary, rng)
            agraph, log_n_lab = _label_shape(
                shape, x_dim, terminal_ops, unary_ops, binary_ops, rng
            )
            if base_prior is not None:
                log_p_base = base_prior._logpdf_single(agraph)
            else:
                log_p_base = 0.0
            log_values[i] = log_n_lab + log_p_base

        log_z_k[k] = (log(n_shapes_k)
                       + float(logsumexp(log_values))
                       - log(m))

    if low_ess:
        warnings.warn(
            f"Sizes with fewer than 30 sampled shapes (low ESS): "
            f"{sorted(low_ess)}",
            stacklevel=2,
        )

    return {
        "log_z_k": log_z_k,
        "shape_counts": shape_cts,
        "n_shapes_sampled": n_sampled,
        "low_ess_sizes": low_ess,
    }
