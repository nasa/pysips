"""Tests for size_calibration_z_k module."""

from math import exp, log
from unittest.mock import MagicMock

import numpy as np
import pytest

from pysips.priors.size_calibration_z_k import (
    ShapeNode,
    compute_labeled_tree_counts,
    compute_shape_counts,
    enumerate_shapes,
    estimate_log_z_k_katz,
    estimate_log_z_k_mc,
    sample_shape,
    _label_shape,
)

# Operator IDs (from bingo)
VARIABLE = 0
CONSTANT = 1
ADDITION = 3
SUBTRACTION = 4
MULTIPLICATION = 5
DIVISION = 6
SIN = 15
COS = 16
EXPONENTIAL = 13


# ===================================================================
# Analytical: compute_labeled_tree_counts
# ===================================================================


class TestComputeLabeledTreeCounts:
    """Tests for compute_labeled_tree_counts."""

    def test_binary_only_catalan(self):
        """n_t=1, n_u=0, n_b=1 produces Catalan numbers at odd sizes."""
        result = compute_labeled_tree_counts(n_t=1, n_u=0, n_b=1, max_size=7)

        assert 1 in result
        assert 2 not in result
        assert 4 not in result
        np.testing.assert_almost_equal(exp(result[1]), 1.0)
        np.testing.assert_almost_equal(exp(result[3]), 1.0)
        np.testing.assert_almost_equal(exp(result[5]), 2.0)
        np.testing.assert_almost_equal(exp(result[7]), 5.0)

    def test_unary_only_chain(self):
        """n_t=1, n_u=2, n_b=0 produces a[k] = 2^(k-1)."""
        result = compute_labeled_tree_counts(n_t=1, n_u=2, n_b=0, max_size=4)

        np.testing.assert_almost_equal(exp(result[1]), 1.0)
        np.testing.assert_almost_equal(exp(result[2]), 2.0)
        np.testing.assert_almost_equal(exp(result[3]), 4.0)
        np.testing.assert_almost_equal(exp(result[4]), 8.0)

    def test_empty_for_zero_max_size(self):
        """max_size=0 returns empty dict."""
        assert compute_labeled_tree_counts(2, 1, 1, 0) == {}

    def test_no_operators(self):
        """n_u=0, n_b=0 produces only size 1."""
        result = compute_labeled_tree_counts(n_t=3, n_u=0, n_b=0, max_size=5)
        assert set(result.keys()) == {1}
        np.testing.assert_almost_equal(exp(result[1]), 3.0)


# ===================================================================
# Shape utilities: ShapeNode, compute_shape_counts, sample, enumerate
# ===================================================================


class TestShapeNode:
    """Tests for ShapeNode equality and hashing."""

    def test_leaf_equality(self):
        """Two leaf nodes are equal."""
        assert ShapeNode("leaf") == ShapeNode("leaf")

    def test_unary_equality(self):
        """Unary nodes with same children are equal."""
        child = ShapeNode("leaf")
        assert ShapeNode("unary", (child,)) == ShapeNode("unary", (child,))

    def test_different_kinds_not_equal(self):
        """Nodes with different kinds are not equal."""
        assert ShapeNode("leaf") != ShapeNode("unary", (ShapeNode("leaf"),))

    def test_hashable_in_set(self):
        """ShapeNodes can be stored in a set."""
        s = {ShapeNode("leaf"), ShapeNode("leaf")}
        assert len(s) == 1


class TestComputeShapeCounts:
    """Tests for compute_shape_counts."""

    def test_binary_only_catalan(self):
        """Binary-only shapes follow Catalan numbers at odd sizes."""
        sc = compute_shape_counts(has_unary=False, has_binary=True, max_size=7)

        assert sc[1] == 1
        assert 2 not in sc  # no even sizes for binary-only
        assert sc[3] == 1
        assert sc[5] == 2
        assert sc[7] == 5

    def test_unary_only(self):
        """Unary-only shapes: exactly 1 per size (chains)."""
        sc = compute_shape_counts(has_unary=True, has_binary=False, max_size=5)
        for k in range(1, 6):
            assert sc[k] == 1

    def test_mixed(self):
        """Both unary and binary: s[2]=1, s[3]=2."""
        sc = compute_shape_counts(has_unary=True, has_binary=True, max_size=3)
        assert sc[1] == 1
        assert sc[2] == 1
        assert sc[3] == 2


class TestSampleShape:
    """Tests for sample_shape."""

    def test_size_1_is_leaf(self):
        """Size 1 always produces a leaf."""
        rng = np.random.default_rng(42)
        sc = compute_shape_counts(True, True, 5)
        shape = sample_shape(1, sc, True, True, rng)
        assert shape.kind == "leaf"
        assert shape.children == ()

    def test_binary_size_3(self):
        """Size 3 binary-only: binary(leaf, leaf)."""
        rng = np.random.default_rng(42)
        sc = compute_shape_counts(False, True, 5)
        shape = sample_shape(3, sc, False, True, rng)
        assert shape.kind == "binary"
        assert len(shape.children) == 2
        assert all(c.kind == "leaf" for c in shape.children)

    def test_produces_correct_node_count(self):
        """Sampled shapes have the requested number of nodes."""
        rng = np.random.default_rng(0)
        sc = compute_shape_counts(True, True, 10)

        def _count_nodes(node):
            return 1 + sum(_count_nodes(c) for c in node.children)

        for k in range(1, 8):
            shape = sample_shape(k, sc, True, True, rng)
            assert _count_nodes(shape) == k


class TestEnumerateShapes:
    """Tests for enumerate_shapes."""

    def test_count_matches_shape_counts(self):
        """Number of enumerated shapes equals compute_shape_counts."""
        sc = compute_shape_counts(True, True, 6)
        for k in range(1, 7):
            shapes = enumerate_shapes(k, sc, True, True)
            assert len(shapes) == sc[k]

    def test_binary_only_size_5(self):
        """Binary-only size 5: exactly 2 shapes (Catalan(2))."""
        sc = compute_shape_counts(False, True, 5)
        shapes = enumerate_shapes(5, sc, False, True)
        assert len(shapes) == 2
        # All should be distinct
        assert len(set(hash(s) for s in shapes)) == 2


# ===================================================================
# Analytical vs shape-based cross-check
# ===================================================================


class TestAnalyticalShapeCrossCheck:
    """Verify analytical tree counts equal shape-count * labeling count."""

    def test_binary_only_cross_check(self):
        """For n_t=2, n_b=1, tree count = sum over shapes of n_t^leaves * n_b^internal."""
        n_t, n_b = 2, 1
        max_size = 7
        analytical = compute_labeled_tree_counts(n_t, 0, n_b, max_size)
        sc = compute_shape_counts(False, True, max_size)

        for k in sorted(analytical.keys()):
            shapes = enumerate_shapes(k, sc, False, True)
            shape_total = 0
            for s in shapes:
                n_leaves, n_binary = _count_leaf_binary(s)
                shape_total += n_t**n_leaves * n_b**n_binary
            np.testing.assert_almost_equal(
                exp(analytical[k]),
                shape_total,
                err_msg=f"Mismatch at size {k}",
            )


def _count_leaf_binary(node):
    """Count leaves and binary nodes in a shape tree."""
    if node.kind == "leaf":
        return 1, 0
    leaves, binaries = 0, 0
    for child in node.children:
        cl, cb = _count_leaf_binary(child)
        leaves += cl
        binaries += cb
    if node.kind == "binary":
        binaries += 1
    return leaves, binaries


# ===================================================================
# Katz hybrid: estimate_log_z_k_katz
# ===================================================================


class TestEstimateLogZkKatz:
    """Tests for estimate_log_z_k_katz with mock Katz model."""

    @staticmethod
    def _make_uniform_katz(vocab):
        """Katz model where all conditionals are uniform over vocab."""
        n = len(vocab)
        lp = log(1.0 / n) if n > 0 else float("-inf")

        left_model = MagicMock()
        left_model.log_prob = MagicMock(return_value=lp)
        right_model = MagicMock()
        right_model.log_prob = MagicMock(return_value=lp)

        model = MagicMock()
        model.left_model = left_model
        model.right_model = right_model
        return model

    def test_produces_finite_values(self):
        """Katz Z_k produces finite log Z_k for small sizes."""
        vocab = [VARIABLE, CONSTANT, ADDITION]
        model = self._make_uniform_katz(vocab)

        result = estimate_log_z_k_katz(
            model,
            operators=[ADDITION],
            x_dim=1,
            max_size=5,
            n_shapes_per_size=100,
            random_state=42,
        )

        log_z_k = result["log_z_k"]
        assert len(log_z_k) > 0
        for k, v in log_z_k.items():
            assert np.isfinite(v), f"Non-finite log Z_k at size {k}"

    def test_multiple_sizes_present(self):
        """Z_k produces values across multiple tree sizes."""
        vocab = [VARIABLE, CONSTANT, ADDITION, SIN]
        model = self._make_uniform_katz(vocab)

        result = estimate_log_z_k_katz(
            model,
            operators=[ADDITION, SIN],
            x_dim=1,
            max_size=7,
            n_shapes_per_size=100,
            random_state=42,
        )

        log_z_k = result["log_z_k"]
        assert len(log_z_k) >= 3

    def test_returns_shape_counts(self):
        """Result includes shape_counts dict."""
        vocab = [VARIABLE, CONSTANT, ADDITION]
        model = self._make_uniform_katz(vocab)

        result = estimate_log_z_k_katz(
            model,
            operators=[ADDITION],
            x_dim=1,
            max_size=3,
            random_state=42,
        )

        assert "shape_counts" in result
        assert "n_shapes_sampled" in result


# ===================================================================
# MC fallback: estimate_log_z_k_mc
# ===================================================================


class TestEstimateLogZkMC:
    """Tests for estimate_log_z_k_mc."""

    def test_uniform_converges_to_analytical(self):
        """MC with base_prior=None converges to analytical tree counts."""
        operators = [ADDITION]
        x_dim = 1
        max_size = 5

        # Analytical: log Z_k for uniform = log(tree_count)
        analytical = compute_labeled_tree_counts(
            n_t=x_dim + 1, n_u=0, n_b=1, max_size=max_size
        )

        result = estimate_log_z_k_mc(
            base_prior=None,
            x_dim=x_dim,
            operators=operators,
            max_size=max_size,
            n_shapes_per_size=2000,
            random_state=42,
        )

        mc_z_k = result["log_z_k"]
        for k in analytical:
            if k in mc_z_k:
                np.testing.assert_almost_equal(
                    mc_z_k[k],
                    analytical[k],
                    decimal=1,
                    err_msg=f"MC Z_k diverges from analytical at size {k}",
                )

    def test_produces_finite_values(self):
        """MC Z_k produces finite values for small sizes."""
        result = estimate_log_z_k_mc(
            base_prior=None,
            x_dim=1,
            operators=[ADDITION, SIN],
            max_size=5,
            n_shapes_per_size=100,
            random_state=42,
        )

        for k, v in result["log_z_k"].items():
            assert np.isfinite(v), f"Non-finite log Z_k at size {k}"


# ===================================================================
# _label_shape
# ===================================================================


class TestLabelShape:
    """Tests for _label_shape."""

    def test_produces_valid_agraph(self):
        """Labeled shape produces a valid AGraphExpression."""
        shape = ShapeNode("binary", (ShapeNode("leaf"), ShapeNode("leaf")))
        rng = np.random.default_rng(42)

        agraph, log_n = _label_shape(
            shape,
            x_dim=2,
            terminal_ops=[VARIABLE, VARIABLE, CONSTANT],
            unary_ops=[],
            binary_ops=[ADDITION],
            rng=rng,
        )

        assert agraph.raw_command_array.shape == (3, 3)
        assert np.isfinite(log_n)

    def test_log_labelings_correct(self):
        """Log labelings = n_leaves*log(n_t) + n_binary*log(n_b)."""
        # binary(leaf, leaf): 2 leaves, 1 binary
        shape = ShapeNode("binary", (ShapeNode("leaf"), ShapeNode("leaf")))
        rng = np.random.default_rng(42)
        n_t = 3  # 2 vars + 1 constant
        n_b = 2

        _, log_n = _label_shape(
            shape,
            x_dim=2,
            terminal_ops=[VARIABLE, VARIABLE, CONSTANT],
            unary_ops=[],
            binary_ops=[ADDITION, MULTIPLICATION],
            rng=rng,
        )

        expected = 2 * log(n_t) + 1 * log(n_b)
        np.testing.assert_almost_equal(log_n, expected)
