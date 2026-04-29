"""Unit tests for KatzPrior and load_katz_model."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pysips.priors import KatzPrior, load_katz_model, save_katz_model
from pysips.priors.katz_backoff import KatzBackoffModel, KatzBackoffTreeModel
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph import (
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    VARIABLE,
    CONSTANT,
)

SAMPLEABLEPRIOR_MODULE = "pysips.priors.samplable_prior"
KATZ_PRIOR_MODULE = "pysips.priors.katz_prior"
BINGO_MIXIN_MODULE = "pysips.bingo_proposal_mixin"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_katz_tree_model(n=2, vocab=None):
    """Return a minimal KatzBackoffTreeModel with the given vocab."""
    if vocab is None:
        vocab = [ADDITION, MULTIPLICATION]
    left = KatzBackoffModel(max_order=n, vocabulary=vocab)
    left.fit([(ADDITION,), (MULTIPLICATION,)])
    right = KatzBackoffModel(max_order=n, vocabulary=vocab)
    right.fit([(ADDITION,), (MULTIPLICATION,)])
    return KatzBackoffTreeModel(left_model=left, right_model=right, n=n)


def _make_mock_tree_model(vocab=None, n=2):
    """Return a mock KatzBackoffTreeModel."""
    if vocab is None:
        vocab = [ADDITION, MULTIPLICATION]
    mock_model = MagicMock(spec=KatzBackoffTreeModel)
    mock_model.n = n
    mock_model.left_model = MagicMock()
    mock_model.left_model.vocabulary = vocab
    mock_model.right_model = MagicMock()
    mock_model.right_model.vocabulary = vocab
    mock_model.log_prob_phrases.return_value = -1.0
    return mock_model


# ---------------------------------------------------------------------------
# KatzPrior.__init__
# ---------------------------------------------------------------------------


class TestKatzPriorInit:
    def test_stores_model(self):
        model = _make_mock_tree_model()
        prior = KatzPrior(model, operators=[ADDITION])
        assert prior.model is model

    def test_explicit_operators_used(self):
        model = _make_mock_tree_model()
        prior = KatzPrior(model, operators=[ADDITION, SUBTRACTION])
        assert set(prior.operators) == {ADDITION, SUBTRACTION}

    def test_operators_derived_from_vocab_when_none(self):
        """When operators=None, non-terminal vocab tokens become operators."""
        vocab = [ADDITION, MULTIPLICATION]
        model = _make_mock_tree_model(vocab=vocab)
        prior = KatzPrior(model, operators=None)
        assert set(prior.operators) == {ADDITION, MULTIPLICATION}

    def test_operators_fallback_to_common_when_vocab_all_terminals(self):
        """Fallback to _COMMON_OPERATORS when vocab has no non-terminals."""
        model = _make_mock_tree_model(vocab=[VARIABLE, CONSTANT])
        prior = KatzPrior(model, operators=None)
        # Should not be empty — _COMMON_OPERATORS is used
        assert len(prior.operators) > 0
        assert VARIABLE not in prior.operators
        assert CONSTANT not in prior.operators

    def test_x_dim_stored(self):
        model = _make_mock_tree_model()
        prior = KatzPrior(model, operators=[ADDITION], x_dim=3)
        assert prior.x_dim == 3

    def test_x_dim_defaults_to_none(self):
        model = _make_mock_tree_model()
        prior = KatzPrior(model, operators=[ADDITION])
        assert prior.x_dim is None

    def test_n_property_reflects_model_n(self):
        model = _make_mock_tree_model(n=3)
        prior = KatzPrior(model, operators=[ADDITION])
        assert prior.n == 3


# ---------------------------------------------------------------------------
# KatzPrior.logpdf
# ---------------------------------------------------------------------------


class TestKatzPriorLogpdf:
    def test_returns_correct_shape(self):
        model = _make_mock_tree_model()
        prior = KatzPrior(model, operators=[ADDITION])
        mock_ags = [MagicMock() for _ in range(5)]
        result = prior.logpdf(mock_ags)
        assert result.shape == (5, 1)

    def test_returns_correct_shape_single(self):
        model = _make_mock_tree_model()
        prior = KatzPrior(model, operators=[ADDITION])
        result = prior.logpdf([MagicMock()])
        assert result.shape == (1, 1)

    def test_calls_model_log_prob_phrases(self, mocker):
        model = _make_mock_tree_model()
        mock_extract = mocker.patch(
            f"{KATZ_PRIOR_MODULE}.extract_phrases",
            return_value=([(ADDITION,)], []),
        )
        prior = KatzPrior(model, operators=[ADDITION])
        ag = MagicMock(spec=[])  # not an EvolvableExpression
        prior.logpdf([ag])
        mock_extract.assert_called_once_with(ag, model.n)
        model.log_prob_phrases.assert_called_once()

    def test_unwraps_evolvable_expression(self, mocker):
        model = _make_mock_tree_model()
        mock_extract = mocker.patch(
            f"{KATZ_PRIOR_MODULE}.extract_phrases",
            return_value=([], []),
        )
        inner = MagicMock()
        evolvable = MagicMock(spec=EvolvableExpression)
        evolvable.expression = inner

        prior = KatzPrior(model, operators=[ADDITION])
        prior.logpdf([evolvable])

        # extract_phrases should have been called with the inner expression
        mock_extract.assert_called_once_with(inner, model.n)

    def test_uses_model_log_prob_value(self, mocker):
        model = _make_mock_tree_model()
        model.log_prob_phrases.return_value = -2.5
        mocker.patch(
            f"{KATZ_PRIOR_MODULE}.extract_phrases",
            return_value=([(ADDITION,)], []),
        )
        prior = KatzPrior(model, operators=[ADDITION])
        result = prior.logpdf([MagicMock(spec=[])])
        np.testing.assert_almost_equal(result[0, 0], -2.5)

    def test_multiple_expressions_scored_independently(self, mocker):
        model = _make_mock_tree_model()
        model.log_prob_phrases.side_effect = [-1.0, -3.0]
        mocker.patch(
            f"{KATZ_PRIOR_MODULE}.extract_phrases",
            return_value=([(ADDITION,)], []),
        )
        prior = KatzPrior(model, operators=[ADDITION])
        result = prior.logpdf([MagicMock(spec=[]), MagicMock(spec=[])])
        np.testing.assert_almost_equal(result[0, 0], -1.0)
        np.testing.assert_almost_equal(result[1, 0], -3.0)


# ---------------------------------------------------------------------------
# KatzPrior.rvs
# ---------------------------------------------------------------------------


class TestKatzPriorRvs:
    def test_rvs_raises_without_x_dim(self):
        model = _make_mock_tree_model()
        prior = KatzPrior(model, operators=[ADDITION])  # no x_dim
        with pytest.raises(ValueError, match="Cannot sample without x_dim"):
            prior.rvs(5)

    def test_rvs_calls_sample_with_correct_params(self, mocker):
        mock_models = [MagicMock(spec=[]) for _ in range(4)]
        mock_sample = mocker.patch(
            f"{SAMPLEABLEPRIOR_MODULE}.sample",
            return_value=(mock_models, None, None),
        )
        # Prevent the real generator/proposal from running (avoids pool loop)
        mocker.patch(f"{BINGO_MIXIN_MODULE}.ComponentGenerator", autospec=True)
        mock_agraph_gen = MagicMock(side_effect=lambda: MagicMock(spec=[]))
        mocker.patch(
            f"{BINGO_MIXIN_MODULE}.AGraphGenerator",
            autospec=True,
            return_value=mock_agraph_gen,
        )
        mocker.patch(f"{BINGO_MIXIN_MODULE}.MutationProposal", autospec=True)
        mocker.patch(f"{BINGO_MIXIN_MODULE}.CrossoverProposal", autospec=True)
        mocker.patch(f"{BINGO_MIXIN_MODULE}.RandomChoiceProposal", autospec=True)

        model = _make_mock_tree_model()
        prior = KatzPrior(
            model,
            operators=[ADDITION],
            x_dim=2,
            num_mcmc_samples=3,
            target_ess=0.7,
        )
        result = prior.rvs(4)
        mock_sample.assert_called_once()
        call_kwargs = mock_sample.call_args.kwargs
        assert call_kwargs["kwargs"]["num_particles"] == 4
        assert call_kwargs["kwargs"]["num_mcmc_samples"] == 3
        assert call_kwargs["kwargs"]["target_ess"] == 0.7
        assert result.shape == (4, 1)

    @pytest.mark.parametrize(
        "instance_seed,call_seed,expected",
        [
            (42, None, 42),
            (42, 99, 99),
        ],
    )
    def test_rvs_random_state(self, mocker, instance_seed, call_seed, expected):
        mock_sample = mocker.patch(
            f"{SAMPLEABLEPRIOR_MODULE}.sample",
            return_value=([MagicMock(spec=[])], None, None),
        )
        mocker.patch(f"{BINGO_MIXIN_MODULE}.ComponentGenerator", autospec=True)
        mock_agraph_gen = MagicMock(side_effect=lambda: MagicMock(spec=[]))
        mocker.patch(
            f"{BINGO_MIXIN_MODULE}.AGraphGenerator",
            autospec=True,
            return_value=mock_agraph_gen,
        )
        mocker.patch(f"{BINGO_MIXIN_MODULE}.MutationProposal", autospec=True)
        mocker.patch(f"{BINGO_MIXIN_MODULE}.CrossoverProposal", autospec=True)
        mocker.patch(f"{BINGO_MIXIN_MODULE}.RandomChoiceProposal", autospec=True)

        model = _make_mock_tree_model()
        prior = KatzPrior(
            model, operators=[ADDITION], x_dim=2, random_state=instance_seed
        )
        prior.rvs(1, random_state=call_seed)
        assert mock_sample.call_args.kwargs["seed"] == expected


# ---------------------------------------------------------------------------
# load_katz_model
# ---------------------------------------------------------------------------


class TestLoadKatzModel:
    def test_loads_prefit_wikipedia(self):
        model = load_katz_model(n=2, corpus="wikipedia")
        assert isinstance(model, KatzBackoffTreeModel)
        assert model.n == 2

    def test_raises_when_fit_if_missing_false(self):
        with pytest.raises(FileNotFoundError, match="No pre-fit Katz model"):
            load_katz_model(n=2, corpus="nonexistent", fit_if_missing=False)

    def test_fit_on_the_fly(self, tmp_path, mocker):
        """Test that load_katz_model fits and saves when prefit is missing."""
        # Point the data dir to a temp directory
        mocker.patch("pysips.priors.katz_prior.KATZ_MODEL_DIR", tmp_path)
        mock_corpus = [MagicMock()]
        mock_load_corpus = mocker.patch(
            "pysips.priors.data.load_corpus.load_corpus", return_value=mock_corpus
        )
        mock_model = _make_katz_tree_model(n=2)
        mock_fit = mocker.patch(
            "pysips.priors.katz_fitting.fit_katz_model", return_value=mock_model
        )

        result = load_katz_model(n=2, corpus="feynman")

        mock_load_corpus.assert_called_once_with("feynman")
        mock_fit.assert_called_once_with(mock_corpus, n=2)
        assert result.n == 2
        # Verify it was saved
        saved_path = tmp_path / "default_katz_n2_feynman.json"
        assert saved_path.exists()


# ---------------------------------------------------------------------------
# save_katz_model round-trip
# ---------------------------------------------------------------------------


class TestSaveKatzModel:
    def test_round_trip(self):
        original = _make_katz_tree_model(n=2)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        save_katz_model(original, path)
        with open(path) as f:
            loaded = KatzBackoffTreeModel.from_dict(json.load(f))
        assert loaded.n == original.n

    def test_saved_file_is_valid_json(self):
        model = _make_katz_tree_model(n=2)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        save_katz_model(model, path)
        with open(path) as f:
            data = json.load(f)
        assert "n" in data
        assert "left_model" in data
        assert "right_model" in data
