import inspect
import numpy as np
import pytest
from pysips.sampler import Metropolis  # Adjust import as needed

IMPORTMODULE = Metropolis.__module__


def dummy_likelihood(x):
    return x.value * 2


@pytest.fixture
def metropolis(mocker):
    mock_likelihood = dummy_likelihood
    mock_proposal = mocker.Mock()
    return Metropolis(
        likelihood=mock_likelihood,
        proposal=mock_proposal,
        prior=None,
        multiprocess=False,
    )


class DummyGraph:
    def __init__(self, value):
        self.value = value
        self.fitness = None


class TestMetropolis:
    def test_evaluate_model_returns_none(self, metropolis):
        assert metropolis.evaluate_model() is None

    def test_evaluate_log_priors_returns_ones(self, metropolis):
        x = np.empty((5, 1))
        np.testing.assert_array_equal(
            metropolis.evaluate_log_priors(x), np.ones((5, 1))
        )

    def test_evaluate_log_likelihood_no_multiproc(self, metropolis):
        graphs = np.array(
            [[DummyGraph(1)], [DummyGraph(2)], [DummyGraph(3)]], dtype=object
        )
        result = metropolis.evaluate_log_likelihood(graphs)
        np.testing.assert_array_equal(result, np.c_[[2, 4, 6]])

    def test_evaluate_log_likelihood_multiproc(self, mocker, metropolis):
        metropolis._is_multiprocess = True

        dummy_pool = mocker.MagicMock()
        dummy_pool.__enter__.return_value = dummy_pool
        dummy_pool.__exit__.return_value = None
        dummy_pool.map.side_effect = lambda func, iterable: list(map(func, iterable))

        mocker.patch(f"{IMPORTMODULE}.Pool", return_value=dummy_pool)

        graphs = np.array(
            [[DummyGraph(1)], [DummyGraph(2)], [DummyGraph(3)]], dtype=object
        )
        result = metropolis.evaluate_log_likelihood(graphs)

        expected = np.c_[[2, 4, 6]]
        np.testing.assert_array_equal(result, expected)

        for g, l in zip(graphs.flatten(), expected.flatten()):
            assert g.fitness == l


def test_smc_metropolis(mocker, metropolis):
    inputs = np.array([[DummyGraph(1)], [DummyGraph(2)]], dtype=object)
    num_samples = 2

    mocker.patch.object(
        metropolis,
        "_initialize_probabilities",
        return_value=(np.array([0, 0]), np.array([10, 10])),
    )
    mocker.patch.object(
        metropolis,
        "_perform_mcmc_step",
        side_effect=lambda inputs, a, b, c: (inputs, b, None, None),
    )

    update_spy = mocker.spy(metropolis._equ_proposal, "update")

    out_inputs, out_log_like = metropolis.smc_metropolis(inputs, num_samples)

    update_spy.assert_called_once()
    called_args, called_kwargs = update_spy.call_args

    assert "gene_pool" in called_kwargs
    np.testing.assert_array_equal(called_kwargs["gene_pool"], inputs.flatten())
    assert np.array_equal(out_inputs, inputs)
    assert np.array_equal(out_log_like, np.array([10, 10]))


def test_smc_metropolis_accepts_cov_kwarg(metropolis):
    sig = inspect.signature(metropolis.smc_metropolis)
    assert "cov" in sig.parameters


def test_proposal_applies_equ_proposal_elementwise():
    equ_proposal = lambda x: x * 2
    m = Metropolis(prior=None, likelihood=lambda x: 0, proposal=equ_proposal)

    x = np.c_[[1.0, 2.0]]
    result = m.proposal(x, None)

    expected = np.c_[[2.0, 4.0]]
    np.testing.assert_array_equal(result, expected)
