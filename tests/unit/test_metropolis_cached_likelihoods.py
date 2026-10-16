"""Regression coverage for kernels supplying initial particle likelihoods."""

import numpy as np
import pytest
from smcpy import VectorMCMCKernel
from smcpy.smc.particles import Particles

from pysips.metropolis import Metropolis


class Proposal:
    """Deterministic proposal with a record of the updated gene pool."""

    def __init__(self):
        self.gene_pool = None

    def __call__(self, value):
        return value + 0.25

    def update(self, gene_pool):
        self.gene_pool = gene_pool.copy()


@pytest.fixture
def chain():
    calls = []

    def likelihood(value):
        calls.append(value)
        return -float(value) ** 2

    proposal = Proposal()
    mcmc = Metropolis(likelihood, proposal, prior=None)
    kernel = VectorMCMCKernel(
        mcmc, param_order=["f"], rng=np.random.default_rng(42)
    )
    return mcmc, kernel, proposal, calls


@pytest.mark.parametrize("cached", [[-1., -4.], np.array([-1., -4.]),
                                    np.array([[-1.], [-4.]]),
                                    np.array([[-1., -4.]])])
@pytest.mark.parametrize("keyword", [False, True])
def test_cached_values_are_accepted_with_original_reevaluation(
        chain, cached, keyword):
    mcmc, _, proposal, calls = chain
    inputs = np.array([[1.], [2.]])
    original = np.array(cached, copy=True)
    if keyword:
        output, values = mcmc.smc_metropolis(
            inputs, 0, log_likes=cached
        )
    else:
        output, values = mcmc.smc_metropolis(inputs, 0, None, cached)
    np.testing.assert_array_equal(output, inputs)
    np.testing.assert_array_equal(values, [[-1.], [-4.]])
    np.testing.assert_array_equal(cached, original)
    np.testing.assert_array_equal(proposal.gene_pool, inputs.ravel())
    assert calls == [1., 2.]


@pytest.mark.parametrize("explicit_none", [False, True])
def test_no_cache_keeps_original_evaluation_path(chain, explicit_none):
    mcmc, _, proposal, calls = chain
    inputs = np.array([[1.], [2.]])
    if explicit_none:
        output, values = mcmc.smc_metropolis(inputs, 0, None, None)
    else:
        output, values = mcmc.smc_metropolis(inputs, 0)
    np.testing.assert_array_equal(output, inputs)
    np.testing.assert_array_equal(values, [[-1.], [-4.]])
    np.testing.assert_array_equal(proposal.gene_pool, inputs.ravel())
    assert calls == [1., 2.]


@pytest.mark.parametrize("cached", [None, [-100., -400.], [float("nan"), 0.]])
def test_initial_likelihoods_are_refit_rather_than_taken_from_cache(chain, cached):
    mcmc, _, proposal, calls = chain
    inputs = np.array([[1.], [2.]])
    output, values = mcmc.smc_metropolis(inputs, 0, None, cached)
    np.testing.assert_array_equal(output, inputs)
    np.testing.assert_array_equal(values, [[-1.], [-4.]])
    np.testing.assert_array_equal(proposal.gene_pool, inputs.ravel())
    assert calls == [1., 2.]


@pytest.mark.parametrize("num_samples", [0, 1, 3])
def test_current_smcpy_kernel_can_mutate_particles(chain, num_samples):
    _, kernel, proposal, calls = chain
    particles = Particles(
        {"f": np.array([1., 2.])}, [-1., -4.], [0., 0.]
    )
    params, values = kernel.mutate_particles(particles, num_samples, None)
    np.testing.assert_allclose(values.ravel(), -params["f"] ** 2)
    np.testing.assert_array_equal(proposal.gene_pool, params["f"])
    assert len(calls) == 2 * (num_samples + 1)
    np.testing.assert_array_equal(particles.params.ravel(), [1., 2.])


@pytest.mark.parametrize("num_samples", [0, 1, 4])
def test_cached_and_uncached_chains_have_identical_seeded_results(
        chain, num_samples):
    mcmc, _, proposal, calls = chain
    inputs = np.array([[1.], [2.]])
    mcmc.rng = np.random.default_rng(91)
    expected_inputs, expected_likes = mcmc.smc_metropolis(
        inputs.copy(), num_samples
    )
    expected_calls = len(calls)
    calls.clear()
    mcmc.rng = np.random.default_rng(91)
    actual_inputs, actual_likes = mcmc.smc_metropolis(
        inputs.copy(), num_samples, None, np.array([-1., -4.])
    )
    np.testing.assert_array_equal(actual_inputs, expected_inputs)
    np.testing.assert_array_equal(actual_likes, expected_likes)
    np.testing.assert_array_equal(proposal.gene_pool, actual_inputs.ravel())
    assert len(calls) == expected_calls
