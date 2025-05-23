import numpy as np
import pytest

from pysips.sampler import sample, run_smc

IMPORTMODULE = sample.__module__


class TestSampleFunction:
    def test_default_kwargs(self, mocker):
        mock_run_smc = mocker.patch(
            f"{IMPORTMODULE}.run_smc",
            return_value=("mock_models", "mock_likelihoods"),
        )

        likelihood = lambda x: x
        proposal = object()
        generator = object()
        seed = 42

        result = sample(likelihood, proposal, generator, seed=seed)

        assert result == ("mock_models", "mock_likelihoods")

        mock_run_smc.assert_called_once()
        args, _ = mock_run_smc.call_args

        assert args[0] == likelihood
        assert args[1] == proposal
        assert args[2] == generator
        assert args[3] is False

        kwargs_passed = args[4]
        rng_passed = args[5]

        assert kwargs_passed == {"num_particles": 5000, "num_mcmc_samples": 10}
        assert isinstance(rng_passed, np.random.Generator)

    def test_custom_kwargs(self, mocker):
        mock_run_smc = mocker.patch(
            f"{IMPORTMODULE}.run_smc", return_value=("mock_models", "mock_likelihoods")
        )

        likelihood = lambda x: x
        proposal = object()
        generator = object()
        custom_kwargs = {"num_particles": 100, "num_mcmc_samples": 3}

        result = sample(likelihood, proposal, generator, kwargs=custom_kwargs, seed=24)

        assert result == ("mock_models", "mock_likelihoods")
        mock_run_smc.assert_called_once()

        args, _ = mock_run_smc.call_args
        assert args[0] == likelihood
        assert args[1] == proposal
        assert args[2] == generator
        assert args[3] is False
        assert args[4] == custom_kwargs


class TestRunSMC:
    @pytest.mark.parametrize("multiproc", [True, False])
    def test_functionality(self, mocker, multiproc):
        mock_rng_instance = mocker.Mock(name="rngInstance")
        mock_rng = mocker.patch(
            f"{IMPORTMODULE}.np.random.default_rng", return_value=mock_rng_instance
        )

        mock_prior_instance = mocker.Mock(name="PriorInstance")
        mock_prior = mocker.patch(
            f"{IMPORTMODULE}.Prior", return_value=mock_prior_instance
        )

        mock_mcmc_instance = mocker.Mock(name="MetropolisInstance")
        mock_metropolis = mocker.patch(
            f"{IMPORTMODULE}.Metropolis", return_value=mock_mcmc_instance
        )

        mock_kernel_instance = mocker.Mock(name="VectorMCMCKernelInstance")
        mock_vector_kernel = mocker.patch(
            f"{IMPORTMODULE}.VectorMCMCKernel", return_value=mock_kernel_instance
        )

        mock_sampler_instance = mocker.Mock(name="AdaptiveSamplerInstance")
        mock_adaptive_sampler = mocker.patch(
            f"{IMPORTMODULE}.AdaptiveSampler", return_value=mock_sampler_instance
        )

        dummy_params = np.array([[1], [2], [3]])
        dummy_step = mocker.Mock(params=dummy_params)
        mock_sampler_instance.sample.return_value = ([dummy_step], None)

        likelihood = mocker.Mock(side_effect=lambda x: x * 10)

        proposal = "proposal"
        generator = "generator"
        kwargs = {"num_particles": 3, "num_mcmc_samples": 4}

        models, likelihoods = sample(
            likelihood,
            proposal,
            generator,
            multiprocess=multiproc,
            kwargs=kwargs,
            seed=0,
        )

        mock_prior.assert_called_once_with(generator)

        mock_metropolis.assert_called_once_with(
            likelihood=likelihood,
            proposal=proposal,
            prior=mock_prior_instance,
            multiprocess=multiproc,
        )

        mock_vector_kernel.assert_called_once_with(
            mock_mcmc_instance, param_order=["f"], rng=mock_rng_instance
        )

        mock_adaptive_sampler.assert_called_once_with(mock_kernel_instance)

        mock_sampler_instance.sample.assert_called_once_with(**kwargs)

        assert mock_sampler_instance._mutator._compute_cov is False

        expected_models = dummy_params[:, 0].tolist()
        assert models == expected_models

        expected_likelihoods = [m * 10 for m in expected_models]
        assert likelihoods == expected_likelihoods
        assert likelihood.call_count == len(expected_models)
