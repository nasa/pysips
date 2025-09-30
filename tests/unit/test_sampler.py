import numpy as np
import pytest

from pysips.sampler import sample, run_smc

IMPORTMODULE = sample.__module__


@pytest.fixture
def sampler_mocks(mocker):
    """Fixture that sets up common mocks for the samplers tests."""
    # Mock all the dependencies
    mocker.patch(f"{IMPORTMODULE}.Prior")
    mocker.patch(f"{IMPORTMODULE}.Metropolis")
    mock_kernel = mocker.patch(f"{IMPORTMODULE}.VectorMCMCKernel")

    # Mock the samplers
    mock_fixed_time_sampler = mocker.patch(f"{IMPORTMODULE}.FixedTimeSampler")
    mock_max_step_sampler = mocker.patch(f"{IMPORTMODULE}.MaxStepSampler")
    mock_adaptive_sampler = mocker.patch(f"{IMPORTMODULE}.AdaptiveSampler")

    # Configure mock samplers with common behavior
    for sampler_mock in [
        mock_fixed_time_sampler,
        mock_max_step_sampler,
        mock_adaptive_sampler,
    ]:
        mock_instance = mocker.Mock()
        mock_instance.sample.return_value = (
            [mocker.Mock(params=np.array([[1]]))],
            None,
        )
        mock_instance.phi_sequence = [1.0]
        mock_instance._mutator = mocker.Mock()
        sampler_mock.return_value = mock_instance

    # Mock likelihood
    mock_likelihood = mocker.Mock(return_value=1.0)

    return {
        "kernel": mock_kernel,
        "fixed_time_sampler": mock_fixed_time_sampler,
        "max_step_sampler": mock_max_step_sampler,
        "adaptive_sampler": mock_adaptive_sampler,
        "likelihood": mock_likelihood,
    }


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
        assert args[3] is None
        assert args[4] is None
        assert args[5] is False

        kwargs_passed = args[6]
        rng_passed = args[7]
        checkpoint_file_passed = args[8]
        show_progress_bar_passed = args[9]

        assert kwargs_passed == {"num_particles": 5000, "num_mcmc_samples": 10}
        assert isinstance(rng_passed, np.random.Generator)
        assert checkpoint_file_passed is None
        assert show_progress_bar_passed is True

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
        assert args[3] is None
        assert args[4] is None
        assert args[5] is False
        assert args[6] == custom_kwargs
        # args[7] is rng
        assert args[8] is None  # checkpoint_file
        assert args[9] is True  # show_progress_bar (default)

    def test_show_progress_bar_false(self, mocker):
        """Test that show_progress_bar=False gets passed through correctly."""
        mock_run_smc = mocker.patch(
            f"{IMPORTMODULE}.run_smc", return_value=("mock_models", "mock_likelihoods")
        )

        likelihood = lambda x: x
        proposal = object()
        generator = object()

        result = sample(likelihood, proposal, generator, show_progress_bar=False, seed=42)

        assert result == ("mock_models", "mock_likelihoods")
        mock_run_smc.assert_called_once()

        args, _ = mock_run_smc.call_args
        assert args[0] == likelihood
        assert args[1] == proposal
        assert args[2] == generator
        assert args[3] is None
        assert args[4] is None
        assert args[5] is False
        # args[6] is kwargs
        # args[7] is rng
        assert args[8] is None  # checkpoint_file
        assert args[9] is False  # show_progress_bar (explicitly set to False)


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
        mock_sampler_instance.phi_sequence = [0, 0.5, 1]

        likelihood = mocker.Mock(side_effect=lambda x: x * 10)

        proposal = "proposal"
        generator = "generator"
        kwargs = {"num_particles": 3, "num_mcmc_samples": 4}

        models, likelihoods, phis = sample(
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

        mock_adaptive_sampler.assert_called_once_with(mock_kernel_instance, show_progress_bar=True)

        mock_sampler_instance.sample.assert_called_once_with(**kwargs)

        assert mock_sampler_instance._mutator._compute_cov is False

        expected_models = dummy_params[:, 0].tolist()
        assert models == expected_models

        expected_likelihoods = [m * 10 for m in expected_models]
        assert likelihoods == expected_likelihoods
        assert likelihood.call_count == len(expected_models)

        expected_phis = [0, 0.5, 1]
        assert phis == expected_phis


class TestSampleLimits:
    def test_both_sample_limits_passed_through(self, mocker):
        """Test that both max_time and max_equation_evals are passed through."""
        mock_run_smc = mocker.patch(
            f"{IMPORTMODULE}.run_smc",
            return_value=("mock_models", "mock_likelihoods", "mock_phis"),
        )

        likelihood = lambda x: x
        proposal = object()
        generator = object()
        max_time = 30.0
        max_equation_evals = 5000

        sample(
            likelihood,
            proposal,
            generator,
            max_time=max_time,
            max_equation_evals=max_equation_evals,
        )

        mock_run_smc.assert_called_once()
        args, _ = mock_run_smc.call_args
        assert args[3] == max_time
        assert args[4] == max_equation_evals

    def test_fixed_time_sampler_when_max_time_specified(self, mocker, sampler_mocks):
        """Test that FixedTimeSampler is used when max_time is specified."""
        max_time = 60.0

        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            generator="generator",
            max_time=max_time,
            max_equation_evals=None,
            multiprocess=False,
            kwargs={"num_particles": 10, "num_mcmc_samples": 5},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=True,
        )

        # Verify FixedTimeSampler was used
        sampler_mocks["fixed_time_sampler"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, max_time, show_progress_bar=True
        )
        sampler_mocks["max_step_sampler"].assert_not_called()
        sampler_mocks["adaptive_sampler"].assert_not_called()

    def test_max_step_sampler_when_max_equation_evals_specified(
        self, mocker, sampler_mocks
    ):
        """Test that MaxStepSampler is used when max_equation_evals is specified (and max_time is None)."""
        max_equation_evals = 10000
        num_particles = 100
        num_mcmc_samples = 10
        expected_max_steps = 10

        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            generator="generator",
            max_time=None,
            max_equation_evals=max_equation_evals,
            multiprocess=False,
            kwargs={
                "num_particles": num_particles,
                "num_mcmc_samples": num_mcmc_samples,
            },
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=True,
        )

        sampler_mocks["max_step_sampler"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, max_steps=expected_max_steps, show_progress_bar=True
        )
        sampler_mocks["fixed_time_sampler"].assert_not_called()
        sampler_mocks["adaptive_sampler"].assert_not_called()

    def test_adaptive_sampler_when_no_limits_specified(self, mocker, sampler_mocks):
        """Test that AdaptiveSampler is used when neither max_time nor max_equation_evals is specified."""
        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            generator="generator",
            max_time=None,
            max_equation_evals=None,
            multiprocess=False,
            kwargs={"num_particles": 10, "num_mcmc_samples": 5},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=True,
        )

        sampler_mocks["adaptive_sampler"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, show_progress_bar=True
        )
        sampler_mocks["fixed_time_sampler"].assert_not_called()
        sampler_mocks["max_step_sampler"].assert_not_called()

    def test_max_time_takes_precedence_over_max_equation_evals(
        self, mocker, sampler_mocks
    ):
        """Test that max_time takes precedence when both max_time and max_equation_evals are specified."""
        max_time = 30.0
        max_equation_evals = 5000

        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            generator="generator",
            max_time=max_time,
            max_equation_evals=max_equation_evals,
            multiprocess=False,
            kwargs={"num_particles": 10, "num_mcmc_samples": 5},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=True,
        )

        sampler_mocks["fixed_time_sampler"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, max_time, show_progress_bar=True
        )
        sampler_mocks["max_step_sampler"].assert_not_called()
        sampler_mocks["adaptive_sampler"].assert_not_called()

    @pytest.mark.parametrize(
        "max_equation_evals,num_particles,num_mcmc_samples,expected_max_steps",
        [
            (10000, 100, 10, 10),
            (5000, 50, 5, 20),
            (1000, 25, 4, 10),
        ],
    )
    def test_max_steps_calculation_correct(
        self,
        mocker,
        sampler_mocks,
        max_equation_evals,
        num_particles,
        num_mcmc_samples,
        expected_max_steps,
    ):
        """Test that max_steps is calculated correctly for MaxStepSampler."""
        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            generator="generator",
            max_time=None,
            max_equation_evals=max_equation_evals,
            multiprocess=False,
            kwargs={
                "num_particles": num_particles,
                "num_mcmc_samples": num_mcmc_samples,
            },
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=True,
        )

        sampler_mocks["max_step_sampler"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, max_steps=expected_max_steps, show_progress_bar=True
        )

    def test_show_progress_bar_false_passed_to_samplers(self, mocker, sampler_mocks):
        """Test that show_progress_bar=False gets passed to the samplers correctly."""
        # Test with AdaptiveSampler (default case)
        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            generator="generator",
            max_time=None,
            max_equation_evals=None,
            multiprocess=False,
            kwargs={"num_particles": 10, "num_mcmc_samples": 5},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=False,
        )

        sampler_mocks["adaptive_sampler"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, show_progress_bar=False
        )
        sampler_mocks["fixed_time_sampler"].assert_not_called()
        sampler_mocks["max_step_sampler"].assert_not_called()

        # Reset mocks for next test
        sampler_mocks["adaptive_sampler"].reset_mock()
        sampler_mocks["fixed_time_sampler"].reset_mock()
        sampler_mocks["max_step_sampler"].reset_mock()

        # Test with FixedTimeSampler  
        max_time = 30.0
        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            generator="generator",
            max_time=max_time,
            max_equation_evals=None,
            multiprocess=False,
            kwargs={"num_particles": 10, "num_mcmc_samples": 5},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=False,
        )

        sampler_mocks["fixed_time_sampler"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, max_time, show_progress_bar=False
        )
        sampler_mocks["adaptive_sampler"].assert_not_called()
        sampler_mocks["max_step_sampler"].assert_not_called()

        # Reset mocks for next test
        sampler_mocks["adaptive_sampler"].reset_mock()
        sampler_mocks["fixed_time_sampler"].reset_mock()
        sampler_mocks["max_step_sampler"].reset_mock()

        # Test with MaxStepSampler
        max_equation_evals = 1000
        num_particles = 25
        num_mcmc_samples = 4
        expected_max_steps = 10
        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            generator="generator",
            max_time=None,
            max_equation_evals=max_equation_evals,
            multiprocess=False,
            kwargs={
                "num_particles": num_particles,
                "num_mcmc_samples": num_mcmc_samples,
            },
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=False,
        )

        sampler_mocks["max_step_sampler"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, max_steps=expected_max_steps, show_progress_bar=False
        )
        sampler_mocks["adaptive_sampler"].assert_not_called()
        sampler_mocks["fixed_time_sampler"].assert_not_called()
