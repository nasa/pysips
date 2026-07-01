import numpy as np
import pytest

from pysips.sampler import sample, run_smc

IMPORTMODULE = sample.__module__


def _make_mock_step(mocker, fitness=-10.0, n=1):
    """Return a mock step whose log_likes is an object array of expressions."""
    log_likes = np.empty((n, 1), dtype=object)
    for i in range(n):
        expr = mocker.Mock()
        expr.fitness = fitness
        log_likes[i, 0] = expr
    return mocker.Mock(log_likes=log_likes)


@pytest.fixture
def sampler_mocks(mocker):
    """Common mocks for sampler strategy-selection tests."""
    mocker.patch(f"{IMPORTMODULE}.ObjectVectorMCMC")
    mocker.patch(f"{IMPORTMODULE}.EquationGeometricPath")
    mock_kernel = mocker.patch(f"{IMPORTMODULE}.VectorMCMCKernel")

    mock_fixed = mocker.patch(f"{IMPORTMODULE}.FixedTimeSampler")
    mock_max = mocker.patch(f"{IMPORTMODULE}.MaxStepSampler")
    mock_adaptive = mocker.patch(f"{IMPORTMODULE}.AdaptiveSampler")

    for mock_cls in [mock_fixed, mock_max, mock_adaptive]:
        inst = mocker.Mock()
        inst.step = _make_mock_step(mocker)
        inst.phi_sequence = [1.0]
        mock_cls.return_value = inst

    return {
        "kernel": mock_kernel,
        "fixed_time": mock_fixed,
        "max_step": mock_max,
        "adaptive": mock_adaptive,
    }


class TestSampleDefaults:
    def test_delegates_to_run_smc_with_defaults(self, mocker):
        mock_run_smc = mocker.patch(
            f"{IMPORTMODULE}.run_smc",
            return_value=(["model"], [-10.0], [1.0]),
        )
        likelihood = mocker.Mock()
        prior = [mocker.Mock()]

        result = sample(likelihood, prior, seed=42)

        assert result == (["model"], [-10.0], [1.0])
        mock_run_smc.assert_called_once()
        args = mock_run_smc.call_args[0]
        assert args[0] is likelihood
        assert args[1] is prior
        assert args[2] is None  # max_time
        assert args[3] is None  # max_equation_evals
        assert args[4] == {"num_particles": 5000, "num_mcmc_samples": 10}
        assert isinstance(args[5], np.random.Generator)
        assert args[6] is None  # checkpoint_file
        assert args[7] is True  # show_progress_bar

    def test_custom_kwargs_override_defaults(self, mocker):
        mock_run_smc = mocker.patch(
            f"{IMPORTMODULE}.run_smc",
            return_value=(["model"], [-10.0], [1.0]),
        )
        custom = {"num_particles": 100, "num_mcmc_samples": 3}
        sample(mocker.Mock(), [mocker.Mock()], kwargs=custom, seed=0)

        args = mock_run_smc.call_args[0]
        assert args[4] == custom

    def test_show_progress_bar_false_passed_through(self, mocker):
        mock_run_smc = mocker.patch(
            f"{IMPORTMODULE}.run_smc",
            return_value=([], [], []),
        )
        sample(mocker.Mock(), [mocker.Mock()], show_progress_bar=False, seed=0)

        args = mock_run_smc.call_args[0]
        assert args[7] is False

    def test_max_time_and_max_equation_evals_passed_through(self, mocker):
        mock_run_smc = mocker.patch(
            f"{IMPORTMODULE}.run_smc",
            return_value=([], [], []),
        )
        sample(mocker.Mock(), [mocker.Mock()], max_time=30.0, max_equation_evals=5000)

        args = mock_run_smc.call_args[0]
        assert args[2] == 30.0
        assert args[3] == 5000


class TestRunSMCSamplerChoice:
    def test_adaptive_sampler_used_by_default(self, mocker, sampler_mocks):
        run_smc(
            mocker.Mock(),
            [mocker.Mock()],
            max_time=None,
            max_equation_evals=None,
            kwargs={"num_particles": 10, "num_mcmc_samples": 5},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=True,
        )
        sampler_mocks["adaptive"].assert_called_once()
        sampler_mocks["fixed_time"].assert_not_called()
        sampler_mocks["max_step"].assert_not_called()

    def test_fixed_time_sampler_when_max_time_given(self, mocker, sampler_mocks):
        run_smc(
            mocker.Mock(),
            [mocker.Mock()],
            max_time=60.0,
            max_equation_evals=None,
            kwargs={"num_particles": 10, "num_mcmc_samples": 5},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=True,
        )
        sampler_mocks["fixed_time"].assert_called_once_with(
            sampler_mocks["kernel"].return_value, 60.0, show_progress_bar=True
        )
        sampler_mocks["max_step"].assert_not_called()
        sampler_mocks["adaptive"].assert_not_called()

    def test_max_step_sampler_when_max_equation_evals_given(
        self, mocker, sampler_mocks
    ):
        # max_steps = 10000 // (100 * 10) = 10
        run_smc(
            mocker.Mock(),
            [mocker.Mock()],
            max_time=None,
            max_equation_evals=10000,
            kwargs={"num_particles": 100, "num_mcmc_samples": 10},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=True,
        )
        sampler_mocks["max_step"].assert_called_once_with(
            sampler_mocks["kernel"].return_value,
            max_steps=10,
            show_progress_bar=True,
        )
        sampler_mocks["fixed_time"].assert_not_called()
        sampler_mocks["adaptive"].assert_not_called()


class TestRunSMCExtractsModels:
    def test_models_extracted_from_log_like_slot(self, mocker, sampler_mocks):
        expr1, expr2 = mocker.Mock(), mocker.Mock()
        expr1.fitness = -5.0
        expr2.fitness = -8.0
        log_likes = np.empty((2, 1), dtype=object)
        log_likes[0, 0] = expr1
        log_likes[1, 0] = expr2
        sampler_mocks["adaptive"].return_value.step = mocker.Mock(log_likes=log_likes)
        sampler_mocks["adaptive"].return_value.phi_sequence = [0.5, 1.0]

        models, log_like_vals, phis = run_smc(
            mocker.Mock(),
            [mocker.Mock()],
            max_time=None,
            max_equation_evals=None,
            kwargs={"num_particles": 2, "num_mcmc_samples": 1},
            rng=mocker.Mock(),
            checkpoint_file=None,
            show_progress_bar=False,
        )

        assert models == [expr1, expr2]
        assert log_like_vals == [-5.0, -8.0]
        assert phis == [0.5, 1.0]


@pytest.fixture
def sampler_mocks(mocker):
    """Fixture that sets up common mocks for the samplers tests."""
    # Mock all the dependencies - patch at source to avoid lazy import issues
    mocker.patch("pysips.priors.improper_uniform_prior.ImproperUniformPrior")
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
        mock_step = mocker.Mock(
            params=np.array([[1]]),
            log_likes=np.array([[0.5]]),
        )
        mock_instance.sample.return_value = (mocker.Mock(), None)
        mock_instance.step = mock_step
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
        prior = object()
        seed = 42

        result = sample(likelihood, proposal, prior, seed=seed)

        assert result == ("mock_models", "mock_likelihoods")

        mock_run_smc.assert_called_once()
        args, _ = mock_run_smc.call_args

        assert args[0] == likelihood
        assert args[1] == proposal
        assert args[2] == prior
        assert args[3] is None  # max_time
        assert args[4] is None  # max_equation_evals
        assert args[5] is False  # multiprocess

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
        custom_kwargs = {"num_particles": 100, "num_mcmc_samples": 3}
        prior = object()

        result = sample(likelihood, proposal, prior, kwargs=custom_kwargs, seed=24)

        assert result == ("mock_models", "mock_likelihoods")
        mock_run_smc.assert_called_once()

        args, _ = mock_run_smc.call_args
        assert args[0] == likelihood
        assert args[1] == proposal
        assert args[2] == prior
        assert args[3] is None  # max_time
        assert args[4] is None  # max_equation_evals
        assert args[5] is False  # multiprocess
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
        prior = object()

        result = sample(likelihood, proposal, prior, show_progress_bar=False, seed=42)

        assert result == ("mock_models", "mock_likelihoods")
        mock_run_smc.assert_called_once()

        args, _ = mock_run_smc.call_args
        assert args[0] == likelihood
        assert args[1] == proposal
        assert args[2] == prior
        assert args[3] is None  # max_time
        assert args[4] is None  # max_equation_evals
        assert args[5] is False  # multiprocess
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
        dummy_log_likes = np.array([[10], [20], [30]])
        dummy_step = mocker.Mock(params=dummy_params, log_likes=dummy_log_likes)
        mock_sampler_instance.sample.return_value = (mocker.Mock(), None)
        mock_sampler_instance.step = dummy_step
        mock_sampler_instance.phi_sequence = [0, 0.5, 1]

        likelihood = mocker.Mock(side_effect=lambda x: x * 10)

        proposal = "proposal"
        prior = "prior"
        kwargs = {"num_particles": 3, "num_mcmc_samples": 4}

        models, likelihoods, phis = sample(
            likelihood,
            proposal,
            prior,
            multiprocess=multiproc,
            kwargs=kwargs,
            seed=0,
        )

        mock_metropolis.assert_called_once_with(
            likelihood=likelihood,
            proposal=proposal,
            prior=prior,
            multiprocess=multiproc,
        )

        mock_vector_kernel.assert_called_once_with(
            mock_mcmc_instance, param_order=["f"], rng=mock_rng_instance
        )

        mock_adaptive_sampler.assert_called_once_with(
            mock_kernel_instance, show_progress_bar=True
        )

        mock_sampler_instance.sample.assert_called_once_with(**kwargs)

        assert mock_sampler_instance._mutator._compute_cov is False

        expected_models = dummy_params[:, 0].tolist()
        assert models == expected_models

        expected_likelihoods = dummy_log_likes.ravel().tolist()
        assert likelihoods == expected_likelihoods

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
        prior = object()
        max_time = 30.0
        max_equation_evals = 5000

        sample(
            likelihood,
            proposal,
            prior,
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
            prior=None,
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
            prior=None,
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
            sampler_mocks["kernel"].return_value,
            max_steps=expected_max_steps,
            show_progress_bar=True,
        )
        sampler_mocks["fixed_time_sampler"].assert_not_called()
        sampler_mocks["adaptive_sampler"].assert_not_called()

    def test_adaptive_sampler_when_no_limits_specified(self, mocker, sampler_mocks):
        """Test that AdaptiveSampler is used when neither max_time nor max_equation_evals is specified."""
        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            prior=None,
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
            prior=None,
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
            prior=None,
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
            sampler_mocks["kernel"].return_value,
            max_steps=expected_max_steps,
            show_progress_bar=True,
        )

    @pytest.mark.parametrize(
        "max_time,max_equation_evals,expected_sampler",
        [
            # AdaptiveSampler case
            (None, None, "adaptive_sampler"),
            # FixedTimeSampler case
            (30.0, None, "fixed_time_sampler"),
            # MaxStepSampler case
            (None, 1000, "max_step_sampler"),
        ],
    )
    def test_show_progress_bar_false_passed_to_samplers(
        self, mocker, sampler_mocks, max_time, max_equation_evals, expected_sampler
    ):
        """Test that show_progress_bar=False gets passed to the samplers correctly."""
        # For MaxStepSampler case, we need specific particle/mcmc values to get expected_max_steps=10
        if max_equation_evals is not None:
            num_particles = 25
            num_mcmc_samples = 4
            expected_max_steps = 10
        else:
            num_particles = 10
            num_mcmc_samples = 5

        run_smc(
            likelihood=sampler_mocks["likelihood"],
            proposal="proposal",
            prior=None,
            max_time=max_time,
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

        # Check the expected sampler was called correctly
        if expected_sampler == "adaptive_sampler":
            sampler_mocks[expected_sampler].assert_called_once_with(
                sampler_mocks["kernel"].return_value, show_progress_bar=False
            )
        elif expected_sampler == "fixed_time_sampler":
            sampler_mocks[expected_sampler].assert_called_once_with(
                sampler_mocks["kernel"].return_value, max_time, show_progress_bar=False
            )
        elif expected_sampler == "max_step_sampler":
            sampler_mocks[expected_sampler].assert_called_once_with(
                sampler_mocks["kernel"].return_value,
                max_steps=expected_max_steps,
                show_progress_bar=False,
            )

        # Verify other samplers were not called
        for sampler_name in [
            "adaptive_sampler",
            "fixed_time_sampler",
            "max_step_sampler",
        ]:
            if sampler_name != expected_sampler:
                sampler_mocks[sampler_name].assert_not_called()
