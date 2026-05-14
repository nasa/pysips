import pytest
import numpy as np

# Import the class to test
from pysips.priors.improper_uniform_prior import (
    ImproperUniformPrior,
    MAX_REPEATS,
    _MAX_STAGNATION,
)


Prior = ImproperUniformPrior
IMPORTMODULE = Prior.__module__


class TestPrior:

    @pytest.fixture
    def mock_generator(self, mocker):
        """Create a mock generator function"""
        generator = mocker.MagicMock()

        # Generate unique values when called
        models = ["model1", "model2", "model3", "model4", "model5"]
        generator.side_effect = models

        return generator, models

    def test_rvs_returns_correct_number_of_models(self, mock_generator, mocker):
        """Test rvs returns the requested number of models"""
        generator, models = mock_generator

        prior = Prior(generator)

        # Request 3 models
        result = prior.rvs(3)

        # Check generator called enough times
        assert generator.call_count == 3

        # Check correct shape and content
        assert result.shape == (3, 1)
        print(result)
        assert set(result.flatten()) == set(models[:3])

    def test_rvs_handles_duplicates(self, mocker):
        """Test rvs correctly handles duplicate models from generator"""
        # Create generator that sometimes returns duplicates
        generator = mocker.MagicMock()
        generator.side_effect = ["model1", "model2", "model2", "model3", "model4"]

        prior = Prior(generator)

        # Request 4 unique models (will need 5 calls due to duplicate)
        result = prior.rvs(4)

        # Check generator called enough times to get 4 unique models
        assert generator.call_count == 5

        # Check correct shape and content (should be 4 unique models)
        assert result.shape == (4, 1)
        assert set(result.flatten()) == {"model1", "model2", "model3", "model4"}

    @pytest.mark.parametrize("N", [0, 1, 5, 10])
    def test_rvs_with_various_sizes(self, mocker, N):
        """Test rvs with different values of N"""
        # Create generator that returns unique models
        counter = [0]

        def gen_unique_models():
            counter[0] += 1
            return f"model_{counter[0]}"

        generator = mocker.MagicMock(side_effect=gen_unique_models)

        prior = Prior(generator)

        # Request N models
        result = prior.rvs(N)

        # Check generator called correct number of times
        assert generator.call_count == N

        # Check result has correct shape
        assert result.shape == (N, 1)

        # Check all models are unique
        if N > 0:
            unique_models = set(result.flatten())
            assert len(unique_models) == N

    def test_warning_issued_for_excessive_repeats(self, mocker):
        """Test that a warning is issued when MAX_REPEATS consecutive duplicates occur"""
        # Create generator that returns the same model repeatedly
        generator = mocker.MagicMock()

        # First returns "model1", then MAX_REPEATS duplicates, then "model2"
        generator.side_effect = ["model1"] + ["model1"] * MAX_REPEATS + ["model2"]

        prior = Prior(generator)

        # Request 2 models, should get a warning due to MAX_REPEATS consecutive duplicates
        with pytest.warns(
            UserWarning, match=f"Generator called {MAX_REPEATS} times in a row"
        ):
            result = prior.rvs(2)

        # Should eventually generate 2 models
        assert result.shape == (2, 1)
        assert (
            generator.call_count == MAX_REPEATS + 2
        )  # 1 unique + MAX_REPEATS duplicates + 1 more unique

        # Verify we got the expected unique models
        assert set(result.flatten()) == {"model1", "model2"}

    def test_warning_issued_only_once(self, mocker):
        """Test that the warning is issued only once per rvs call"""
        # Create generator with many duplicates
        generator = mocker.MagicMock()

        # First "model1", then 2*MAX_REPEATS duplicates of "model1", then "model2"
        generator.side_effect = ["model1"] + ["model1"] * (2 * MAX_REPEATS) + ["model2"]
        prior = Prior(generator)

        # Should see exactly 1 warning despite having 2*MAX_REPEATS duplicates
        with pytest.warns(UserWarning) as record:
            result = prior.rvs(2)

        # Verify we got exactly 1 warning
        assert len(record) == 1

        # Verify the warning has the expected message
        assert f"Generator called {MAX_REPEATS} times in a row" in str(
            record[0].message
        )

        # Verify we got the 2 unique models
        assert result.shape == (2, 1)
        assert set(result.flatten()) == {"model1", "model2"}

    def test_exhausted_generator_returns_duplicates(self, mocker):
        """Requesting more models than the generator can produce falls back to
        duplicates after _MAX_STAGNATION consecutive failures."""
        # Generator can only ever produce 1 unique model.
        generator = mocker.MagicMock(return_value="only_model")
        prior = Prior(generator)

        # Request 3 models from a generator that only produces 1 unique value.
        with pytest.warns(UserWarning):
            result = prior.rvs(3)

        # Should return the requested number of slots, filled with duplicates.
        assert result.shape == (3, 1)
        assert set(result.flatten()) == {"only_model"}
        # Generator should have been called at most _MAX_STAGNATION + 1 times
        # (1 unique model + _MAX_STAGNATION consecutive failures).
        assert generator.call_count <= _MAX_STAGNATION + 1

    def test_exhausted_generator_warning_issued_before_fallback(self, mocker):
        """The UserWarning is issued at MAX_REPEATS, not only at _MAX_STAGNATION."""
        generator = mocker.MagicMock(return_value="only_model")
        prior = Prior(generator)

        with pytest.warns(
            UserWarning, match=f"Generator called {MAX_REPEATS} times in a row"
        ) as record:
            prior.rvs(2)

        # Exactly one warning (issued at MAX_REPEATS, not again at _MAX_STAGNATION).
        assert len(record) == 1

    def test_exhausted_generator_does_not_hang(self, mocker):
        """Regression test: rvs() must terminate even when the generator
        cannot produce enough unique models (was an infinite loop before fix)."""
        generator = mocker.MagicMock(side_effect=lambda: "model")
        prior = Prior(generator)

        import threading
        result_holder = []
        def run():
            with pytest.warns(UserWarning):
                result_holder.append(prior.rvs(10))

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5)  # Must finish well within 5 seconds.
        assert not t.is_alive(), "rvs() did not terminate — infinite loop detected"
        assert result_holder[0].shape == (10, 1)
