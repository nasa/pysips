import pytest

from pysips.crossover_proposal import CrossoverProposal

IMPORTMODULE = CrossoverProposal.__module__


class TestCrossoverProposal:

    @pytest.fixture
    def mock_model(self, mocker):
        """Create a mock model for testing"""
        mock_model = mocker.MagicMock()
        mock_model.__eq__.side_effect = lambda x: x is mock_model
        return mock_model

    @pytest.fixture
    def mock_gene_pool(self, mocker, mock_model):
        """Create a mock gene pool with several models"""
        model1 = mocker.MagicMock()
        model2 = mocker.MagicMock()

        # Configure models to be different from each other
        model1.__eq__.side_effect = lambda x: x is model1
        model2.__eq__.side_effect = lambda x: x is model2

        return [mock_model, model1, model2]

    def test_initialization(self, mock_gene_pool, mocker):
        """Test proper initialization of CrossoverProposal"""
        # Mock dependencies
        mock_agraph_crossover = mocker.MagicMock()
        mocker.patch(
            f"{IMPORTMODULE}.AGraphCrossover", return_value=mock_agraph_crossover
        )
        mock_random = mocker.patch("numpy.random.default_rng")

        # Initialize CrossoverProposal
        seed = 42
        crossover_proposal = CrossoverProposal(mock_gene_pool, seed=seed)

        # Assertions
        assert crossover_proposal._gene_pool == mock_gene_pool
        mock_random.assert_called_once_with(seed)

    def test_call_method_selects_different_parent(
        self, mock_gene_pool, mock_model, mocker
    ):
        """Test that __call__ selects a parent different from the input model"""
        # Mock AGraphCrossover
        mock_crossover = mocker.MagicMock()
        child_1 = mocker.MagicMock(name="child_1")
        child_2 = mocker.MagicMock(name="child_2")
        mock_crossover.return_value = (child_1, child_2)
        mocker.patch(f"{IMPORTMODULE}.AGraphCrossover", return_value=mock_crossover)

        # Mock random number generator
        mock_rng = mocker.MagicMock()
        mock_rng.integers.side_effect = [0, 1]
        mock_rng.random.return_value = 0.4  # Will select first child

        # Initialize CrossoverProposal with mocked RNG
        crossover_proposal = CrossoverProposal(mock_gene_pool)
        crossover_proposal._rng = mock_rng

        # Call the method
        result = crossover_proposal(mock_model)

        # Assertions
        # Check that crossover was called with model and a different parent
        mock_crossover.assert_called_once()
        args = mock_crossover.call_args[0]
        assert args[0] == mock_model  # First arg is the input model
        assert args[1] == mock_gene_pool[1]  # Second arg should be a different model
        assert args[1] != mock_model  # Ensure it's not the same model

        # Check we got the expected child
        assert result == child_1

    @pytest.mark.parametrize(
        "random_value, expected_child_index",
        [
            (0.3, 0),
            (0.8, 1),
        ],
    )
    def test_call_method_selects_child_based_on_random(
        self, mock_gene_pool, mock_model, mocker, random_value, expected_child_index
    ):
        """Test that __call__ selects the appropriate child based on random value"""
        mock_crossover = mocker.MagicMock()
        child_1 = mocker.MagicMock(name="child_1")
        child_2 = mocker.MagicMock(name="child_2")
        children = (child_1, child_2)
        mock_crossover.return_value = children
        mocker.patch(f"{IMPORTMODULE}.AGraphCrossover", return_value=mock_crossover)

        # Mock random number generator
        mock_rng = mocker.MagicMock()
        mock_rng.integers.return_value = 1
        mock_rng.random.return_value = random_value

        # Initialize and call
        crossover_proposal = CrossoverProposal(mock_gene_pool)
        crossover_proposal._rng = mock_rng
        result = crossover_proposal(mock_model)

        # Assertions
        expected_child = children[expected_child_index]
        assert result == expected_child

    def test_update_method(self, mock_gene_pool, mocker):
        """Test update method updates the gene pool"""
        # Mock AGraphCrossover
        mocker.patch(f"{IMPORTMODULE}.AGraphCrossover")

        # Initialize
        crossover_proposal = CrossoverProposal(mock_gene_pool)

        # Create new gene pool
        new_model1 = mocker.MagicMock(name="new_model1")
        new_model2 = mocker.MagicMock(name="new_model2")
        new_gene_pool = (new_model1, new_model2)

        # Update gene pool
        crossover_proposal.update(new_gene_pool)

        # Assertions
        assert crossover_proposal._gene_pool == list(new_gene_pool)

    # @pytest.mark.parametrize("seed", [None, 0, 42, 12345])
    # def test_random_seed(self, mock_gene_pool, seed, mocker):
    #     """Test that random seed is properly used"""
    #     # Mock AGraphCrossover
    #     mocker.patch(f"{IMPORTMODULE}.AGraphCrossover")

    #     # Mock numpy random generator
    #     mock_rng = mocker.patch("numpy.random.default_rng")

    #     # Initialize with different seeds
    #     CrossoverProposal(mock_gene_pool, seed=seed)

    #     # Check the RNG was initialized with the right seed
    #     mock_rng.assert_called_once_with(seed)

    def test_empty_gene_pool_error(self, mocker):
        """Test that an empty gene pool results in appropriate failure"""
        # Mock AGraphCrossover
        mocker.patch(f"{IMPORTMODULE}.AGraphCrossover")

        # Initialize with empty pool
        crossover_proposal = CrossoverProposal([])

        # Call should raise exception due to empty pool
        with pytest.raises(ValueError) as excinfo:
            crossover_proposal(mocker.MagicMock())
