import pytest
import numpy as np

from pysips.random_choice_proposal import RandomChoiceProposal


class TestRandomChoiceProposal:

    @pytest.fixture
    def mock_model(self, mocker):
        """Create a mock model for testing"""
        return mocker.MagicMock()

    @pytest.fixture
    def mock_proposals(self, mocker):
        """Create mock proposals for testing"""
        prop1 = mocker.MagicMock(name="proposal1")
        prop1.return_value = mocker.MagicMock(name="result1")

        prop2 = mocker.MagicMock(name="proposal2")
        prop2.return_value = mocker.MagicMock(name="result2")

        prop3 = mocker.MagicMock(name="proposal3")
        prop3.return_value = mocker.MagicMock(name="result3")

        return [prop1, prop2, prop3]

    @pytest.fixture
    def probabilities(self):
        """Create probabilities for testing"""
        return [0.2, 0.5, 0.3]

    def test_initialization(self, mock_proposals, probabilities, mocker):
        """Test proper initialization of RandomChoiceProposal"""
        # Mock numpy random
        mock_random = mocker.patch("numpy.random.default_rng")

        seed = 42
        _ = RandomChoiceProposal(
            mock_proposals, probabilities, exclusive=True, seed=seed
        )
        mock_random.assert_called_once_with(seed)

    def test_update_method(self, mock_proposals, probabilities, mocker):
        """Test update method calls update on all proposals"""
        # Initialize proposal
        proposal = RandomChoiceProposal(mock_proposals, probabilities)

        # Call update with args and kwargs
        test_args = ["arg1", "arg2"]
        test_kwargs = {"key1": "value1", "key2": "value2"}
        proposal.update(*test_args, **test_kwargs)

        # Assertions - each proposal should have update called with same args/kwargs
        for mock_prop in mock_proposals:
            mock_prop.update.assert_called_once_with(*test_args, **test_kwargs)

    def test_call_exclusive_mode(
        self, mock_proposals, probabilities, mock_model, mocker
    ):
        """Test __call__ method in exclusive mode selects one proposal and applies it"""
        # Mock RNG to control which proposal is selected
        mock_rng = mocker.MagicMock()
        mock_rng.random.return_value = 0.5  # Will select proposals[1] (0.2 < 0.5 < 0.7)

        # Initialize proposal with mocked RNG
        proposal = RandomChoiceProposal(mock_proposals, probabilities, exclusive=True)
        proposal._rng = mock_rng

        # Call the proposal
        result = proposal(mock_model)

        # Assertions
        mock_proposals[1].assert_called_once_with(mock_model)

        # Only the selected proposal should be called
        mock_proposals[0].assert_not_called()
        mock_proposals[2].assert_not_called()

        # Result should be the output of the selected proposal
        assert result == mock_proposals[1].return_value

    def test_call_non_exclusive_mode(
        self, mock_proposals, probabilities, mock_model, mocker
    ):
        """Test __call__ method in non-exclusive mode can select multiple proposals"""
        # Mock RNG to make proposals 0 and 2 be selected
        mock_rng = mocker.MagicMock()
        # For each proposal: [0:True, 1:False, 2:True]
        mock_rng.random.side_effect = [0.1, 0.9, 0.1]  # < prob = select, > prob = skip
        # Prevent random shuffling for predictable testing
        mock_rng.shuffle.side_effect = lambda x: None

        # Initialize proposal with mocked RNG
        proposal = RandomChoiceProposal(mock_proposals, probabilities, exclusive=False)
        proposal._rng = mock_rng

        # Call the proposal
        result = proposal(mock_model)

        # Assertions - first proposal should be called with original model
        mock_proposals[0].assert_called_once_with(mock_model)

        # Second proposal should not be called
        mock_proposals[1].assert_not_called()

        # Third proposal should be called with result from first proposal
        mock_proposals[2].assert_called_once_with(mock_proposals[0].return_value)

        # Result should be the output of the last applied proposal
        assert result == mock_proposals[2].return_value

    def test_call_non_exclusive_empty_first_round(
        self, mock_proposals, probabilities, mock_model, mocker
    ):
        """Test __call__ in non-exclusive mode retries if no proposals are selected initially"""
        # Mock RNG to reject all on first pass, then select one on second pass
        mock_rng = mocker.MagicMock()
        # First round: all above threshold (none selected)
        # Second round: select proposal 0
        mock_rng.random.side_effect = [0.9, 0.9, 0.9, 0.1, 0.9, 0.9]

        # Initialize proposal
        proposal = RandomChoiceProposal(mock_proposals, probabilities, exclusive=False)
        proposal._rng = mock_rng

        # Call the proposal
        result = proposal(mock_model)

        # Assertions
        mock_proposals[0].assert_called_once_with(mock_model)
        mock_proposals[1].assert_not_called()
        mock_proposals[2].assert_not_called()

        # Result should be the output of the selected proposal
        assert result == mock_proposals[0].return_value
