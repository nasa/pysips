import pytest

from pysips.mutation_proposal import MutationProposal


IMPORTMODULE = MutationProposal.__module__


class TestMutationProposal:

    @pytest.fixture
    def basic_setup(self):
        """Basic setup for mutation proposal tests"""
        X_dim = 2
        operators = ["+", "-", "*"]
        return X_dim, operators

    @pytest.fixture
    def mock_model(self, mocker):
        """Create a mock model for testing"""
        return mocker.MagicMock()

    def test_initialization_passes_info_to_bingo(self, mocker):
        """Test proper initialization of MutationProposal"""
        X_dim = 3
        operators = ["+", "-", "*", "sin"]

        mock_component_gen = mocker.MagicMock()
        mock_component_gen_cls = mocker.patch(
            f"{IMPORTMODULE}.ComponentGenerator", return_value=mock_component_gen
        )
        mock_agraph_mutation = mocker.MagicMock()
        mock_agraph_mutation_cls = mocker.patch(
            f"{IMPORTMODULE}.AGraphMutation", return_value=mock_agraph_mutation
        )

        mutation_proposal = MutationProposal(
            x_dim=X_dim,
            operators=operators,
            terminal_probability=0.2,
            constant_probability=0.5,
            command_probability=0.1,
            node_probability=0.3,
            parameter_probability=0.15,
            prune_probability=0.25,
            fork_probability=0.2,
            repeat_mutation_probability=0.1,
            seed=42,
        )

        mock_component_gen_cls.assert_called_once_with(
            input_x_dimension=X_dim, terminal_probability=0.2, constant_probability=0.5
        )

        assert mock_component_gen.add_operator.call_count == len(operators)
        for op in operators:
            mock_component_gen.add_operator.assert_any_call(op)

        mock_agraph_mutation_cls.assert_called_once_with(
            mock_component_gen,
            0.1,  # command_probability
            0.3,  # node_probability
            0.15,  # parameter_probability
            0.25,  # prune_probability
            0.2,  # fork_probability
        )

    def test_do_mutation_single(self, basic_setup, mock_model, mocker):
        """Test _do_mutation method without repeat"""
        X_dim, operators = basic_setup

        # Setup mocks
        mock_mutation = mocker.MagicMock()
        mutated_model = mocker.MagicMock()
        mock_mutation.return_value = mutated_model

        # Patch random generator to ensure no repeat
        mock_rng = mocker.MagicMock()
        mock_rng.random.return_value = 1.0  # > repeat_mutation_prob

        mocker.patch(f"{IMPORTMODULE}.ComponentGenerator")
        mocker.patch(f"{IMPORTMODULE}.AGraphMutation", return_value=mock_mutation)

        mutation_proposal = MutationProposal(x_dim=X_dim, operators=operators)
        mutation_proposal._rng = mock_rng

        result = mutation_proposal._do_mutation(mock_model)
        mock_mutation.assert_called_once_with(mock_model)
        assert result == mutated_model
        assert mock_rng.random.call_count == 1

    def test_do_mutation_with_repeat(self, basic_setup, mock_model, mocker):
        """Test _do_mutation method with repeated mutations"""
        X_dim, operators = basic_setup

        mock_mutation = mocker.MagicMock()
        mutated_model1 = mocker.MagicMock(name="mutated_model1")
        mutated_model2 = mocker.MagicMock(name="mutated_model2")
        mutated_model3 = mocker.MagicMock(name="mutated_model3")
        mock_mutation.side_effect = [mutated_model1, mutated_model2, mutated_model3]

        # Patch random generator to ensure repeat twice
        mock_rng = mocker.MagicMock()
        mock_rng.random.side_effect = [
            0.1,
            0.2,
            0.9,
        ]  # First two < repeat_mutation_prob

        mocker.patch(f"{IMPORTMODULE}.ComponentGenerator")
        mocker.patch(f"{IMPORTMODULE}.AGraphMutation", return_value=mock_mutation)

        mutation_proposal = MutationProposal(
            x_dim=X_dim, operators=operators, repeat_mutation_probability=0.5
        )
        mutation_proposal._rng = mock_rng

        result = mutation_proposal(mock_model)

        assert mock_mutation.call_count == 3
        mock_mutation.assert_any_call(mock_model)
        mock_mutation.assert_any_call(mutated_model1)
        mock_mutation.assert_any_call(mutated_model2)
        assert result == mutated_model3
        assert mock_rng.random.call_count == 3

    def test_call_different_model(self, basic_setup, mock_model, mocker):
        """Test __call__ when mutation immediately produces a different model"""
        X_dim, operators = basic_setup

        mocker.patch(f"{IMPORTMODULE}.ComponentGenerator")

        mutated_model = mocker.MagicMock()
        mock_mutation = mocker.MagicMock(return_value=mutated_model)
        mocker.patch(f"{IMPORTMODULE}.AGraphMutation", return_value=mock_mutation)

        # Ensure models are different
        mock_model.__eq__.return_value = False

        mutation_proposal = MutationProposal(
            x_dim=X_dim, operators=operators, repeat_mutation_probability=0.0
        )
        result = mutation_proposal(mock_model)

        # Assertions
        mock_mutation.assert_called_once_with(mock_model)
        assert result == mutated_model

    def test_call_retry_mutation(self, basic_setup, mock_model, mocker):
        """Test __call__ when mutation initially produces an equivalent model"""
        X_dim, operators = basic_setup

        mocker.patch(f"{IMPORTMODULE}.ComponentGenerator")

        equivalent_model = mock_model
        different_model = mocker.MagicMock()
        mock_mutation = mocker.MagicMock(
            side_effect=[equivalent_model, different_model]
        )
        mocker.patch(f"{IMPORTMODULE}.AGraphMutation", return_value=mock_mutation)

        mock_model.__eq__.side_effect = lambda other: other is equivalent_model

        mutation_proposal = MutationProposal(
            x_dim=X_dim, operators=operators, repeat_mutation_probability=0.0
        )
        result = mutation_proposal(mock_model)

        assert mock_mutation.call_count == 2
        assert result == different_model

    def test_update_method(self, basic_setup, mocker):
        """Test update method (should be a no-op)"""
        X_dim, operators = basic_setup

        mocker.patch(f"{IMPORTMODULE}.ComponentGenerator")
        mocker.patch(f"{IMPORTMODULE}.AGraphMutation")

        mutation_proposal = MutationProposal(x_dim=X_dim, operators=operators)

        # This should not raise any errors
        mutation_proposal.update(some_param=10)
        mutation_proposal.update()

    @pytest.mark.parametrize("seed", [None, 0, 42, 12345])
    def test_random_seed(self, mocker, basic_setup, seed):
        """Test that random seed is properly used"""
        X_dim, operators = basic_setup

        # Create class with different seeds
        mocker.patch(f"{IMPORTMODULE}.ComponentGenerator"),
        mocker.patch(f"{IMPORTMODULE}.AGraphMutation"),
        mock_rng = mocker.patch("numpy.random.default_rng")

        _ = MutationProposal(x_dim=X_dim, operators=operators, seed=seed)
        mock_rng.assert_called_once_with(seed)
