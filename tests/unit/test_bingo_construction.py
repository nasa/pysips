import pytest

from pysips.bingo_construction import (
    BingoConstructionConfig,
    build_agraph_generator,
    build_agraph_proposal,
)
from pysips.random_choice_proposal import RandomChoiceProposal


class TestBingoConstruction:
    def test_build_agraph_generator_uses_default_constant_probability(self, mocker):
        mock_component_generator = mocker.patch(
            "pysips.bingo_construction.ComponentGenerator", autospec=True
        )
        mock_agraph_generator = mocker.patch(
            "pysips.bingo_construction.AGraphGenerator", autospec=True
        )

        config = BingoConstructionConfig()
        build_agraph_generator(2, ["+", "*"], config)

        mock_component_generator.assert_called_once_with(
            input_x_dimension=2,
            terminal_probability=0.1,
            constant_probability=1 / 3,
        )
        component_instance = mock_component_generator.return_value
        assert component_instance.add_operator.call_args_list[0].args == ("+",)
        assert component_instance.add_operator.call_args_list[1].args == ("*",)
        mock_agraph_generator.assert_called_once_with(24, 24, component_instance)

    def test_returns_random_choice_proposal(self):
        config = BingoConstructionConfig()
        generator = build_agraph_generator(x_dim=1, operators=["+", "*"], bingo_config=config)
        proposal = build_agraph_proposal(
            x_dim=1,
            generator=generator,
            operators=["+", "*"],
            bingo_config=config,
        )
        assert isinstance(proposal, RandomChoiceProposal)

    def test_normal_pool_fills_to_requested_size(self, mocker):
        config = BingoConstructionConfig(crossover_pool_size=5)

        counter = [0]

        def gen():
            counter[0] += 1
            obj = mocker.MagicMock()
            obj.__hash__ = lambda self: counter[0]
            obj.__eq__ = lambda self, other: self is other
            return obj

        mock_gen = mocker.MagicMock(side_effect=gen)
        build_agraph_proposal(x_dim=1, generator=mock_gen, operators=["+"], bingo_config=config)

        assert mock_gen.call_count >= 5

    def test_exhausted_generator_does_not_hang(self, mocker):
        config = BingoConstructionConfig(crossover_pool_size=50)
        single_obj = mocker.MagicMock()
        single_obj.__hash__ = mocker.MagicMock(return_value=42)
        single_obj.__eq__ = mocker.MagicMock(return_value=True)
        mock_gen = mocker.MagicMock(return_value=single_obj)

        import threading

        result_holder = []

        def run():
            result_holder.append(
                build_agraph_proposal(
                    x_dim=1,
                    generator=mock_gen,
                    operators=["+"],
                    bingo_config=config,
                )
            )

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "build_agraph_proposal() did not terminate"
        assert isinstance(result_holder[0], RandomChoiceProposal)

    def test_exhausted_generator_call_count_bounded(self, mocker):
        pool_size = 10
        config = BingoConstructionConfig(crossover_pool_size=pool_size)

        call_count = [0]

        def gen():
            call_count[0] += 1
            return "only_model"

        mock_gen = mocker.MagicMock(side_effect=gen)
        build_agraph_proposal(x_dim=1, generator=mock_gen, operators=["+"], bingo_config=config)

        assert call_count[0] <= pool_size + 100 + 1
