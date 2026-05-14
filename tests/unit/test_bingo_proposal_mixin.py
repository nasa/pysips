import pytest

from pysips.bingo_proposal_mixin import BingoProposalMixin
from pysips.random_choice_proposal import RandomChoiceProposal


class _ConcreteMixin(BingoProposalMixin):
    """Minimal concrete subclass so we can instantiate BingoProposalMixin."""


class TestGetProposal:
    """Tests for BingoProposalMixin._get_proposal()."""

    def _make_mixin(self, **kwargs):
        return _ConcreteMixin(**kwargs)

    def test_returns_random_choice_proposal(self, mocker):
        """_get_proposal() returns a RandomChoiceProposal."""
        mixin = self._make_mixin()
        generator = mixin._get_generator(x_dim=1, operators=["+", "*"])
        proposal = mixin._get_proposal(
            x_dim=1, generator=generator, operators=["+", "*"]
        )
        assert isinstance(proposal, RandomChoiceProposal)

    def test_normal_pool_fills_to_requested_size(self, mocker):
        """When the generator has enough unique models, the crossover pool
        reaches the requested crossover_pool_size."""
        mixin = self._make_mixin(crossover_pool_size=5)

        # Generator that always produces distinct objects.
        counter = [0]

        def gen():
            counter[0] += 1
            obj = mocker.MagicMock()
            obj.__hash__ = lambda self: counter[0]
            obj.__eq__ = lambda self, other: self is other
            return obj

        mock_gen = mocker.MagicMock(side_effect=gen)
        proposal = mixin._get_proposal(x_dim=1, generator=mock_gen, operators=["+"])

        # Should have called the generator at least 5 times to fill the pool.
        assert mock_gen.call_count >= 5

    def test_exhausted_generator_does_not_hang(self, mocker):
        """Regression: _get_proposal() must not loop forever when the generator
        cannot produce crossover_pool_size unique models (was an infinite loop)."""
        mixin = self._make_mixin(crossover_pool_size=50)

        # Generator only ever returns the same object.
        single_obj = mocker.MagicMock()
        single_obj.__hash__ = mocker.MagicMock(return_value=42)
        single_obj.__eq__ = mocker.MagicMock(return_value=True)
        mock_gen = mocker.MagicMock(return_value=single_obj)

        import threading

        result_holder = []

        def run():
            result_holder.append(
                mixin._get_proposal(x_dim=1, generator=mock_gen, operators=["+"])
            )

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5)
        assert (
            not t.is_alive()
        ), "_get_proposal() did not terminate — infinite loop detected"
        assert isinstance(result_holder[0], RandomChoiceProposal)

    def test_exhausted_generator_call_count_bounded(self, mocker):
        """When the generator is exhausted, the total number of calls is bounded
        (at most crossover_pool_size + 100 consecutive failures)."""
        pool_size = 10
        mixin = self._make_mixin(crossover_pool_size=pool_size)

        call_count = [0]

        def gen():
            call_count[0] += 1
            return "only_model"

        mock_gen = mocker.MagicMock(side_effect=gen)
        mixin._get_proposal(x_dim=1, generator=mock_gen, operators=["+"])

        # 1 unique model + up to 100 consecutive failures before breaking.
        assert call_count[0] <= pool_size + 100 + 1
