import pytest
from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator


from pysips.prior import Prior


def get_generator(
    X_dim,
    operators,
    terminal_probability=0.1,
    constant_probability=0.5,
    max_complexity=48,
    **kwargs,
):
    USE_PYTHON = True
    USE_SIMPLIFICATION = True
    component_generator = ComponentGenerator(
        input_x_dimension=X_dim,
        terminal_probability=terminal_probability,
        constant_probability=constant_probability,
    )
    for comp in operators:
        component_generator.add_operator(comp)
    generator = AGraphGenerator(
        max_complexity,
        component_generator,
        use_python=USE_PYTHON,
        use_simplification=USE_SIMPLIFICATION,
    )

    return generator


@pytest.mark.parametrize("N", [10, 500])
def test_prior_makes_initial_set_of_unique_expressions(N):
    """Test rvs with different values of N"""
    generator = get_generator(1, ["+", "-", "*"])
    prior = Prior(generator)

    result = prior.rvs(N)

    assert result.shape == (N, 1)
    unique_models = set(result.flatten())
    assert len(unique_models) == N
