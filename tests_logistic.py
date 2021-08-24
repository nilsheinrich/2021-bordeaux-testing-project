import numpy as np
import pytest
from hands_on_logistic import logistic_map, iterative_f


#SEED = 1
#random_state = np.random.RandomState(SEED)


@pytest.mark.notiterative
@pytest.mark.parametrize("x, r, expected", [(0.1, 2.2, 0.198), (0.2, 3.4, 0.344), (0.75, 1.7, 0.31875)])
def test_logistic_map(r, x, expected):
    a = logistic_map(r, x)
    assert np.isclose(a, expected)


@pytest.mark.iterative
#@pytest.mark.parametrize('x', list(np.arange(0.1, 1, 0.01)))
@pytest.mark.parametrize('x', np.random.uniform(1e-5, 1, 10))
def test_convergence(x):
    output = iterative_f(100, x, 1.5)[-1]
    expected = 1/3.
    assert np.isclose(output, expected)    


