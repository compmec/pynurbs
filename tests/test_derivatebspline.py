import numpy as np
import pytest

from compmec.nurbs import SplineBaseFunction
from compmec.nurbs.knotspace import GeneratorKnotVector


@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=["tests/test_basefunctions.py::test_end"], scope="session"
)
def test_begin():
    pass


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_Constructor():
    n, p = 5, 2
    U = GeneratorKnotVector.uniform(p=p, n=n)
    N = SplineBaseFunction(U)
    dN = N.derivate()
    assert type(dN) == type(N)


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_Constructor"])
def test_Evaluator():
    n, p = 5, 2
    U = GeneratorKnotVector.uniform(p=p, n=n)
    N = SplineBaseFunction(U)
    dN = N.derivate()
    u = np.linspace(0, 1, 129)
    Nu = N(u)
    dNu = dN(u)


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_begin", "test_Evaluator"])
def test_end():
    pass
