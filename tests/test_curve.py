import pytest
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs import SplineCurve
from compmec.nurbs.curves import BaseCurve
from compmec.nurbs.knotspace import GeneratorKnotVector
import numpy as np


@pytest.mark.order(3)
@pytest.mark.dependency(
	depends=["tests/test_basefunctions.py::test_end"],
    scope='session')
def test_begin():
    pass


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_CreationSplineCurve():
    N = SplineBaseFunction([0, 0, 1, 1]) # p = 1, n = 2
    P = np.random.rand(2, 3)
    C = SplineCurve(N, P)
    assert isinstance(C, BaseCurve)
    assert callable(N)
 

@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_CreationSplineCurve"])
def test_end():
    pass

def main():
    test_begin()
    test_CreationSplineCurve()
    test_end()

if __name__ == "__main__":
    main()
