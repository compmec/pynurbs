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
    p = np.random.randint(1, 5)
    n = np.random.randint(p+1, p+7)
    n = p+1
    U = GeneratorKnotVector.uniform(p=p, n=n)
    N = SplineBaseFunction([0, 0, 1, 1]) # p = 1, n = 2
    P = np.random.rand(2, 3)
    C = SplineCurve(N, P)
    assert isinstance(C, BaseCurve)
    assert callable(C)
 

@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationSplineCurve"])
def test_SplineScalarCurve():
    npts = 129
    for p in range(1, 4):
        for n in range(p+1, p+7):
            U = GeneratorKnotVector.random(p=p, n=n)
            N = SplineBaseFunction(U)
            P = np.random.rand(n)
            C = SplineCurve(N, P)
            t = np.linspace(0, 1, npts)
            Cuv = C(t)
            assert Cuv.shape == t.shape

@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationSplineCurve"])
def test_SplineVectorialCurve():
    npts = 129
    dim = np.random.randint(1, 5)
    for p in range(1, 4):
        for n in range(p+1, p+7):
            U = GeneratorKnotVector.random(p=p, n=n)
            N = SplineBaseFunction(U)
            P = np.random.rand(n, dim)
            C = SplineCurve(N, P)
            t = np.linspace(0, 1, npts)
            Cuv = C(t)
            assert Cuv.shape == (npts, dim)

@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_SplineScalarCurve", "test_SplineVectorialCurve"])
def test_end():
    pass

def main():
    test_begin()
    test_CreationSplineCurve()
    test_SplineScalarCurve()
    test_SplineVectorialCurve()
    test_end()

if __name__ == "__main__":
    main()
