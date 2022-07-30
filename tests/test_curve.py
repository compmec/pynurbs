from xml.etree.ElementTree import PI
import pytest
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs import SplineCurve
from compmec.nurbs.curves import BaseCurve
from compmec.nurbs.knotspace import KnotVector, getU_uniform, getU_random
from geomdl import BSpline
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
@pytest.mark.timeout(20)
@pytest.mark.dependency(depends=["test_CreationSplineCurve"])
def test_with_geomdl():
    """
    This function uses the library for Nurbs in, if the data are compatible
        https://github.com/orbingol/NURBS-Python
    """
    ntests = 10
    dim = 2
    utest = np.linspace(0, 1, 129)
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p+1, p+11)
        U = getU_random(n=n, p=p)
        N = SplineBaseFunction(U)
        P = np.random.rand(n, dim)
        Pgeomdl = []
        for i, Pi in enumerate(P):
            Pgeomdl.append(list(Pi))
        Ccustom = SplineCurve(N, P)
        Cgeomdl = BSpline.Curve()
        Cgeomdl.degree = p
        Cgeomdl.ctrlpts = Pgeomdl
        Cgeomdl.knotvector = U
        for i, ui in enumerate(utest):
            Pcustom = Ccustom(ui)
            Pgeomdl = Cgeomdl.evaluate_single(ui)
            np.testing.assert_allclose(Pcustom, Pgeomdl)
    

@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_CreationSplineCurve", "test_with_geomdl"])
def test_end():
    pass

def main():
    test_begin()
    test_CreationSplineCurve()
    test_with_geomdl()
    test_end()

if __name__ == "__main__":
    main()
