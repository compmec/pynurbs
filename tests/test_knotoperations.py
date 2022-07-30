import pytest
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs import SplineCurve
from compmec.nurbs.knotspace import KnotVector, getU_random, getU_uniform
from compmec.nurbs.advanced import insert_knot_basefunction, insert_knot_controlpoints
from matplotlib import pyplot as plt 
from geomdl import BSpline
from geomdl.operations import insert_knot
import numpy as np

@pytest.mark.order(4)
@pytest.mark.dependency(
	depends=["tests/test_curve.py::test_end"],
    scope='session')
def test_begin():
    pass


@pytest.mark.order(4)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_insertknot_basefunction_basic():
    N = SplineBaseFunction([0, 0, 1, 1])
    newN = insert_knot_basefunction(N, 0.5)
    assert newN.U == [0, 0, 0.5, 1, 1]

    N = SplineBaseFunction([0, 0, 0, 1, 1, 1])
    newN = insert_knot_basefunction(N, 0.5)
    assert newN.U == [0, 0, 0, 0.5, 1, 1, 1]

    N = SplineBaseFunction([0, 0, 0, 0, 1, 1, 1, 1])
    newN = insert_knot_basefunction(N, 0.5)
    assert newN.U == [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
    newN = insert_knot_basefunction(newN, 0.5)
    assert newN.U == [0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1]

    N = SplineBaseFunction([0, 0, 0, 0, 1, 1, 1, 1])
    newN = insert_knot_basefunction(N, 0.5, 2)
    assert newN.U == [0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1]


@pytest.mark.order(4)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_insertknot_basefunction_basic"])
def test_insertknot_curve_basic():
    n, p = 10, 3
    U = getU_uniform(n=n, p=p)
    N = SplineBaseFunction(U)
    P = np.random.rand(n, 2)
    C = SplineCurve(N, P)
    knot = 0.01+0.98*np.random.rand()
    newN = insert_knot_basefunction(N, knot)
    newP = insert_knot_controlpoints(N.U, P, knot)
    newC = SplineCurve(newN, newP)
    utest = np.linspace(0, 1, 2*n+1)
    Cugood = C(utest)
    Cutest = newC(utest)
    np.testing.assert_almost_equal(Cugood, Cutest)


@pytest.mark.order(4)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_insertknot_curve_basic"])
def test_insertknot_with_geomdl():
    """
    This function uses the library for Nurbs in, if the data are compatible
        https://github.com/orbingol/NURBS-Python
    """
    ntests = 10
    dim = 2
    utest = np.linspace(0, 1, 129)
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = 10*np.random.randint(p+1, p+11)
        U = getU_random(n=n, p=p)
        knotvector = KnotVector(U)
        N = SplineBaseFunction(knotvector)
        P = np.random.rand(n, dim)
        Pgeomdl = []
        for i, Pi in enumerate(P):
            Pgeomdl.append(list(Pi))
        Cgeomdl = BSpline.Curve()
        Cgeomdl.degree = p
        Cgeomdl.ctrlpts = Pgeomdl
        Cgeomdl.knotvector = U

        knot = 0.01+0.98*np.random.rand()
        newN = insert_knot_basefunction(N, knot)
        newP = insert_knot_controlpoints(N.U, P, knot)
        newCcustom = SplineCurve(newN, newP)

        insert_knot(obj=Cgeomdl, param=[knot], num=[1])
        for i, ui in enumerate(utest):
            Pcustom = newCcustom(ui)
            Pgeomdl = Cgeomdl.evaluate_single(ui)
            np.testing.assert_allclose(Pcustom, Pgeomdl)


@pytest.mark.order(4)
@pytest.mark.dependency(depends=["test_begin", "test_curve_basic"])
def test_end():
    pass

def main():
    test_insertknot_basefunction_basic()
    test_insertknot_curve_basic()
    test_insertknot_with_geomdl()
    # test_curve_plot()


if __name__ == "__main__":
    main()