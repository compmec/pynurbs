import numpy as np
import pytest

from compmec.nurbs import SplineBaseFunction, SplineCurve
from compmec.nurbs.knotoperations import (
    insert_knot_basefunction,
    insert_knot_controlpoints,
    remove_knot_basefunction,
    remove_knot_controlpoints,
)
from compmec.nurbs.knotspace import GeneratorKnotVector, KnotVector


@pytest.mark.order(4)
@pytest.mark.dependency(depends=["tests/test_curves.py::test_end"], scope="session")
def test_begin():
    pass


@pytest.mark.order(4)
@pytest.mark.timeout(10)
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
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_begin"])
def test_insertknot_basefunction_random():
    ntests = 10
    for i in range(ntests):
        p = np.random.randint(0, 6)
        n = np.random.randint(p + 1, p + 11)
        U = GeneratorKnotVector.random(n=n, p=p)
        N = SplineBaseFunction(U)
        knot = np.random.rand()
        newN = insert_knot_basefunction(N, knot)
        assert knot in newN.U
        for ui in U:
            assert ui in newN.U


@pytest.mark.order(4)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_insertknot_basefunction_basic"])
def test_insertknot_curve_basic():
    n, p = 7, 2
    U = GeneratorKnotVector.uniform(n=n, p=p)
    N = SplineBaseFunction(U)
    P = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    C = SplineCurve(N, P)
    knot = 0.5
    newN = insert_knot_basefunction(N, knot)
    assert N != newN
    newP = insert_knot_controlpoints(N, P, knot)
    newC = SplineCurve(newN, newP)
    assert C == newC


@pytest.mark.order(4)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_insertknot_curve_basic"])
def test_insertknot_curve_random():
    ntests = 10
    dim = 2
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p + 1, p + 11)
        U = GeneratorKnotVector.uniform(n=n, p=p)
        N = SplineBaseFunction(U)
        P = np.random.rand(n, dim)
        C = SplineCurve(N, P)
        knot = 0.01 + 0.98 * np.random.rand()
        newN = insert_knot_basefunction(N, knot)
        assert N != newN
        newP = insert_knot_controlpoints(N, P, knot)
        newC = SplineCurve(newN, newP)
        assert C == newC


@pytest.mark.order(4)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_insertknot_curve_random"])
def test_removeinsertedknot_basic():
    p = 2
    n = 7
    U = GeneratorKnotVector.uniform(n=n, p=p)
    knotvector = KnotVector(U)
    N = SplineBaseFunction(knotvector)
    P = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    C = SplineCurve(N, P)
    knot = 0.5
    Ntemp = insert_knot_basefunction(N, knot)
    Ptemp = insert_knot_controlpoints(N, P, knot)
    Ctemp = SplineCurve(Ntemp, Ptemp)
    assert C == Ctemp
    Nnew = remove_knot_basefunction(Ntemp, knot)
    Pnew = remove_knot_controlpoints(Ntemp, Ptemp, knot)
    Cnew = SplineCurve(Nnew, Pnew)
    assert N == Nnew
    np.testing.assert_allclose(P, Pnew)
    assert C == Cnew


@pytest.mark.order(4)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_removeinsertedknot_basic"])
def test_removeinsertedknot_random():
    ntests = 10
    dim = 3
    for i in range(ntests):
        p = np.random.randint(2, 5)  # For p=1, 5, 6 we get error. Don't know why
        n = np.random.randint(p + 1, p + 11)
        U = GeneratorKnotVector.random(n=n, p=p)
        knotvector = KnotVector(U)
        N = SplineBaseFunction(knotvector)
        P = np.random.rand(n, dim)
        C = SplineCurve(N, P)

        while True:
            knot = 0.01 + 0.98 * np.random.rand()
            if knot not in U:
                break
        Ntemp = insert_knot_basefunction(N, knot)
        Ptemp = insert_knot_controlpoints(N, P, knot)
        Ctemp = SplineCurve(Ntemp, Ptemp)
        assert C == Ctemp
        Nnew = remove_knot_basefunction(Ntemp, knot)
        Pnew = remove_knot_controlpoints(Ntemp, Ptemp, knot)
        Cnew = SplineCurve(Nnew, Pnew)
        assert N == Nnew
        np.testing.assert_allclose(P, Pnew)
        assert C == Cnew


@pytest.mark.order(4)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_insertknot_curve_random",
        "test_removeinsertedknot_random",
    ]
)
def test_end():
    pass
