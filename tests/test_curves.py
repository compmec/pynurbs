import numpy as np
import pytest

from compmec.nurbs import KnotVector, RationalCurve, SplineCurve
from compmec.nurbs.curves import BaseCurve
from compmec.nurbs.knotspace import GeneratorKnotVector


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=["tests/test_basefunctions.py::test_end"], scope="session"
)
def test_begin():
    pass


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_begin"])
def test_CreationSplineCurve():
    p = np.random.randint(0, 5)
    ndim = np.random.randint(1, 4)
    n = np.random.randint(p + 1, p + 7)
    U = GeneratorKnotVector.uniform(p=p, n=n)
    P = np.random.rand(n, ndim)
    C = SplineCurve(U, P)
    assert isinstance(C, BaseCurve)
    assert callable(C)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_CreationSplineCurve"])
def test_SplineScalarCurve():
    npts = 17
    for p in range(1, 4):
        for n in range(p + 1, p + 7):
            U = GeneratorKnotVector.random(p=p, n=n)
            P = np.random.rand(n)
            C = SplineCurve(U, P)
            t = np.linspace(0, 1, npts)
            Cuv = C(t)
            assert Cuv.shape == t.shape


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_CreationSplineCurve"])
def test_SplineVectorialCurve():
    npts = 17
    for dim in range(1, 6):
        for p in range(1, 4):
            for n in range(p + 1, p + 7):
                U = GeneratorKnotVector.random(p=p, n=n)
                P = np.random.rand(n, dim)
                C = SplineCurve(U, P)
                t = np.linspace(0, 1, npts)
                Cuv = C(t)
                assert Cuv.shape == (npts, dim)


@pytest.mark.order(3)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_CreationSplineCurve"])
def test_EqualDiffCurves():
    npts = 17
    dim = np.random.randint(1, 5)
    for p in range(1, 4):
        for n in range(p + 1, p + 7):
            U = GeneratorKnotVector.random(p=p, n=n)
            P1 = np.random.rand(n, dim)
            P2 = np.random.rand(n, dim)
            P3 = np.copy(P1)
            C1 = SplineCurve(U, P1)
            C2 = SplineCurve(U, P2)
            C3 = SplineCurve(U, P3)
            t = np.linspace(0, 1, npts)
            assert C1 == C3
            assert C1 != C2
            assert C2 != C3


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_EqualDiffCurves"])
def test_SumDiffTwoCurves():
    npts = 129
    dim = np.random.randint(1, 5)
    for p in range(1, 4):
        for n in range(p + 1, p + 7):
            U = GeneratorKnotVector.random(p=p, n=n)
            P1 = np.random.rand(n, dim)
            P2 = np.random.rand(n, dim)
            Ps = P1 + P2
            Pd = P1 - P2
            C1 = SplineCurve(U, P1)
            C2 = SplineCurve(U, P2)
            Cs = SplineCurve(U, Ps)
            Cd = SplineCurve(U, Pd)
            t = np.linspace(0, 1, npts)
            assert C1 + C2 == Cs
            assert C1 - C2 == Cd


@pytest.mark.order(3)
@pytest.mark.timeout(10)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_SplineScalarCurve",
        "test_SplineVectorialCurve",
        "test_SumDiffTwoCurves",
    ]
)
def test_others():
    p = np.random.randint(0, 6)
    n = np.random.randint(p + 3, p + 7)
    ndim = np.random.randint(1, 4)
    U = GeneratorKnotVector.random(p=p, n=n)
    P = np.random.uniform(-1, 1, (n, ndim))
    w = np.random.uniform(1, 2, n)
    C = SplineCurve(U, P)
    C.derivate()
    assert C.p == p
    assert C.n == n

    C = RationalCurve(U, P)
    C.derivate()
    C.w = w

    with pytest.raises(ValueError):
        P = np.random.uniform(-1, 1, (n + 1, ndim))
        SplineCurve(U, P)
    with pytest.raises(ValueError):
        P = np.random.uniform(-1, 1, (n + 1, ndim))
        RationalCurve(U, P)

    with pytest.raises(TypeError):
        C == 1

    with pytest.raises(TypeError):
        C + 1


@pytest.mark.order(3)
@pytest.mark.timeout(10)
@pytest.mark.skip(reason="To correct algorithms of knot insertion and remotion")
@pytest.mark.dependency(depends=["test_others"])
def test_others2():
    p = np.random.randint(0, 6)
    n = np.random.randint(p + 3, p + 7)
    ndim = np.random.randint(1, 4)
    U1 = GeneratorKnotVector.random(p=p, n=n)
    U2 = GeneratorKnotVector.random(p=p, n=n)
    P1 = np.random.uniform(-1, 1, (n, ndim))
    P2 = np.random.uniform(-1, 1, (n, ndim))
    P3 = np.random.uniform(-1, 1, (n, ndim + 1))
    C1 = SplineCurve(U1, P1)
    C2 = SplineCurve(U2, P2)
    C3 = SplineCurve(U2, P3)
    with pytest.raises(ValueError):
        C1 + C2
    with pytest.raises(ValueError):
        C2 + C3
    C1.knot_insert(0.5)
    C2.knot_insert([0.2, 0.7])
    with pytest.raises(ValueError):
        C1 + C2
    with pytest.raises(ValueError):
        C1.knot_insert([[0.5]])
    C1.knot_remove(0.5)
    C2.knot_remove([0.2, 0.7])
    with pytest.raises(ValueError):
        C1.knot_remove([[0.5]])

    C1orig = SplineCurve(U1, P1)
    C1 = SplineCurve(U1, P1)
    C1.degree_increase()
    C1.degree_decrease()
    assert C1 == C1orig


@pytest.mark.order(3)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_others2"])
def test_others3():
    p = np.random.randint(0, 6)
    n = np.random.randint(p + 3, p + 7)
    ndim = np.random.randint(1, 4)
    U1 = GeneratorKnotVector.random(p=p, n=n)
    U2 = GeneratorKnotVector.random(p=p, n=n)
    P1 = np.random.uniform(-1, 1, (n, ndim))
    P2 = np.random.uniform(-1, 1, (n, ndim))
    P3 = np.random.uniform(-1, 1, (n, ndim + 1))
    C1 = SplineCurve(U1, P1)
    C2 = SplineCurve(U2, P2)
    C3 = SplineCurve(U2, P3)
    C1 = RationalCurve(U1, P1)
    C1.w = np.random.uniform(1, 2, n)
    C2 = RationalCurve(U2, P2)
    C2.w = C1.w
    C3 = RationalCurve(U1, P1)
    C3.w = np.random.uniform(1, 2, n)
    C4 = RationalCurve(U1, P1)
    C4.w = C1.w
    assert C1 != C2
    assert C1 != C3
    assert C2 != C3
    assert C1 == C4


@pytest.mark.order(3)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_others3"])
def test_curve_insert_oneknot():
    Uorig = [0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5]  # Example 5.1 nurbs book
    Uorig = np.array(Uorig, dtype="float64") / 5  # p = 3, n = 8
    Uorig = KnotVector(Uorig)
    p, npts = 3, 8
    knot = 0.5
    ndim = 3
    P = np.random.uniform(-1, 1, (npts, ndim))
    Corig = SplineCurve(Uorig, P)

    Q = np.zeros((npts + 1, ndim), dtype="float64")
    Q[:3] = P[:3]
    Q[3] = (1 / 6) * P[2] + (5 / 6) * P[3]
    Q[4] = (1 / 2) * P[3] + (1 / 2) * P[4]
    Q[5] = (5 / 6) * P[4] + (1 / 6) * P[5]
    Q[6:] = P[5:]

    Uinse = list(Uorig)
    Uinse.insert(5, knot)
    Uinse = KnotVector(Uinse)
    Cinse = SplineCurve(Uinse, Q)

    assert Corig == Cinse


@pytest.mark.order(3)
@pytest.mark.timeout(10)
@pytest.mark.skip(reason="sometimes fail")
@pytest.mark.dependency(depends=["test_curve_insert_oneknot"])
def test_curve_insertremove_oneknot_random():
    ntests = 100
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p + 1, p + 7)
        dim = np.random.randint(1, 6)
        if dim == 0:
            controlpoints = np.random.uniform(-1, 1, n)
        else:
            controlpoints = np.random.uniform(-1, 1, (n, dim))
        knotvector = GeneratorKnotVector.random(p=p, n=n)
        curve = SplineCurve(knotvector, controlpoints)

        knot = np.random.uniform(0.01, 0.99)
        curve.knot_insert(knot)
        curve.knot_remove(knot)

        assert curve.U == knotvector
        np.testing.assert_allclose(curve.P, controlpoints, atol=1e-12)


@pytest.mark.order(3)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_others3"])
def test_curve_increasedecrease_degree_random():
    ntests = 100
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p + 1, p + 7)
        dim = np.random.randint(1, 6)
        if dim == 0:
            controlpoints = np.random.uniform(-1, 1, n)
        else:
            controlpoints = np.random.uniform(-1, 1, (n, dim))
        knotvector = GeneratorKnotVector.random(p=p, n=n)
        curve = SplineCurve(knotvector, controlpoints)

        curve.degree_increase()
        curve.degree_decrease()

        assert curve.U == knotvector
        np.testing.assert_allclose(curve.P, controlpoints, atol=1e-12)


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_SplineScalarCurve",
        "test_SplineVectorialCurve",
        "test_SumDiffTwoCurves",
        "test_others",
        "test_curve_insertremove_oneknot_random",
        "test_curve_increasedecrease_degree_random",
    ]
)
def test_end():
    pass
