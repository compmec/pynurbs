import numpy as np
import pytest

from compmec.nurbs.basefunctions import SplineBaseFunction
from compmec.nurbs.curves import SplineCurve
from compmec.nurbs.degreeoperations import (
    degree_elevation_basefunction,
    degree_elevation_controlpoints,
    degree_reduction_basefunction,
    degree_reduction_controlpoints,
)
from compmec.nurbs.knotspace import GeneratorKnotVector


@pytest.mark.order(5)
@pytest.mark.dependency(
    depends=["tests/test_knotoperations.py::test_end"], scope="session"
)
def test_begin():
    pass


@pytest.mark.order(5)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_begin"])
def test_degreeelevation_basefuncion_bezier():
    for p in range(0, 5):
        U = GeneratorKnotVector.bezier(p)
        N = SplineBaseFunction(U)
        Ntest = degree_elevation_basefunction(N)
        U = GeneratorKnotVector.bezier(p + 1)
        Ngood = SplineBaseFunction(U)
        assert Ntest == Ngood


@pytest.mark.order(5)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_begin"])
def test_degreeelevation_basefuncion_simple():
    N = SplineBaseFunction([0, 0, 0.5, 1, 1])
    Ntest = degree_elevation_basefunction(N)
    Ngood = SplineBaseFunction([0, 0, 0, 0.5, 0.5, 1, 1, 1])
    assert Ntest == Ngood

    N = SplineBaseFunction([0, 0, 0.2, 0.7, 1, 1])
    Ntest = degree_elevation_basefunction(N)
    Ngood = SplineBaseFunction([0, 0, 0, 0.2, 0.2, 0.7, 0.7, 1, 1, 1])
    assert Ntest == Ngood

    N = SplineBaseFunction([0, 0, 0, 0.2, 0.7, 1, 1, 1])
    Ntest = degree_elevation_basefunction(N)
    Ngood = SplineBaseFunction([0, 0, 0, 0, 0.2, 0.2, 0.7, 0.7, 1, 1, 1, 1])
    assert Ntest == Ngood


@pytest.mark.order(5)
@pytest.mark.timeout(20)
@pytest.mark.dependency(depends=["test_degreeelevation_basefuncion_simple"])
def test_degreeelevation_controlpoints_bezier():
    ntests = 5
    dim = 1
    utest = np.linspace(0, 1, 33)
    for i in range(ntests):
        p = np.random.randint(1, 6)
        U = GeneratorKnotVector.bezier(p=p)
        N = SplineBaseFunction(U)
        P = np.random.rand(p + 1, dim)
        C = SplineCurve(N, P)

        Ninc = degree_elevation_basefunction(N)
        Pinc = degree_elevation_controlpoints(N, P)
        Cinc = SplineCurve(Ninc, Pinc)

        Cgood = C(utest)
        Ctest = Cinc(utest)
        np.testing.assert_allclose(Ctest, Cgood)


@pytest.mark.order(5)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_degreeelevation_controlpoints_bezier"])
def test_degreeelevation_controlpoints_uniform():
    ntests = 5
    dim = 1
    utest = np.linspace(0, 1, 33)
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p + 1, p + 11)
        U = GeneratorKnotVector.uniform(p=p, n=n)
        N = SplineBaseFunction(U)
        P = np.random.rand(n, dim)
        C = SplineCurve(N, P)

        Ninc = degree_elevation_basefunction(N)
        Pinc = degree_elevation_controlpoints(N, P)
        Cinc = SplineCurve(Ninc, Pinc)

        Cgood = C(utest)
        Ctest = Cinc(utest)
        np.testing.assert_allclose(Ctest, Cgood)


@pytest.mark.order(5)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_degreeelevation_controlpoints_uniform"])
def test_degreeelevation_controlpoints_random():
    ntests = 5
    dim = 1
    utest = np.linspace(0, 1, 33)
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p + 1, p + 11)
        U = GeneratorKnotVector.random(p=p, n=n)
        N = SplineBaseFunction(U)
        P = np.random.rand(n, dim)
        C = SplineCurve(N, P)

        Ninc = degree_elevation_basefunction(N)
        Pinc = degree_elevation_controlpoints(N, P)
        Cinc = SplineCurve(Ninc, Pinc)

        Cgood = C(utest)
        Ctest = Cinc(utest)
        np.testing.assert_allclose(Ctest, Cgood)


@pytest.mark.order(5)
@pytest.mark.timeout(10)
@pytest.mark.dependency(depends=["test_degreeelevation_controlpoints_random"])
def test_degreeelevationreduction_controlpoints_random():
    ntests = 5
    dim = 1
    utest = np.linspace(0, 1, 33)
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p + 1, p + 11)
        U = GeneratorKnotVector.random(p=p, n=n)
        N = SplineBaseFunction(U)
        P = np.random.rand(n, dim)
        C = SplineCurve(N, P)

        Ninc = degree_elevation_basefunction(N)
        Pinc = degree_elevation_controlpoints(N, P)

        Nred = degree_reduction_basefunction(Ninc)
        Pred = degree_reduction_controlpoints(Ninc, Pinc)
        Cred = SplineCurve(Nred, Pred)

        Cgood = C(utest)
        Ctest = Cred(utest)
        np.testing.assert_allclose(Ctest, Cgood)


@pytest.mark.order(5)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_degreeelevation_controlpoints_random",
        "test_degreeelevationreduction_controlpoints_random",
    ]
)
def test_end():
    pass
