import numpy as np
import pytest

from compmec.nurbs import RationalFunction, SplineFunction
from compmec.nurbs.knotspace import GeneratorKnotVector


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["tests/test_knotspace.py::test_end"], scope="session")
def test_begin():
    pass


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_begin"])
def test_CreationSplineFunction():
    N = SplineFunction([0, 0, 1, 1])
    assert callable(N)
    assert N.degree == 1
    assert N.npts == 2
    N = SplineFunction([0, 0, 0.5, 1, 1])
    assert callable(N)
    assert N.degree == 1
    assert N.npts == 3
    N = SplineFunction([0, 0, 0, 1, 1, 1])
    assert callable(N)
    assert N.degree == 2
    assert N.npts == 3
    N = SplineFunction([0, 0, 0, 0.5, 1, 1, 1])
    assert callable(N)
    assert N.degree == 2
    assert N.npts == 4


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_CreationSplineFunction"])
def test_SplineEvaluationFunctions_p1n2():
    knotvector = [0, 0, 1, 1]  # degree = 1, npts = 2
    N = SplineFunction(knotvector)
    N[0, 0]
    N[1, 0]
    N[0, 1]
    N[1, 1]
    N[:, 0]
    N[:, 1]
    N[:]


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_SplineEvaluationFunctions_p1n2"])
def test_SplineEvaluationFunctions_p1n3():
    knotvector = [0, 0, 0.5, 1, 1]  # degree = 1, npts = 3
    N = SplineFunction(knotvector)
    N[0, 0]
    N[1, 0]
    N[2, 0]
    N[0, 1]
    N[1, 1]
    N[2, 1]
    N[:, 0]
    N[:, 1]
    N[:]


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_SplineEvaluationFunctions_p1n2"])
def test_somesinglevalues_p1n2():
    knotvector = [0, 0, 1, 1]  # degree = 1, npts = 2
    N = SplineFunction(knotvector)
    assert N[0, 0](0.0) == 0
    assert N[0, 0](0.5) == 0
    assert N[0, 0](1.0) == 0
    assert N[1, 0](0.0) == 1
    assert N[1, 0](0.5) == 1
    assert N[1, 0](1.0) == 1
    assert N[0, 1](0.0) == 1
    assert N[0, 1](0.5) == 0.5
    assert N[0, 1](1.0) == 0
    assert N[1, 1](0.0) == 0
    assert N[1, 1](0.5) == 0.5
    assert N[1, 1](1.0) == 1


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_somesinglevalues_p1n2"])
def test_somesinglevalues_p2n3():
    knotvector = [0, 0, 0, 1, 1, 1]  # degree = 2, npts = 3
    N = SplineFunction(knotvector)
    assert N[0, 0](0.0) == 0
    assert N[0, 0](0.5) == 0
    assert N[0, 0](1.0) == 0
    assert N[1, 0](0.0) == 0
    assert N[1, 0](0.5) == 0
    assert N[1, 0](1.0) == 0
    assert N[2, 0](0.0) == 1
    assert N[2, 0](0.5) == 1
    assert N[2, 0](1.0) == 1
    assert N[0, 1](0.0) == 0
    assert N[0, 1](0.5) == 0
    assert N[0, 1](1.0) == 0
    assert N[1, 1](0.0) == 1
    assert N[1, 1](0.5) == 0.5
    assert N[1, 1](1.0) == 0
    assert N[2, 1](0.0) == 0
    assert N[2, 1](0.5) == 0.5
    assert N[2, 1](1.0) == 1
    assert N[0, 2](0.0) == 1
    assert N[0, 2](0.5) == 0.25
    assert N[0, 2](1.0) == 0
    assert N[1, 2](0.0) == 0
    assert N[1, 2](0.5) == 0.5
    assert N[1, 2](1.0) == 0
    assert N[2, 2](0.0) == 0
    assert N[2, 2](0.5) == 0.25
    assert N[2, 2](1.0) == 1


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_somesinglevalues_p1n2"])
def test_tablevalues_p1n2():
    knotvector = [0, 0, 1, 1]  # degree = 1, npts = 2
    utest = np.linspace(0, 1, 11)
    N = SplineFunction(knotvector)
    N0 = N[:, 0]
    N1 = N[:, 1]
    M0test = N0(utest)
    M0good = np.array([[0] * 11, [1] * 11])
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array([np.linspace(1, 0, 11), np.linspace(0, 1, 11)])
    np.testing.assert_allclose(M1test, M1good)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_tablevalues_p1n2"])
def test_tablevalues_p1n3():
    knotvector = [0, 0, 0.5, 1, 1]  # degree = 1, npts = 3
    utest = np.linspace(0, 1, 11)
    N = SplineFunction(knotvector)
    N0 = N[:, 0]
    N1 = N[:, 1]
    M0test = N0(utest)
    M0good = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ]
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = [
        [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ]
    np.testing.assert_allclose(M1test, M1good)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_tablevalues_p1n2", "test_somesinglevalues_p2n3"])
def test_tablevalues_p2n3():
    knotvector = [0, 0, 0, 1, 1, 1]  # degree = 2, npts = 3
    utest = np.linspace(0, 1, 11)
    N = SplineFunction(knotvector)
    N0 = N[:, 0]
    N1 = N[:, 1]
    N2 = N[:, 2]
    M0test = N0(utest)
    M0good = np.array([[0] * 11, [0] * 11, [1] * 11])
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array([[0] * 11, np.linspace(1, 0, 11), np.linspace(0, 1, 11)])
    np.testing.assert_allclose(M1test, M1good)
    M2test = N2(utest)
    M2good = np.array(
        [
            [1.0, 0.81, 0.64, 0.49, 0.36, 0.25, 0.16, 0.09, 0.04, 0.01, 0.0],
            [0.0, 0.18, 0.32, 0.42, 0.48, 0.50, 0.48, 0.42, 0.32, 0.18, 0.0],
            [0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0],
        ]
    )
    np.testing.assert_allclose(M2test, M2good)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_tablevalues_p2n3"])
def test_tablevalues_p2n4():
    knotvector = [0, 0, 0, 0.5, 1, 1, 1]  # degree = 2, npts = 4
    utest = np.linspace(0, 1, 11)
    N = SplineFunction(knotvector)
    N0 = N[:, 0]
    N1 = N[:, 1]
    N2 = N[:, 2]
    M0test = N0(utest)
    M0good = np.array(
        [
            [0] * 11,
            [0] * 11,
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array(
        [
            [0] * 11,
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ]
    )
    np.testing.assert_allclose(M1test, M1good)
    M2test = N2(utest)
    M2good = np.array(
        [
            [1, 0.64, 0.36, 0.16, 0.04, 0.0, 0.00, 0.00, 0.00, 0.00, 0],
            [0, 0.34, 0.56, 0.66, 0.64, 0.5, 0.32, 0.18, 0.08, 0.02, 0],
            [0, 0.02, 0.08, 0.18, 0.32, 0.5, 0.64, 0.66, 0.56, 0.34, 0],
            [0, 0.00, 0.00, 0.00, 0.00, 0.0, 0.04, 0.16, 0.36, 0.64, 1],
        ]
    )
    np.testing.assert_allclose(M2test, M2good)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_tablevalues_p2n3"])
def test_tablevalues_p3n4():
    knotvector = [0, 0, 0, 0, 1, 1, 1, 1]  # degree = 3, npts = 4
    utest = np.linspace(0, 1, 11)
    N = SplineFunction(knotvector)
    N0 = N[:, 0]
    N1 = N[:, 1]
    N2 = N[:, 2]
    N3 = N[:, 3]
    M0test = N0(utest)
    M0good = np.array([[0] * 11, [0] * 11, [0] * 11, [1] * 11])
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array(
        [[0] * 11, [0] * 11, np.linspace(1, 0, 11), np.linspace(0, 1, 11)]
    )
    np.testing.assert_allclose(M1test, M1good)
    M2test = N2(utest)
    M2good = np.array(
        [
            [0] * 11,
            [1.0, 0.81, 0.64, 0.49, 0.36, 0.25, 0.16, 0.09, 0.04, 0.01, 0.0],
            [0.0, 0.18, 0.32, 0.42, 0.48, 0.50, 0.48, 0.42, 0.32, 0.18, 0.0],
            [0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0],
        ]
    )
    np.testing.assert_allclose(M2test, M2good)
    M3test = N3(utest)
    M3good = np.array(
        [
            [1, 0.729, 0.512, 0.343, 0.216, 0.125, 0.064, 0.027, 0.008, 0.001, 0],
            [0, 0.243, 0.384, 0.441, 0.432, 0.375, 0.288, 0.189, 0.096, 0.027, 0],
            [0, 0.027, 0.096, 0.189, 0.288, 0.375, 0.432, 0.441, 0.384, 0.243, 0],
            [0, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.512, 0.729, 1],
        ]
    )
    np.testing.assert_allclose(M3test, M3good)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_tablevalues_p2n3", "test_tablevalues_p3n4"])
def test_tablevalues_p3n5():
    knotvector = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]  # degree = 3, npts = 5
    utest = np.linspace(0, 1, 11)
    N = SplineFunction(knotvector)
    N0 = N[:, 0]
    N1 = N[:, 1]
    N2 = N[:, 2]
    N3 = N[:, 3]
    M0test = N0(utest)
    M0good = np.array(
        [
            [0] * 11,
            [0] * 11,
            [0] * 11,
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array(
        [
            [0] * 11,
            [0] * 11,
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ]
    )
    np.testing.assert_allclose(M1test, M1good)
    M2test = N2(utest)
    M2good = np.array(
        [
            [0] * 11,
            [1, 0.64, 0.36, 0.16, 0.04, 0.0, 0.00, 0.00, 0.00, 0.00, 0],
            [0, 0.34, 0.56, 0.66, 0.64, 0.5, 0.32, 0.18, 0.08, 0.02, 0],
            [0, 0.02, 0.08, 0.18, 0.32, 0.5, 0.64, 0.66, 0.56, 0.34, 0],
            [0, 0.00, 0.00, 0.00, 0.00, 0.0, 0.04, 0.16, 0.36, 0.64, 1],
        ]
    )
    np.testing.assert_allclose(M2test, M2good)
    M3test = N3(utest)
    M3good = np.array(
        [
            [1, 0.512, 0.216, 0.064, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0],
            [0, 0.434, 0.592, 0.558, 0.416, 0.250, 0.128, 0.054, 0.016, 0.002, 0],
            [0, 0.052, 0.176, 0.324, 0.448, 0.500, 0.448, 0.324, 0.176, 0.052, 0],
            [0, 0.002, 0.016, 0.054, 0.128, 0.250, 0.416, 0.558, 0.592, 0.434, 0],
            [0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.008, 0.064, 0.216, 0.512, 1],
        ]
    )
    np.testing.assert_allclose(M3test, M3good)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_tablevalues_p3n5"])
def test_tableUuniform_sum1():
    ntests = 10
    for i in range(ntests):
        degree = np.random.randint(1, 6)
        npts = np.random.randint(degree + 1, degree + 21)
        knotvector = GeneratorKnotVector.uniform(degree, npts)
        u = np.random.rand(11)
        N = SplineFunction(knotvector)
        for j in range(degree + 1):
            M = N[:, j](u)
            assert np.all(M >= 0)
            for k in range(len(u)):
                np.testing.assert_almost_equal(np.sum(M[:, k]), 1)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_tableUuniform_sum1"])
def test_tableUrandom_sum1():
    ntests = 10
    for i in range(ntests):
        degree = np.random.randint(1, 6)
        npts = np.random.randint(degree + 1, degree + 21)
        knotvector = GeneratorKnotVector.random(degree, npts)
        u = np.random.rand(11)
        N = SplineFunction(knotvector)
        for j in range(degree + 1):
            M = N[:, j](u)
            assert np.all(M >= 0)
            for k in range(len(u)):
                np.testing.assert_almost_equal(np.sum(M[:, k]), 1)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_begin"])
def test_CreationRationalFunction():
    R = RationalFunction([0, 0, 1, 1])
    assert callable(R)
    assert R.degree == 1
    assert R.npts == 2
    R = RationalFunction([0, 0, 0.5, 1, 1])
    assert callable(R)
    assert R.degree == 1
    assert R.npts == 3
    R = RationalFunction([0, 0, 0, 1, 1, 1])
    assert callable(R)
    assert R.degree == 2
    assert R.npts == 3
    R = SplineFunction([0, 0, 0, 0.5, 1, 1, 1])
    assert callable(R)
    assert R.degree == 2
    assert R.npts == 4


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_CreationRationalFunction"])
def test_RationalEvaluationFunctions_p1n2():
    knotvector = [0, 0, 1, 1]  # degree = 1, npts = 2
    R = RationalFunction(knotvector)
    R[0, 0]
    R[1, 0]
    R[0, 1]
    R[1, 1]
    R[:, 0]
    R[:, 1]
    R[:]


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_RationalEvaluationFunctions_p1n2"])
def test_rational_somesinglevalues_p1n2():
    knotvector = [0, 0, 1, 1]  # degree = 1, npts = 2
    R = RationalFunction(knotvector)
    assert R[0, 0](0.0) == 0
    assert R[0, 0](0.5) == 0
    assert R[0, 0](1.0) == 0
    assert R[1, 0](0.0) == 1
    assert R[1, 0](0.5) == 1
    assert R[1, 0](1.0) == 1
    assert R[0, 1](0.0) == 1
    assert R[0, 1](0.5) == 0.5
    assert R[0, 1](1.0) == 0
    assert R[1, 1](0.0) == 0
    assert R[1, 1](0.5) == 0.5
    assert R[1, 1](1.0) == 1


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_rational_somesinglevalues_p1n2"])
def test_rational_tableUuniform_sum1():
    ntests = 10
    for i in range(ntests):
        degree = np.random.randint(1, 6)
        npts = np.random.randint(degree + 1, degree + 21)
        knotvector = GeneratorKnotVector.uniform(degree, npts)
        u = np.random.rand(11)
        R = RationalFunction(knotvector)
        for j in range(degree + 1):
            M = R[:, j](u)
            assert np.all(M >= 0)
            for k in range(len(u)):
                np.testing.assert_almost_equal(np.sum(M[:, k]), 1)


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(
    depends=["test_CreationSplineFunction", "test_CreationRationalFunction"]
)
def test_basefunction_fails():
    degree, npts = 4, 6
    knotvector = GeneratorKnotVector.uniform(degree, npts)
    N = SplineFunction(knotvector)
    with pytest.raises(ValueError):
        N(1.1)
    with pytest.raises(TypeError):
        N["asd"]
    with pytest.raises(TypeError):
        N["1"]
    with pytest.raises(TypeError):
        N[{1: 1}]
    with pytest.raises(IndexError):
        N[npts]
    with pytest.raises(IndexError):
        N[npts + 1]
    with pytest.raises(IndexError):
        N[3, degree + 1]
    with pytest.raises(IndexError):
        N[:, degree + 1]
    N[-1, degree]
    N[-npts, degree]
    with pytest.raises(IndexError):
        N[-npts - 1, degree]
    with pytest.raises(IndexError):
        N[-npts - 1, degree]
    with pytest.raises(TypeError):
        N[0, "1"]
    with pytest.raises(TypeError):
        N[0, "asd"]
    with pytest.raises(TypeError):
        N[0, {1: 1}]
    N[0, 0]
    with pytest.raises(IndexError):
        N[0, -1]
    with pytest.raises(IndexError):
        N[0, degree, 0]

    R = RationalFunction(knotvector)
    with pytest.raises(ValueError):
        w = np.linspace(-1, 1, npts)
        R.weights = w

    R1 = RationalFunction(knotvector)
    R1.weights = np.random.uniform(0.5, 1.5, npts)
    R2 = RationalFunction(knotvector)
    R2.weights = np.random.uniform(0.5, 1.5, npts)
    knotvector = GeneratorKnotVector.random(degree, npts)
    with pytest.raises(ValueError):
        R1.weights = "asd"
    with pytest.raises(ValueError):
        R1.weights = np.random.uniform(1, 2, npts - 1)
    with pytest.raises(ValueError):
        R1.weights = np.random.uniform(1, 2, npts + 1)
    with pytest.raises(ValueError):
        R1.weights = np.random.uniform(1, 2, size=(npts, 2))


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(
    depends=["test_CreationSplineFunction", "test_CreationRationalFunction"]
)
def test_comparation():
    Uorg = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]  # degree = 3, npts = 5
    Udif = [0, 0, 0, 0, 0.3, 1, 1, 1, 1]  # degree = 3, npts = 5
    worg = np.random.uniform(1, 2, 5)
    wdif = np.random.uniform(1, 2, 5)
    N1 = SplineFunction(Uorg)
    N2 = SplineFunction(Uorg)
    N3 = SplineFunction(Udif)
    R1 = RationalFunction(Uorg)
    R1.weights = worg
    R2 = RationalFunction(Uorg)
    R2.weights = worg
    R3 = RationalFunction(Uorg)
    R3.weights = wdif
    R4 = RationalFunction(Udif)
    R4.weights = wdif
    R5 = RationalFunction(Udif)
    R5.weights = worg

    assert N1 == N2
    assert R1 == R2
    assert N1 != N3
    assert R1 != R3
    assert R3 != R4
    assert R2 != R5
    assert R4 != R5

    N1.derivate()
    assert N1 != N2

    with pytest.raises(TypeError):
        assert N1 == "asd"
    with pytest.raises(TypeError):
        assert N1 == 1


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(depends=["test_comparation"])
def test_insert_remove_knot():
    Uorg = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]  # degree = 3, npts = 5
    weights = np.random.uniform(1, 2, 5)
    N1 = SplineFunction(Uorg)
    N2 = SplineFunction(Uorg)
    R1 = RationalFunction(Uorg)
    R1.weights = weights
    R2 = RationalFunction(Uorg)
    R2.weights = weights

    assert N1 == N2
    assert R1 == R2
    N1.knot_insert(0.5)
    R1.knot_insert(0.5)
    assert N1 != N2
    assert R1 != R2
    N1.knot_remove(0.5)
    R1.knot_remove(0.5)
    assert N1 == N2
    assert R1 == R2


@pytest.mark.order(3)
@pytest.mark.timeout(5)
@pytest.mark.dependency(
    depends=["test_CreationSplineFunction", "test_CreationRationalFunction"]
)
def test_derivate_functions():
    knotvector = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]  # degree = 3, npts = 5
    weights = np.random.uniform(1, 2, 5)
    N = SplineFunction(knotvector)
    R = RationalFunction(knotvector)
    R.weights = weights

    N.derivate()
    R.derivate()


@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_tableUuniform_sum1",
        "test_tableUrandom_sum1",
        "test_CreationRationalFunction",
        "test_RationalEvaluationFunctions_p1n2",
        "test_rational_somesinglevalues_p1n2",
        "test_rational_tableUuniform_sum1",
        "test_basefunction_fails",
        "test_insert_remove_knot",
    ]
)
def test_end():
    pass
