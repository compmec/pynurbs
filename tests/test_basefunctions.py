import pytest
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs.knotspace import getU_uniform, getU_random
import numpy as np


@pytest.mark.order(2)
@pytest.mark.dependency(
	depends=["tests/test_knotspace.py::test_end"],
    scope='session')
def test_begin():
    pass


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_CreationSplineBaseFunction():
    N = SplineBaseFunction([0, 0, 1, 1])
    assert callable(N)
    assert N.p == 1
    assert N.n == 2
    N = SplineBaseFunction([0, 0, 0.5, 1, 1])
    assert callable(N)
    assert N.p == 1
    assert N.n == 3
    N = SplineBaseFunction([0, 0, 0, 1, 1, 1])
    assert callable(N)
    assert N.p == 2
    assert N.n == 3
    N = SplineBaseFunction([0, 0, 0, 0.5, 1, 1, 1])
    assert callable(N)
    assert N.p == 2
    assert N.n == 4

    



@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationSplineBaseFunction"])
def test_SplineEvaluationFunctions_p1n2():
    U = [0, 0, 1, 1]  # p = 1, n = 2
    N = SplineBaseFunction(U)
    N[0, 0]
    N[1, 0]
    N[2, 0]
    N[0, 1]
    N[1, 1]
    N[2, 1]
    N[:, 0]
    N[:, 1]
    N[:]

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_SplineEvaluationFunctions_p1n2"])
def test_SplineEvaluationFunctions_p1n3():
    U = [0, 0, 0.5, 1, 1]  # p = 1, n = 3
    N = SplineBaseFunction(U)
    N[0, 0]
    N[1, 0]
    N[2, 0]
    N[3, 0]
    N[0, 1]
    N[1, 1]
    N[2, 1]
    N[3, 1]
    N[:, 0]
    N[:, 1]
    N[:]

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_SplineEvaluationFunctions_p1n2"])
def test_somesinglevalues_p1n2():
    U = [0, 0, 1, 1]  # p = 1, n = 2
    N = SplineBaseFunction(U)
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

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_somesinglevalues_p1n2"])
def test_somesinglevalues_p2n3():
    U = [0, 0, 0, 1, 1, 1]  # p = 2, n = 3
    N = SplineBaseFunction(U)
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

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_somesinglevalues_p1n2"])
def test_tablevalues_p1n2():
    U = [0, 0, 1, 1]  # p = 1, n = 2
    utest = np.linspace(0, 1, 11)
    N = SplineBaseFunction(U)
    N0 = N[:, 0]
    N1 = N[:, 1]
    M0test = N0(utest)
    M0good = np.array([[0]*11,[1]*11])
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array([np.linspace(1, 0, 11), np.linspace(0, 1, 11)])
    np.testing.assert_allclose(M1test, M1good)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tablevalues_p1n2"])
def test_tablevalues_p1n3():
    U = [0, 0, 0.5, 1, 1]  # p = 1, n = 3
    utest = np.linspace(0, 1, 11)
    N = SplineBaseFunction(U)
    N0 = N[:, 0]
    N1 = N[:, 1]
    M0test = N0(utest)
    M0good = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = [[1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    np.testing.assert_allclose(M1test, M1good)


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tablevalues_p1n2", "test_somesinglevalues_p2n3"])
def test_tablevalues_p2n3():
    U = [0, 0, 0, 1, 1, 1]  # p = 2, n = 3
    utest = np.linspace(0, 1, 11)
    N = SplineBaseFunction(U)
    N0 = N[:, 0]
    N1 = N[:, 1]
    N2 = N[:, 2]
    M0test = N0(utest)
    M0good = np.array([[0]*11,
                       [0]*11,
                       [1]*11])
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array([[0]*11,
                       np.linspace(1, 0, 11),
                       np.linspace(0, 1, 11)])
    np.testing.assert_allclose(M1test, M1good)
    M2test = N2(utest)
    M2good = np.array([[1.0, 0.81, 0.64, 0.49, 0.36, 0.25, 0.16, 0.09, 0.04, 0.01, 0.],
                       [0.,  0.18, 0.32, 0.42, 0.48, 0.50, 0.48, 0.42, 0.32, 0.18, 0.],
                       [0.,  0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.]])
    np.testing.assert_allclose(M2test, M2good)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tablevalues_p2n3"])
def test_tablevalues_p2n4():
    U = [0, 0, 0, 0.5, 1, 1, 1]  # p = 2, n = 4
    utest = np.linspace(0, 1, 11)
    N = SplineBaseFunction(U)
    N0 = N[:, 0]
    N1 = N[:, 1]
    N2 = N[:, 2]
    M0test = N0(utest)
    M0good = np.array([[0]*11,
                       [0]*11,
                       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array([[0]*11,
                       [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    np.testing.assert_allclose(M1test, M1good)
    M2test = N2(utest)
    M2good = np.array([[ 1, 0.64, 0.36, 0.16, 0.04, 0.0, 0.00, 0.00, 0.00, 0.00, 0],
                       [ 0, 0.34, 0.56, 0.66, 0.64, 0.5, 0.32, 0.18, 0.08, 0.02, 0],
                       [ 0, 0.02, 0.08, 0.18, 0.32, 0.5, 0.64, 0.66, 0.56, 0.34, 0],
                       [ 0, 0.00, 0.00, 0.00, 0.00, 0.0, 0.04, 0.16, 0.36, 0.64, 1]])
    np.testing.assert_allclose(M2test, M2good)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tablevalues_p2n3"])
def test_tablevalues_p3n4():
    U = [0, 0, 0, 0, 1, 1, 1, 1]  # p = 3, n = 4
    utest = np.linspace(0, 1, 11)
    N = SplineBaseFunction(U)
    N0 = N[:, 0]
    N1 = N[:, 1]
    N2 = N[:, 2]
    N3 = N[:, 3]
    M0test = N0(utest)
    M0good = np.array([[0]*11,
                       [0]*11,
                       [0]*11,
                       [1]*11])
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array([[0]*11,
                       [0]*11,
                       np.linspace(1, 0, 11),
                       np.linspace(0, 1, 11)])
    np.testing.assert_allclose(M1test, M1good)
    M2test = N2(utest)
    M2good = np.array([[0]*11,
                       [1.0, 0.81, 0.64, 0.49, 0.36, 0.25, 0.16, 0.09, 0.04, 0.01, 0.],
                       [0.,  0.18, 0.32, 0.42, 0.48, 0.50, 0.48, 0.42, 0.32, 0.18, 0.],
                       [0.,  0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.]])
    np.testing.assert_allclose(M2test, M2good)
    M3test = N3(utest)
    M3good = np.array([[ 1, 0.729, 0.512, 0.343, 0.216, 0.125, 0.064, 0.027, 0.008, 0.001, 0],
                       [ 0, 0.243, 0.384, 0.441, 0.432, 0.375, 0.288, 0.189, 0.096, 0.027, 0],
                       [ 0, 0.027, 0.096, 0.189, 0.288, 0.375, 0.432, 0.441, 0.384, 0.243, 0],
                       [ 0, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.512, 0.729, 1]])
    np.testing.assert_allclose(M3test, M3good)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tablevalues_p2n3", "test_tablevalues_p3n4"])
def test_tablevalues_p3n5():
    U = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]  # p = 3, n = 5
    utest = np.linspace(0, 1, 11)
    N = SplineBaseFunction(U)
    N0 = N[:, 0]
    N1 = N[:, 1]
    N2 = N[:, 2]
    N3 = N[:, 3]
    M0test = N0(utest)
    M0good = np.array([[0]*11,
                       [0]*11,
                       [0]*11,
                       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    np.testing.assert_allclose(M0test, M0good)
    M1test = N1(utest)
    M1good = np.array([[0]*11,
                       [0]*11,
                       [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    np.testing.assert_allclose(M1test, M1good)
    M2test = N2(utest)
    M2good = np.array([[0]*11,
                       [ 1, 0.64, 0.36, 0.16, 0.04, 0.0, 0.00, 0.00, 0.00, 0.00, 0],
                       [ 0, 0.34, 0.56, 0.66, 0.64, 0.5, 0.32, 0.18, 0.08, 0.02, 0],
                       [ 0, 0.02, 0.08, 0.18, 0.32, 0.5, 0.64, 0.66, 0.56, 0.34, 0],
                       [ 0, 0.00, 0.00, 0.00, 0.00, 0.0, 0.04, 0.16, 0.36, 0.64, 1]])
    np.testing.assert_allclose(M2test, M2good)
    M3test = N3(utest)
    M3good = np.array([[ 1, 0.512, 0.216, 0.064, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0],
                       [ 0, 0.434, 0.592, 0.558, 0.416, 0.250, 0.128, 0.054, 0.016, 0.002, 0],
                       [ 0, 0.052, 0.176, 0.324, 0.448, 0.500, 0.448, 0.324, 0.176, 0.052, 0],
                       [ 0, 0.002, 0.016, 0.054, 0.128, 0.250, 0.416, 0.558, 0.592, 0.434, 0],
                       [ 0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.008, 0.064, 0.216, 0.512, 1]])
    np.testing.assert_allclose(M3test, M3good)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tablevalues_p3n5"])
def test_tableUuniform_sum1():
    ntests = 10
    for i in range(ntests):
        p = np.random.randint(0, 6)
        n = np.random.randint(p+1, p+21)
        U = getU_uniform(n, p)
        u = np.random.rand(11)
        N = SplineBaseFunction(U)
        for j in range(p+1):
            M = N[:, j](u)
            assert np.all(M >= 0)
            for k in range(len(u)):
                np.testing.assert_almost_equal(np.sum(M[:, k]), 1)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tableUuniform_sum1"])
def test_tableUrandom_sum1():
    ntests = 10
    for i in range(ntests):
        p = np.random.randint(0, 6)
        n = np.random.randint(p+1, p+21)
        U = getU_random(n, p)
        u = np.random.rand(11)
        N = SplineBaseFunction(U)
        for j in range(p+1):
            M = N[:, j](u)
            assert np.all(M >= 0)
            for k in range(len(u)):
                np.testing.assert_almost_equal(np.sum(M[:, k]), 1)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationSplineBaseFunction"])
def test_comparetwo_splinebasefunctions():
    ntests = 10
    for i in range(ntests):
        p = np.random.randint(0, 6)
        n = np.random.randint(p+1, p+11)
        U = list(getU_random(n=n, p=p))
        N1 = SplineBaseFunction(U)
        N2 = SplineBaseFunction(U)
        assert N1 == N2


@pytest.mark.order(2)
@pytest.mark.dependency(depends=["test_begin", "test_tableUuniform_sum1",
    "test_tableUrandom_sum1", "test_comparetwo_splinebasefunctions"])
def test_end():
    pass

def main():
    test_SplineEvaluationFunctions_p1n2()
    test_SplineEvaluationFunctions_p1n3()
    test_somesinglevalues_p1n2()
    test_SplineEvaluationFunctions_p1n3()
    test_tablevalues_p1n2()
    test_tablevalues_p1n3()
    test_tablevalues_p2n3()
    test_tablevalues_p2n4()
    test_tablevalues_p3n4()
    test_tablevalues_p3n5()
    test_tableUuniform_sum1()
    test_tableUrandom_sum1()

if __name__ == "__main__":
    main()
