import pytest
import numpy as np
from compmec.nurbs import KnotVector
from compmec.nurbs.knotspace import getU_random, getU_uniform

@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
    pass


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_CreationClass():
    KnotVector([0, 0, 1, 1])
    KnotVector([0, 0, 0, 1, 1, 1])
    KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    KnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationClass"])
def test_FailCreationClass():
    with pytest.raises(TypeError):
        KnotVector(-1)
        KnotVector({1: 1})


    with pytest.raises(ValueError):
        KnotVector([0, 0, 0, 1, 1])
        KnotVector([0, 0, 1, 1, 1, ])
        KnotVector([0, 0, 0, 0, 1, 1, 1])
        KnotVector([0, 0, 0, 1, 1, 1, 1])
        KnotVector([-1, -1, 1, 1])
        KnotVector([0, 0, 2, 2])

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationClass"])
def test_ValuesOfP():
    V = KnotVector([0, 0, 1, 1])
    assert V.p == 1
    V = KnotVector([0, 0, 0, 1, 1, 1])
    assert V.p == 2
    V = KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    assert V.p == 3

    V = KnotVector([0, 0, 0.5, 1, 1])
    assert V.p == 1
    V = KnotVector([0, 0, 0.2, 0.6, 1, 1])
    assert V.p == 1
    V = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
    assert V.p == 2
    V = KnotVector([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert V.p == 2
    V = KnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    assert V.p == 3
    V = KnotVector([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert V.p == 3

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationClass"])
def test_ValuesOfN():
    V = KnotVector([0, 0, 1, 1])
    assert V.n == 2
    V = KnotVector([0, 0, 0, 1, 1, 1])
    assert V.n == 3
    V = KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    assert V.n == 4
    V = KnotVector([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert V.n == 5

    V = KnotVector([0, 0, 0.5, 1, 1])
    assert V.n == 3
    V = KnotVector([0, 0, 0.2, 0.6, 1, 1])
    assert V.n == 4
    V = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
    assert V.n == 4
    V = KnotVector([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert V.n == 5
    V = KnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    assert V.n == 5
    V = KnotVector([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert V.n == 6


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_ValuesOfP", "test_ValuesOfN"])
def test_findSpots_single():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1]) # p = 1, n =7
    assert U.spot(0) == 1
    assert U.spot(0.1) == 1
    assert U.spot(0.2) == 2
    assert U.spot(0.3) == 2
    assert U.spot(0.4) == 3
    assert U.spot(0.5) == 4
    assert U.spot(0.6) == 5
    assert U.spot(0.7) == 5
    assert U.spot(0.8) == 6
    assert U.spot(0.9) == 6
    assert U.spot(1.0) == 7

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_findSpots_single"])
def test_findSpots_array():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1]) # p = 1, n =7
    array = np.linspace(0, 1, 11)  # (0, 0.1, 0.2, ..., 0.9, 1.0)
    suposedspots = U.spot(array)
    correctspots = [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7]
    np.testing.assert_equal(suposedspots, correctspots)

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationClass"])
def test_generateUuniform():
    ntests = 100
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p+1, p+11)
        U = getU_uniform(n=n, p=p)
        assert isinstance(U, KnotVector)
        assert U.n == n
        assert U.p == p


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationClass"])
def test_generateUrandom():
    ntests = 100
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p+1, p+11)
        U = getU_random(n=n, p=p)
        assert isinstance(U, KnotVector)
        assert U.n == n
        assert U.p == p

@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_begin", "test_findSpots_array", "test_generateUuniform", "test_generateUrandom"])
def test_end():
    pass

def main():
    test_begin()
    test_CreationClass()
    test_FailCreationClass()
    test_ValuesOfP()
    test_ValuesOfN()
    test_findSpots_single()
    test_findSpots_array()
    test_end()

if __name__ == "__main__":
    main()