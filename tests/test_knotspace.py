import pytest
import numpy as np
from compmec.nurbs import KnotVector


def test_CreationClass():
    KnotVector([0, 0, 1, 1])
    KnotVector([0, 0, 0, 1, 1, 1])
    KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    KnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])


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



def test_findSpots():
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
    array = np.linspace(0, 1, 11)  # (0, 0.1, 0.2, ..., 0.9, 1.0)
    suposedspots = U.spot(array)
    correctspots = [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7]
    np.testing.assert_equal(suposedspots, correctspots)

def main():
    test_CreationClass()
    test_FailCreationClass()
    test_ValuesOfP()
    test_ValuesOfN()

if __name__ == "__main__":
    main()