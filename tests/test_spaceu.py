import pytest
from compmec.nurbs import VectorU


def test_CreationClass():
    VectorU([0, 0, 1, 1])
    VectorU([0, 0, 0, 1, 1, 1])
    VectorU([0, 0, 0, 0, 1, 1, 1, 1])
    VectorU([0, 0, 0, 0, 0.5, 1, 1, 1, 1])


def test_FailCreationClass():
    with pytest.raises(TypeError):
        VectorU(-1)
        VectorU({1: 1})


    with pytest.raises(ValueError):
        VectorU([0, 0, 0, 1, 1])
        VectorU([0, 0, 1, 1, 1, ])
        VectorU([0, 0, 0, 0, 1, 1, 1])
        VectorU([0, 0, 0, 1, 1, 1, 1])
        VectorU([-1, -1, 1, 1])
        VectorU([0, 0, 2, 2])

def test_ValuesOfP():
    V = VectorU([0, 0, 1, 1])
    assert V.p == 1
    V = VectorU([0, 0, 0, 1, 1, 1])
    assert V.p == 2
    V = VectorU([0, 0, 0, 0, 1, 1, 1, 1])
    assert V.p == 3

    V = VectorU([0, 0, 0.5, 1, 1])
    assert V.p == 1
    V = VectorU([0, 0, 0.2, 0.6, 1, 1])
    assert V.p == 1
    V = VectorU([0, 0, 0, 0.5, 1, 1, 1])
    assert V.p == 2
    V = VectorU([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert V.p == 2
    V = VectorU([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    assert V.p == 3
    V = VectorU([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert V.p == 3

def test_ValuesOfN():
    V = VectorU([0, 0, 1, 1])
    assert V.n == 2
    V = VectorU([0, 0, 0, 1, 1, 1])
    assert V.n == 3
    V = VectorU([0, 0, 0, 0, 1, 1, 1, 1])
    assert V.n == 4
    V = VectorU([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert V.n == 5

    V = VectorU([0, 0, 0.5, 1, 1])
    assert V.n == 3
    V = VectorU([0, 0, 0.2, 0.6, 1, 1])
    assert V.n == 4
    V = VectorU([0, 0, 0, 0.5, 1, 1, 1])
    assert V.n == 4
    V = VectorU([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert V.n == 5
    V = VectorU([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    assert V.n == 5
    V = VectorU([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert V.n == 6

def main():
    test_CreationClass()
    test_FailCreationClass()
    test_ValuesOfP()
    test_ValuesOfN()

if __name__ == "__main__":
    main()