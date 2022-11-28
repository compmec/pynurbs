import numpy as np
import pytest

from compmec.nurbs import KnotVector
from compmec.nurbs.knotspace import GeneratorKnotVector


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
    with pytest.raises(TypeError):
        KnotVector({1: 1})
    with pytest.raises(TypeError):
        KnotVector(["asd", {1.1: 1}])

    with pytest.raises(ValueError):
        KnotVector([0, 0, 0, 1, 1])
    with pytest.raises(ValueError):
        U = [0, 0, 1, 1, 1]
        KnotVector(U)
    with pytest.raises(ValueError):
        KnotVector([0, 0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([-1, -1, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([0, 0, 2, 2])
    with pytest.raises(ValueError):
        KnotVector([0, 0, 0.7, 0.2, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([[0, 0, 0.7, 0.2, 1, 1], [0, 0, 0.7, 0.2, 1, 1]])
    with pytest.raises(ValueError):
        KnotVector([[0, 0, 0.7, 0.2, 1, 1], [0, 0, 0.7, 0.2, 1, 1]])

    KnotVector([0, 0, 0, 0.5, 0.5, 1, 1, 1])
    KnotVector([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])
    with pytest.raises(ValueError):
        U = [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1]
        KnotVector(U)
    with pytest.raises(ValueError):
        U = [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]
        KnotVector(U)


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationClass", "test_FailCreationClass"])
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
def test_findspans_single():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])  # p = 1, n =7
    assert U.span(0) == 1
    assert U.span(0.1) == 1
    assert U.span(0.2) == 2
    assert U.span(0.3) == 2
    assert U.span(0.4) == 3
    assert U.span(0.5) == 4
    assert U.span(0.6) == 5
    assert U.span(0.7) == 5
    assert U.span(0.8) == 6
    assert U.span(0.9) == 6
    assert U.span(1.0) == 7

    with pytest.raises(ValueError):
        U.span(-0.1)
    with pytest.raises(ValueError):
        U.span(1.1)
    with pytest.raises(TypeError):
        U.span("asd")


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_ValuesOfP", "test_ValuesOfN"])
def test_findmult_single():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])  # p = 1, n =7
    assert U.mult(0) == 2
    assert U.mult(0.1) == 0
    assert U.mult(0.2) == 1
    assert U.mult(0.3) == 0
    assert U.mult(0.4) == 1
    assert U.mult(0.5) == 1
    assert U.mult(0.6) == 1
    assert U.mult(0.7) == 0
    assert U.mult(0.8) == 1
    assert U.mult(0.9) == 0
    assert U.mult(1.0) == 2

    with pytest.raises(ValueError):
        U.mult(-0.1)
    with pytest.raises(ValueError):
        U.mult(1.1)
    with pytest.raises(TypeError):
        U.mult("asd")


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_findspans_single"])
def test_findspans_array():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])  # p = 1, n =7
    array = np.linspace(0, 1, 11)  # (0, 0.1, 0.2, ..., 0.9, 1.0)
    suposedspans = U.span(array)
    correctspans = [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7]
    np.testing.assert_equal(suposedspans, correctspans)


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_findmult_single"])
def test_findmults_array():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])  # p = 1, n =7
    array = np.linspace(0, 1, 11)  # (0, 0.1, 0.2, ..., 0.9, 1.0)
    suposedmults = U.mult(array)
    correctmults = [2, 0, 1, 0, 1, 1, 1, 0, 1, 0, 2]
    np.testing.assert_equal(suposedmults, correctmults)


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationClass"])
def test_generateUbezier():
    for p in range(0, 9):
        U = GeneratorKnotVector.bezier(p=p)
        assert isinstance(U, KnotVector)
        assert U.n == p + 1
        assert U.p == p


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_CreationClass"])
def test_generateUuniform():
    ntests = 100
    for i in range(ntests):
        p = np.random.randint(1, 6)
        n = np.random.randint(p + 1, p + 11)
        U = GeneratorKnotVector.uniform(n=n, p=p)
        assert isinstance(U, KnotVector)
        assert U.n == n
        assert U.p == p


@pytest.mark.order(1)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["test_CreationClass"])
def test_generateUrandom():
    ntests = 1000
    for i in range(ntests):
        p = np.random.randint(0, 6)
        n = np.random.randint(p + 1, p + 11)
        U = GeneratorKnotVector.random(n=n, p=p)
        assert isinstance(U, KnotVector)
        assert U.n == n
        assert U.p == p


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=["test_generateUbezier", "test_generateUuniform", "test_generateUrandom"]
)
def test_generatorUfails():
    with pytest.raises(ValueError):
        GeneratorKnotVector.bezier(p=-1)
    for p in range(1, 6):
        with pytest.raises(ValueError):
            GeneratorKnotVector.uniform(n=p, p=p)
        with pytest.raises(ValueError):
            GeneratorKnotVector.uniform(n=p - 1, p=p)
        with pytest.raises(ValueError):
            GeneratorKnotVector.random(n=p, p=p)
        with pytest.raises(ValueError):
            GeneratorKnotVector.random(n=p - 1, p=p)

    with pytest.raises(TypeError):
        GeneratorKnotVector.bezier(p="asd")
    with pytest.raises(TypeError):
        GeneratorKnotVector.bezier(p={1: 1})
    with pytest.raises(TypeError):
        GeneratorKnotVector.uniform(n=3.0, p=2)
    with pytest.raises(TypeError):
        GeneratorKnotVector.uniform(n=3, p=2.0)
    with pytest.raises(TypeError):
        GeneratorKnotVector.uniform(n="3", p=2)
    with pytest.raises(TypeError):
        GeneratorKnotVector.uniform(n=3, p="2")
    with pytest.raises(TypeError):
        GeneratorKnotVector.random(n=3.0, p=2)
    with pytest.raises(TypeError):
        GeneratorKnotVector.random(n=3, p=2.0)


@pytest.mark.order(1)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["test_generateUuniform"])
def test_comparetwo_knotvectors():
    ntests = 10
    for i in range(ntests):
        p = np.random.randint(0, 6)
        n = np.random.randint(p + 1, p + 11)
        U1 = GeneratorKnotVector.uniform(n=n, p=p)
        U2 = GeneratorKnotVector.uniform(n=n, p=p)
        assert U1 == U2


@pytest.mark.order(1)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["test_generateUuniform"])
def test_compare_knotvectors_fail():
    p = np.random.randint(0, 6)
    n = np.random.randint(p + 3, p + 11)
    U1 = GeneratorKnotVector.uniform(p=p, n=n)
    with pytest.raises(TypeError):
        assert U1 == 1
    with pytest.raises(TypeError):
        assert U1 == "asd"
    with pytest.raises(Exception):
        assert U1 == [[0, 0, 0, 0.5, 1, 1, 1]]
    U2 = GeneratorKnotVector.uniform(p=p + 1, n=n + 1)
    U3 = GeneratorKnotVector.uniform(p=p + 1, n=n + 2)
    U4 = GeneratorKnotVector.uniform(p=p, n=n + 1)
    U5 = GeneratorKnotVector.random(p=p, n=n)
    assert U1 != U2
    assert U1 != U3
    assert U1 != U4
    assert U1 != U5
    assert U2 != U3
    assert U2 != U4
    assert U2 != U5
    assert U3 != U4
    assert U3 != U5
    assert U4 != U5


@pytest.mark.order(1)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["test_generateUuniform"])
def test_insert_knot_remove():
    Uorg = [0, 0, 0, 0, 1, 1, 1, 1]
    Uinc0 = [0, 0, 0, 0, 1, 1, 1, 1]
    Uinc1 = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
    Uinc2 = [0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1]
    Uinc3 = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]
    Uinc4 = [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]
    Uinc5 = [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]

    U0 = KnotVector(Uinc0)
    U1 = KnotVector(Uinc1)
    U2 = KnotVector(Uinc2)
    U3 = KnotVector(Uinc3)
    U4 = KnotVector(Uinc4)
    with pytest.raises(ValueError):
        KnotVector(Uinc5)

    Uo = KnotVector(Uorg)
    assert Uo == U0
    Uo.knot_insert(0.5)
    assert Uo == U1
    Uo.knot_insert(0.5)
    assert Uo == U2
    Uo.knot_insert(0.5)
    assert Uo == U3
    Uo.knot_insert(0.5)
    assert Uo == U4

    Uo = KnotVector(Uorg)
    Uo.knot_insert(0.5, 2)
    assert Uo == U2

    Uo = KnotVector(Uorg)
    Uo.knot_insert(0.5, 3)
    assert Uo == U3

    Uo = KnotVector(Uorg)
    Uo.knot_insert(0.5, 4)
    assert Uo == U4

    Uo = KnotVector(Uinc4)
    Uo.knot_remove(0.5)
    assert Uo == U3
    Uo.knot_remove(0.5)
    assert Uo == U2
    Uo.knot_remove(0.5)
    assert Uo == U1
    Uo.knot_remove(0.5)
    assert Uo == U0
    with pytest.raises(ValueError):
        Uo.knot_remove(0.5)

    Uo = KnotVector(Uinc4)
    Uo.knot_remove(0.5, 2)
    assert Uo == U2

    Uo = KnotVector(Uinc4)
    Uo.knot_remove(0.5, 3)
    assert Uo == U1

    Uo = KnotVector(Uinc4)
    Uo.knot_remove(0.5, 4)
    assert Uo == U0

    Uo = KnotVector(Uinc4)
    with pytest.raises(TypeError):
        U0.knot_remove("asd")
    with pytest.raises(TypeError):
        U0.knot_remove(0.5, "ads")
    with pytest.raises(ValueError):
        U0.knot_remove(0.5, 0)
    with pytest.raises(ValueError):
        U0.knot_remove(0.5, -1)
    with pytest.raises(ValueError):
        U0.knot_remove(-0.5)


@pytest.mark.order(1)
@pytest.mark.timeout(4)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_findspans_array",
        "test_findmults_array",
        "test_generateUuniform",
        "test_generateUrandom",
        "test_generatorUfails",
        "test_comparetwo_knotvectors",
        "test_compare_knotvectors_fail",
        "test_insert_knot_remove",
    ]
)
def test_end():
    pass
