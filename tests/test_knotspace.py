from copy import copy
from fractions import Fraction

import numpy as np
import pytest

from pynurbs import GeneratorKnotVector, KnotVector


@pytest.mark.order(2)
@pytest.mark.dependency(depends=["tests/test_heavy.py::test_end"], scope="session")
def test_begin():
    pass


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_Creation():
    """
    Tests if creates a knotvector correctly
    """
    KnotVector([0, 1])
    KnotVector([0, 0, 1, 1])
    KnotVector([0, 0, 0, 1, 1, 1])
    KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    KnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])

    KnotVector([0, 0.5, 1])
    KnotVector([0, 0, 0.5, 0.5, 1, 1])
    KnotVector([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])

    KnotVector([0, 4])
    KnotVector([-4, 0])
    KnotVector([0, 0, 4, 4])
    KnotVector([-4, -4, 0, 0])

    KnotVector([0, 0, 0.25, 0.5, 0.75, 1, 1])
    KnotVector([0, 0, 0.5, 0.5, 1, 1])
    KnotVector([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    KnotVector([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_Creation"])
def test_FailCreation():
    """
    Test some invalid creation cases, which should raise error
    """
    with pytest.raises(ValueError):
        KnotVector(-1)
    with pytest.raises(ValueError):
        KnotVector({1: 1})
    with pytest.raises(ValueError):
        KnotVector(["asd", {1.1: 1}])

    with pytest.raises(ValueError):
        KnotVector([0, 0, 0, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([0, 0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([0, 0, 0.7, 0.2, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([[0, 0, 0.2, 0.7, 1, 1], [0, 0, 0.2, 0.7, 1, 1]])
    with pytest.raises(ValueError):
        KnotVector([[0, 0, 0.7, 0.2, 1, 1], [0, 0, 0.7, 0.2, 1, 1]])

    # Internal multiplicity error
    with pytest.raises(ValueError):
        KnotVector([0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1])
    with pytest.raises(ValueError):
        KnotVector([0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1])


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_Creation", "test_FailCreation"])
def test_ValuesDegree():
    V = KnotVector([0, 0, 1, 1])
    assert V.degree == 1
    V = KnotVector([0, 0, 0, 1, 1, 1])
    assert V.degree == 2
    V = KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    assert V.degree == 3

    V = KnotVector([0, 0, 0.5, 1, 1])
    assert V.degree == 1
    V = KnotVector([0, 0, 0.2, 0.6, 1, 1])
    assert V.degree == 1
    V = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
    assert V.degree == 2
    V = KnotVector([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert V.degree == 2
    V = KnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    assert V.degree == 3
    V = KnotVector([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert V.degree == 3


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_Creation", "test_FailCreation"])
def test_ValuesNumberPoints():
    V = KnotVector([0, 0, 1, 1])
    assert V.npts == 2
    V = KnotVector([0, 0, 0, 1, 1, 1])
    assert V.npts == 3
    V = KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    assert V.npts == 4
    V = KnotVector([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert V.npts == 5

    V = KnotVector([0, 0, 0.5, 1, 1])
    assert V.npts == 3
    V = KnotVector([0, 0, 0.2, 0.6, 1, 1])
    assert V.npts == 4
    V = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
    assert V.npts == 4
    V = KnotVector([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert V.npts == 5
    V = KnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    assert V.npts == 5
    V = KnotVector([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert V.npts == 6


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_ValuesDegree", "test_ValuesNumberPoints"])
def test_findspans_single():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])
    assert U.degree == 1
    assert U.npts == 7
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
    assert U.span(1.0) == 6

    with pytest.raises(ValueError):
        U.span(-0.1)  # Outside interval
    with pytest.raises(ValueError):
        U.span(1.1)  # Outside interval
    with pytest.raises(ValueError):
        U.span("asd")  # Not a number


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_ValuesDegree", "test_ValuesNumberPoints"])
def test_findmult_single():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])
    assert U.degree == 1
    assert U.npts == 7
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
        U.mult(-0.1)  # Outside interval
    with pytest.raises(ValueError):
        U.mult(1.1)  # Outside interval
    with pytest.raises(ValueError):
        U.mult("asd")  # Not a number


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_findspans_single"])
def test_findspans_array():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])
    array = np.linspace(0, 1, 11)  # (0, 0.1, 0.2, ..., 0.9, 1.0)
    suposedspans = U.span(array)
    correctspans = [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 6]
    assert U.degree == 1
    assert U.npts == 7
    np.testing.assert_equal(suposedspans, correctspans)


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_findmult_single"])
def test_findmult_array():
    U = KnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])
    array = np.linspace(0, 1, 11)  # (0, 0.1, 0.2, ..., 0.9, 1.0)
    suposedmults = U.mult(array)
    correctmults = [2, 0, 1, 0, 1, 1, 1, 0, 1, 0, 2]
    assert U.degree == 1
    assert U.npts == 7
    np.testing.assert_equal(suposedmults, correctmults)


@pytest.mark.order(2)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["test_ValuesDegree", "test_ValuesNumberPoints"])
def test_CompareKnotvector():
    U1 = KnotVector([0, 0, 1, 1])
    U2 = KnotVector([0, 0, 1, 1])
    assert U1 == U2
    assert U1 == [0, 0, 1, 1]

    U3 = KnotVector([0, 0, 0.5, 1, 1])
    assert U1 != U3

    assert U1 != 0
    assert U1 != "asad"


class TestOperations:
    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["test_CompareKnotvector"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(depends=["TestOperations::test_begin"])
    def test_scale(self):
        U1 = KnotVector([0, 0, 1, 1])
        U2 = KnotVector([0, 0, 2, 2])
        U3 = KnotVector([1, 1, 3, 3])
        U4 = KnotVector([2, 2, 6, 6])

        U = copy(U1)
        assert U == U1
        assert U != U2
        U.scale(2)
        assert U != U1
        assert U == U2
        U.scale(1 / 2)
        assert U == U1
        assert U != U2
        U *= 2
        assert U != U1
        assert U == U2
        U *= 1 / 2
        assert U == U1
        assert U != U2
        U /= 1 / 2  # times 2
        assert U != U1
        assert U == U2
        U /= 2
        assert U == U1
        assert U != U2

        U = copy(U3)
        assert U == U3
        assert U != U2
        U.scale(2)
        assert U != U3
        assert U == U4
        U.scale(1 / 2)
        assert U == U3
        assert U != U4
        U *= 2
        assert U != U3
        assert U == U4
        U *= 1 / 2
        assert U == U3
        assert U != U4
        U /= 1 / 2  # times 2
        assert U != U3
        assert U == U4
        U /= 2
        assert U == U3
        assert U != U4

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(depends=["TestOperations::test_begin"])
    def test_shift(self):
        U1 = KnotVector([0, 0, 1, 1])
        U2 = KnotVector([1, 1, 2, 2])
        U = copy(U1)
        assert U == U1
        assert U != U2
        U.shift(1)  # Shift all vector
        assert U == U2

        U = copy(U1)
        U += 1
        assert U == U2

        U = copy(U2)
        assert U == U2
        assert U != U1
        U.shift(-1)
        assert U == U1

        U = copy(U2)
        U -= 1
        assert U == U1

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(depends=["TestOperations::test_begin"])
    def test_normalize(self):
        knotvector = KnotVector([1.0, 1.0, 2.0, 3.0, 3.0])
        knotvector.normalize()
        assert knotvector == [0, 0, 0.5, 1, 1]

        knotvector = KnotVector([2, 2, 3, 4, 4])
        knotvector.normalize()
        assert knotvector == [0, 0, 0.5, 1, 1]

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(depends=["TestOperations::test_begin"])
    def test_convert(self):
        knotvector = KnotVector([1.0, 1.0, 2.0, 3.0, 3.0])
        knotvector.convert(int)
        for knot in knotvector:
            assert isinstance(knot, int)
        knotvector.convert(float)
        for knot in knotvector:
            assert isinstance(knot, float)
        knotvector.convert(Fraction)
        for knot in knotvector:
            assert isinstance(knot, Fraction)

        knotvector = KnotVector([0, 0, 0.5, 1, 1])
        knotvector.convert(Fraction)
        for knot in knotvector:
            assert isinstance(knot, Fraction)
        assert knotvector == [0, 0, 0.5, 1, 1]

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(
        depends=[
            "TestOperations::test_begin",
            "TestOperations::test_scale",
            "TestOperations::test_shift",
            "TestOperations::test_normalize",
            "TestOperations::test_convert",
        ]
    )
    def test_fails(self):
        knotvector = KnotVector([0.0, 0.0, 0.5, 1.0, 1.0])
        with pytest.raises(ValueError):
            knotvector.convert(int)

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(
        depends=[
            "TestOperations::test_begin",
            "TestOperations::test_scale",
            "TestOperations::test_shift",
            "TestOperations::test_normalize",
            "TestOperations::test_convert",
            "TestOperations::test_fails",
        ]
    )
    def test_end(self):
        pass


class TestGenerator:
    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=["test_CompareKnotvector", "TestOperations::test_end"]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestGenerator::test_begin"])
    def test_bezier(self):
        Ugood = KnotVector([0, 0, 1, 1])
        assert Ugood.degree == 1
        assert Ugood.npts == 2
        Utest = GeneratorKnotVector.bezier(1)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood
        for knot in Utest:
            assert isinstance(knot, int)

        Ugood = KnotVector([0, 0, 0, 1, 1, 1])
        assert Ugood.degree == 2
        assert Ugood.npts == 3
        Utest = GeneratorKnotVector.bezier(2)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood
        for knot in Utest:
            assert isinstance(knot, int)

        Ugood = KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
        assert Ugood.degree == 3
        assert Ugood.npts == 4
        Utest = GeneratorKnotVector.bezier(3)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood
        for knot in Utest:
            assert isinstance(knot, int)

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=["TestGenerator::test_begin", "TestGenerator::test_bezier"]
    )
    def test_integer(self):
        Ugood = KnotVector([0, 0, 1, 1])
        assert Ugood.degree == 1
        assert Ugood.npts == 2
        Utest = GeneratorKnotVector.integer(1, 2)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood
        for knot in Utest:
            assert isinstance(knot, int)

        Ugood = KnotVector([0, 0, 1, 2, 2])
        assert Ugood.degree == 1
        assert Ugood.npts == 3
        Utest = GeneratorKnotVector.integer(1, 3)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood
        for knot in Utest:
            assert isinstance(knot, int)

        Ugood = KnotVector([0, 0, 1, 2, 3, 4, 4])
        assert Ugood.degree == 1
        assert Ugood.npts == 5
        Utest = GeneratorKnotVector.integer(1, 5)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood
        for knot in Utest:
            assert isinstance(knot, int)

        ntests = 100
        for i in range(ntests):
            degree = np.random.randint(1, 6)
            npts = np.random.randint(degree + 1, degree + 11)
            Utest = GeneratorKnotVector.integer(degree, npts)
            assert isinstance(Utest, KnotVector)
            assert Utest.npts == npts
            assert Utest.degree == degree
            for knot in Utest:
                assert isinstance(knot, int)

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestGenerator::test_begin",
            "TestGenerator::test_bezier",
            "TestGenerator::test_integer",
        ]
    )
    def test_uniform(self):
        Ugood = KnotVector([0, 0, 1, 1])
        assert Ugood.degree == 1
        assert Ugood.npts == 2
        Utest = GeneratorKnotVector.uniform(1, 2)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood

        Ugood = KnotVector([0, 0, 0.5, 1, 1])
        assert Ugood.degree == 1
        assert Ugood.npts == 3
        Utest = GeneratorKnotVector.uniform(1, 3)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood

        Ugood = KnotVector([0, 0, 0.25, 0.5, 0.75, 1, 1])
        assert Ugood.degree == 1
        assert Ugood.npts == 5
        Utest = GeneratorKnotVector.uniform(1, 5)
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood

        ntests = 100
        for i in range(ntests):
            degree = np.random.randint(1, 6)
            npts = np.random.randint(degree + 1, degree + 11)
            Utest = GeneratorKnotVector.uniform(degree, npts)
            assert isinstance(Utest, KnotVector)
            assert Utest.npts == npts
            assert Utest.degree == degree

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(
        depends=[
            "TestGenerator::test_begin",
            "TestGenerator::test_bezier",
            "TestGenerator::test_integer",
            "TestGenerator::test_uniform",
        ]
    )
    def test_random(self):
        ntests = 1000
        for i in range(ntests):
            degree = np.random.randint(1, 6)
            npts = np.random.randint(degree + 1, degree + 11)
            knotvect = GeneratorKnotVector.random(degree, npts)
            assert isinstance(knotvect, KnotVector)
            assert knotvect.npts == npts
            assert knotvect.degree == degree

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(
        depends=[
            "TestGenerator::test_begin",
            "TestGenerator::test_bezier",
            "TestGenerator::test_integer",
            "TestGenerator::test_uniform",
            "TestGenerator::test_random",
        ]
    )
    def test_weighted(self):
        Ugood = KnotVector([0, 0, 1, 1])
        assert Ugood.degree == 1
        assert Ugood.npts == 2
        Utest = GeneratorKnotVector.weight(1, [1])
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood

        Ugood = KnotVector([0, 0, 2, 2])
        assert Ugood.degree == 1
        assert Ugood.npts == 2
        Utest = GeneratorKnotVector.weight(1, [2])
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood

        Ugood = KnotVector([0, 0, 2, 3, 3])
        assert Ugood.degree == 1
        assert Ugood.npts == 3
        Utest = GeneratorKnotVector.weight(1, [2, 1])
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood

        Ugood = KnotVector([0, 0, 0, 1, 2, 3, 3, 3])
        assert Ugood.degree == 2
        assert Ugood.npts == 5
        Utest = GeneratorKnotVector.weight(2, [1, 1, 1])
        assert isinstance(Utest, KnotVector)
        assert Utest == Ugood

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestGenerator::test_begin",
            "TestGenerator::test_bezier",
            "TestGenerator::test_integer",
            "TestGenerator::test_uniform",
            "TestGenerator::test_random",
            "TestGenerator::test_weighted",
        ]
    )
    def test_clstype(self):
        cls_types = [int, float, Fraction]
        # bezier
        for degree in range(0, 7):
            for cls in cls_types:
                knotvector = GeneratorKnotVector.bezier(degree, cls)
                for knot in knotvector:
                    assert isinstance(knot, cls)
        # integers
        for degree in range(0, 7):
            for npts in range(degree + 1, degree + 11):
                for cls in cls_types:
                    knotvector = GeneratorKnotVector.integer(degree, npts, cls)
                    for knot in knotvector:
                        assert isinstance(knot, cls)
        # weight
        for degree in range(0, 7):
            for npts in range(degree + 1, degree + 11):
                for cls in cls_types:
                    vector = np.random.randint(1, 5, npts - degree)
                    vector = [cls(number) for number in vector]
                    knotvector = GeneratorKnotVector.weight(degree, vector)
                    for knot in knotvector:
                        assert isinstance(knot, cls)
        cls_types = [float, Fraction]
        # uniform
        for degree in range(0, 7):
            for npts in range(degree + 1, degree + 11):
                for cls in cls_types:
                    knotvector = GeneratorKnotVector.uniform(degree, npts, cls)
                    for knot in knotvector:
                        assert isinstance(knot, cls)
        # random
        for degree in range(0, 7):
            for npts in range(degree + 1, degree + 11):
                for cls in cls_types:
                    knotvector = GeneratorKnotVector.random(degree, npts, cls)
                    for knot in knotvector:
                        assert isinstance(knot, cls)

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestGenerator::test_begin",
            "TestGenerator::test_bezier",
            "TestGenerator::test_integer",
            "TestGenerator::test_uniform",
            "TestGenerator::test_random",
            "TestGenerator::test_weighted",
            "TestGenerator::test_clstype",
        ]
    )
    def test_fails(self):
        with pytest.raises(AssertionError):
            GeneratorKnotVector.bezier(degree=-1)
        for degree in range(1, 6):
            with pytest.raises(AssertionError):
                GeneratorKnotVector.uniform(degree, npts=degree)
            with pytest.raises(AssertionError):
                GeneratorKnotVector.uniform(degree, npts=degree - 1)
            with pytest.raises(AssertionError):
                GeneratorKnotVector.random(degree, npts=degree)
            with pytest.raises(AssertionError):
                GeneratorKnotVector.random(degree, npts=degree - 1)

        with pytest.raises(AssertionError):
            GeneratorKnotVector.bezier(degree="asd")
        with pytest.raises(AssertionError):
            GeneratorKnotVector.bezier(degree={1: 1})
        with pytest.raises(AssertionError):
            GeneratorKnotVector.uniform(degree=2, npts=3.0)
        with pytest.raises(AssertionError):
            GeneratorKnotVector.uniform(degree=2.0, npts=3)
        with pytest.raises(AssertionError):
            GeneratorKnotVector.uniform(degree=2, npts="3")
        with pytest.raises(AssertionError):
            GeneratorKnotVector.uniform(degree="2", npts=3)
        with pytest.raises(AssertionError):
            GeneratorKnotVector.random(degree=2, npts=3.0)
        with pytest.raises(AssertionError):
            GeneratorKnotVector.random(degree=2.0, npts=3)

    @pytest.mark.order(2)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(
        depends=[
            "TestGenerator::test_begin",
            "TestGenerator::test_bezier",
            "TestGenerator::test_integer",
            "TestGenerator::test_uniform",
            "TestGenerator::test_random",
            "TestGenerator::test_weighted",
            "TestGenerator::test_clstype",
            "TestGenerator::test_fails",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(2)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["TestGenerator::test_end"])
def test_compare_knotvectors_fail():
    degree = np.random.randint(1, 6)
    npts = np.random.randint(degree + 3, degree + 11)
    U1 = GeneratorKnotVector.uniform(degree, npts)
    assert not (U1 == 1)
    assert U1 != 1
    assert U1 != "asd"
    assert U1 != [[0, 0, 0, 0.5, 1, 1, 1]]
    U2 = GeneratorKnotVector.uniform(degree + 1, npts + 1)
    U3 = GeneratorKnotVector.uniform(degree + 1, npts + 2)
    U4 = GeneratorKnotVector.uniform(degree, npts + 1)
    U5 = GeneratorKnotVector.random(degree, npts)
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


@pytest.mark.order(2)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["TestGenerator::test_end"])
def test_insert_knot_remove():
    Uinc0 = [0, 0, 0, 1, 1, 1]
    Uinc1 = [0, 0, 0, 0.5, 1, 1, 1]
    Uinc2 = [0, 0, 0, 0.5, 0.5, 1, 1, 1]
    Uinc3 = [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]
    Uinc4 = [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1]

    U0 = KnotVector(Uinc0)
    U1 = KnotVector(Uinc1)
    U2 = KnotVector(Uinc2)
    U3 = KnotVector(Uinc3)
    with pytest.raises(ValueError):
        KnotVector(Uinc4)

    Uo = KnotVector(Uinc0)
    assert Uo == U0
    Uo += []  #  Knot insert
    assert Uo == U0
    Uo += [0.5]
    assert Uo == U1
    Uo += [0.5]
    assert Uo == U2
    Uo += [0.5]
    assert Uo == U3
    with pytest.raises(ValueError):
        Uo += [0.5]

    Uo = KnotVector(Uinc0)
    Uo += [0.5, 0.5]
    assert Uo == U2
    Uo += []
    with pytest.raises(ValueError):
        # Can insert only once, not twice
        Uo += [0.5, 0.5]

    Uo = KnotVector(Uinc0)
    Uo += [0.5, 0.5, 0.5]
    assert Uo == U3

    Uo = KnotVector(Uinc3)
    assert Uo == U3
    Uo -= [0.5]
    assert Uo == U2
    Uo -= [0.5]
    assert Uo == U1
    Uo -= [0.5]
    assert Uo == U0
    with pytest.raises(ValueError):
        Uo -= [0.5]
    with pytest.raises(ValueError):
        Uo -= [0.5]
    with pytest.raises(ValueError):
        # Cannot remove 0 times
        Uo -= [0.5, 0.5]

    Uo = KnotVector(Uinc3)
    assert Uo == U3
    Uo -= [0.5, 0.5]
    assert Uo == U1

    Uo = KnotVector(Uinc3)
    assert Uo == U3
    Uo -= [0.5, 0.5, 0.5]
    assert Uo == U0

    Uo = KnotVector(Uinc3)
    with pytest.raises(ValueError):
        U0 -= ["asd"]
    with pytest.raises(ValueError):
        U0 -= [-0.5]
    with pytest.raises(ValueError):
        U0 -= [0.25]
    with pytest.raises(ValueError):
        U0 -= [0.5] * 4

    with pytest.raises(ValueError):
        U0 -= [0]  # Take out one extremity


@pytest.mark.order(2)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["TestGenerator::test_end"])
def test_degree_change():
    U = KnotVector([0, 0, 0, 1, 1, 1])
    assert U == [0, 0, 0, 1, 1, 1]
    assert U.degree == 2
    assert U.npts == 3
    U.degree = 1
    assert U == [0, 0, 1, 1]
    U.degree += 1
    assert U == [0, 0, 0, 1, 1, 1]
    U.degree -= 1
    assert U == [0, 0, 1, 1]
    U.degree = 4
    assert U == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    U = KnotVector([0, 0, 0, 1, 1, 2, 2, 2])
    assert U == [0, 0, 0, 1, 1, 2, 2, 2]
    assert U.degree == 2
    assert U.npts == 5
    U.degree = 1
    assert U == [0, 0, 1, 2, 2]
    U.degree += 1
    assert U == [0, 0, 0, 1, 1, 2, 2, 2]
    U.degree -= 1
    assert U == [0, 0, 1, 2, 2]
    U.degree = 4
    assert U == [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    U.degree = 1
    assert U == [0, 0, 1, 2, 2]
    U.degree = 0
    assert U == [0, 2]
    U.degree = 1
    assert U == [0, 0, 2, 2]
    U.degree = 3
    assert U == [0, 0, 0, 0, 2, 2, 2, 2]


@pytest.mark.order(2)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["TestGenerator::test_end"])
def test_or_and():
    U1 = KnotVector([0, 0, 0, 1, 1, 1])
    U2 = KnotVector([0, 0, 1, 1])
    assert U1 | U2 == U1
    assert U2 | U1 == U1
    assert U1 & U2 == U2
    assert U2 & U1 == U2

    U1 = KnotVector([0, 0, 0.2, 1, 1])
    U2 = KnotVector([0, 0, 0.8, 1, 1])
    assert U1 | U2 == [0, 0, 0.2, 0.8, 1, 1]
    assert U2 | U1 == [0, 0, 0.2, 0.8, 1, 1]
    assert U1 & U2 == [0, 0, 1, 1]
    assert U2 & U1 == [0, 0, 1, 1]

    U1 = KnotVector([0, 0, 0, 1, 1, 2, 2, 2])
    U2 = KnotVector([0, 0, 1, 2, 2, 3, 3])
    with pytest.raises(ValueError):
        U1 | U2
    with pytest.raises(ValueError):
        U1 & U2

    U1 = KnotVector([0, 0, 0, 1, 1, 2, 2, 2])
    U2 = KnotVector([0, 0, 0, 1, 2, 2, 3, 3, 3])
    with pytest.raises(ValueError):
        U1 | U2
    with pytest.raises(ValueError):
        U1 & U2


@pytest.mark.order(2)
@pytest.mark.timeout(4)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestGenerator::test_end",
    ]
)
def test_others():
    knotvect = [0, 0.2, 0.4, 0.4, 0.8, 1]
    with pytest.raises(ValueError):
        KnotVector(knotvect)
    knotvect = [0, 0, 0.5, 1, 1]
    knotvect = KnotVector(knotvect)
    knotvect = KnotVector(knotvect)

    newvect = knotvect + 1
    newvect = knotvect - 1
    newvect = knotvect * 2
    newvect = knotvect / 2
    newvect = 2 * knotvect

    np.testing.assert_allclose(knotvect.knots, [0, 0.5, 1])

    str(knotvect)
    knotvect.__repr__()


@pytest.mark.order(2)
@pytest.mark.timeout(4)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_Creation",
        "test_FailCreation",
    ]
)
def test_fractions():
    from fractions import Fraction as frac

    knotvect = [frac(0), frac(1, 5), frac(2, 5), frac(3, 5), frac(4, 5), frac(1)]
    knotvect = KnotVector(knotvect)
    assert knotvect.degree == 0
    assert knotvect.npts == 5
    for i, knot in enumerate(knotvect):
        assert type(knot) is frac
    assert knotvect[0] == 0
    assert float(knotvect[1]) == 0.2
    assert float(knotvect[2]) == 0.4
    assert float(knotvect[3]) == 0.6
    assert float(knotvect[4]) == 0.8
    assert knotvect[5] == 1

    knotvect = GeneratorKnotVector.uniform(1, 5, frac)
    assert knotvect.degree == 1
    assert knotvect.npts == 5
    for knot in knotvect:
        assert type(knot) is frac


@pytest.mark.order(2)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_Creation",
        "test_FailCreation",
        "test_ValuesDegree",
        "test_ValuesNumberPoints",
        "test_findspans_single",
        "test_findmult_single",
        "test_findspans_array",
        "test_findmult_array",
        "TestGenerator::test_end",
        "test_compare_knotvectors_fail",
        "test_insert_knot_remove",
        "test_degree_change",
        "test_or_and",
        "test_others",
        "test_fractions",
    ]
)
def test_end():
    pass
