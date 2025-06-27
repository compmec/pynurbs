import numpy as np
import pytest

from pynurbs.core import ImmutableKnotVector


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
    pass


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_Creation():
    """
    Tests if creates a ImmutableKnotVector correctly
    """
    ImmutableKnotVector([0, 1])
    ImmutableKnotVector([0, 0, 1, 1])
    ImmutableKnotVector([0, 0, 0, 1, 1, 1])
    ImmutableKnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    ImmutableKnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])

    ImmutableKnotVector([0, 0.5, 1])
    ImmutableKnotVector([0, 0, 0.5, 0.5, 1, 1])
    ImmutableKnotVector([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])

    ImmutableKnotVector([0, 4])
    ImmutableKnotVector([-4, 0])
    ImmutableKnotVector([0, 0, 4, 4])
    ImmutableKnotVector([-4, -4, 0, 0])

    ImmutableKnotVector([0, 0, 0.25, 0.5, 0.75, 1, 1])
    ImmutableKnotVector([0, 0, 0.5, 0.5, 1, 1])
    ImmutableKnotVector([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    ImmutableKnotVector([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_Creation"])
def test_FailCreation():
    """
    Test some invalid creation cases, which should raise error
    """
    with pytest.raises(ValueError):
        ImmutableKnotVector(-1)
    with pytest.raises(ValueError):
        ImmutableKnotVector({1: 1})
    with pytest.raises(ValueError):
        ImmutableKnotVector(["asd", {1.1: 1}])

    with pytest.raises(ValueError):
        ImmutableKnotVector([0, 0, 0, 1, 1])
    with pytest.raises(ValueError):
        ImmutableKnotVector([0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        ImmutableKnotVector([0, 0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        ImmutableKnotVector([0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        ImmutableKnotVector([0, 0, 0.7, 0.2, 1, 1])
    with pytest.raises(ValueError):
        ImmutableKnotVector([[0, 0, 0.2, 0.7, 1, 1], [0, 0, 0.2, 0.7, 1, 1]])
    with pytest.raises(ValueError):
        ImmutableKnotVector([[0, 0, 0.7, 0.2, 1, 1], [0, 0, 0.7, 0.2, 1, 1]])

    # Internal multiplicity error
    with pytest.raises(ValueError):
        ImmutableKnotVector([0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1])
    with pytest.raises(ValueError):
        ImmutableKnotVector([0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1])


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_Creation", "test_FailCreation"])
def test_ValuesDegree():
    V = ImmutableKnotVector([0, 0, 1, 1])
    assert V.degree == 1
    V = ImmutableKnotVector([0, 0, 0, 1, 1, 1])
    assert V.degree == 2
    V = ImmutableKnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    assert V.degree == 3

    V = ImmutableKnotVector([0, 0, 0.5, 1, 1])
    assert V.degree == 1
    V = ImmutableKnotVector([0, 0, 0.2, 0.6, 1, 1])
    assert V.degree == 1
    V = ImmutableKnotVector([0, 0, 0, 0.5, 1, 1, 1])
    assert V.degree == 2
    V = ImmutableKnotVector([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert V.degree == 2
    V = ImmutableKnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    assert V.degree == 3
    V = ImmutableKnotVector([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert V.degree == 3


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_Creation", "test_FailCreation"])
def test_ValuesNumberPoints():
    V = ImmutableKnotVector([0, 0, 1, 1])
    assert V.npts == 2
    V = ImmutableKnotVector([0, 0, 0, 1, 1, 1])
    assert V.npts == 3
    V = ImmutableKnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    assert V.npts == 4
    V = ImmutableKnotVector([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert V.npts == 5

    V = ImmutableKnotVector([0, 0, 0.5, 1, 1])
    assert V.npts == 3
    V = ImmutableKnotVector([0, 0, 0.2, 0.6, 1, 1])
    assert V.npts == 4
    V = ImmutableKnotVector([0, 0, 0, 0.5, 1, 1, 1])
    assert V.npts == 4
    V = ImmutableKnotVector([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert V.npts == 5
    V = ImmutableKnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    assert V.npts == 5
    V = ImmutableKnotVector([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert V.npts == 6


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_ValuesDegree", "test_ValuesNumberPoints"])
def test_findspans_single():
    U = ImmutableKnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])
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


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_ValuesDegree", "test_ValuesNumberPoints"])
def test_findmult_single():
    U = ImmutableKnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])
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


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_findspans_single"])
def test_findspans_array():
    U = ImmutableKnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])
    array = np.linspace(0, 1, 11)  # (0, 0.1, 0.2, ..., 0.9, 1.0)
    suposedspans = U.span(array)
    correctspans = [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 6]
    assert U.degree == 1
    assert U.npts == 7
    np.testing.assert_equal(suposedspans, correctspans)


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_findmult_single"])
def test_findmult_array():
    U = ImmutableKnotVector([0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1])
    array = np.linspace(0, 1, 11)  # (0, 0.1, 0.2, ..., 0.9, 1.0)
    suposedmults = U.mult(array)
    correctmults = [2, 0, 1, 0, 1, 1, 1, 0, 1, 0, 2]
    assert U.degree == 1
    assert U.npts == 7
    np.testing.assert_equal(suposedmults, correctmults)


@pytest.mark.order(1)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["test_ValuesDegree", "test_ValuesNumberPoints"])
def test_CompareImmutableKnotVector():
    U1 = ImmutableKnotVector([0, 0, 1, 1])
    U2 = ImmutableKnotVector([0, 0, 1, 1])
    assert U1 == U2
    assert U1 == (0, 0, 1, 1)

    U3 = ImmutableKnotVector([0, 0, 0.5, 1, 1])
    assert U1 != U3

    assert U1 != 0
    assert U1 != "asad"


@pytest.mark.order(1)
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
        "test_compare_ImmutableKnotVectors_fail",
        "test_insert_knot_remove",
        "test_degree_change",
        "test_or_and",
        "test_others",
        "test_fractions",
    ]
)
def test_end():
    pass
