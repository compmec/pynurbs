import pytest

from pynurbs.core.knotvector import ImmutableKnotVector
from pynurbs.operations.knotvector import (
    decrease_degree,
    increase_degree,
    insert_knots,
    remove_knots,
)


@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=[
        "tests/core/test_knotvector.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_begin"])
def test_insert_knots():
    knotvector = ImmutableKnotVector([0, 0, 1, 2, 2])
    assert insert_knots(knotvector, [0.5, 1.5]) == (0, 0, 0.5, 1, 1.5, 2, 2)


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_begin"])
def test_remove_knots():
    knotvector = ImmutableKnotVector([0, 0, 1, 2, 2])
    assert remove_knots(knotvector, [1]) == (0, 0, 2, 2)


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_begin"])
def test_increase_degree():
    knotvector = ImmutableKnotVector([0, 0, 1, 2, 2])
    assert increase_degree(knotvector, 1) == (0, 0, 0, 1, 1, 2, 2, 2)
    assert increase_degree(knotvector, 2) == (0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2)


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_begin"])
def test_decrease_degree():
    knotvector = ImmutableKnotVector([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    assert decrease_degree(knotvector, 1) == (0, 0, 0, 1, 2, 2, 2)
    assert decrease_degree(knotvector, 2) == (0, 0, 2, 2)


@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_insert_knots",
        "test_remove_knots",
        "test_increase_degree",
        "test_decrease_degree",
    ]
)
def test_all():
    pass
