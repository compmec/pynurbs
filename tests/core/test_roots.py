import pytest

from pynurbs.core.polynomial import Polynomial
from pynurbs.core.roots import division, roots


@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=[
        "tests/core/test_polynomial.py::test_all",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_begin"])
def test_division():
    poly = Polynomial([0, 1])
    doly = Polynomial([1])

    qoly, roly = division(poly, doly)
    assert qoly == poly
    assert roly == 0

    qoly, roly = division(doly, poly)
    assert qoly == 0
    assert roly == 1

    all_numerators = (
        (3,),
        (9,),
        (0, 1),
        (0, -2),
        (-6, 8, -1, 2),
        (-2, 5, -3, 5, 5),
        (7, 3, -5, -8, 3),
        (7, -9, 2, 0, -2, -5),
        (-1, -9, 10),
    )
    polys = tuple(map(Polynomial, all_numerators))

    all_denominators = (
        (7, 10, -3),
        (2, -7, -10),
        (-9, -7, -3),
        (10, -6, 9, 7),
        (-1, -8, -10, 6),
        (-1, -10, 5, 9, 8),
        (9, -1, -2, 7, 5),
        (2, 6, -3, -7, -5),
        (0, 8, 10, 1, -4, -5),
        (5, -10, 4, 7, 2, -9),
    )
    dolys = tuple(map(Polynomial, all_denominators))

    for poly in polys:
        for doly in dolys:
            qoly, roly = division(poly, doly)
            diff = doly * qoly + roly - poly
            assert all(abs(coef) < 1e-9 for coef in diff)


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_begin"])
def test_roots():
    x = Polynomial([0, 1])
    values = roots(x**2 + 3 * x + 2)
    print(values)
    assert values == (-2, -1)
    values = roots(x**3 - 6 * x**2 + 11 * x - 6)
    print(values)
    assert values == (1, 2, 3)


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_division", "test_roots"])
def test_all():
    pass
