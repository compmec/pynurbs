from fractions import Fraction
from numbers import Integral, Real
from typing import Tuple

import numpy as np
import pytest

from pynurbs.core.piecepoly import PiecewisePolynomial, Polynomial, find_span


def get_random_knots(
    start: Real, end: Real, nsegs: int, cls: type = float
) -> Tuple[Real, ...]:
    """
    Computes the (nsegs+1) knots that are in the interval [start, end]
    These knots are randomly distributed in
    """
    nodes = np.cumsum(np.random.randint(1, 17, nsegs))
    if cls is int:
        cls = Fraction
    nodes = [cls(0)] + list(map(cls, nodes))
    nodes = [start + node * (end - start) / nodes[-1] for node in nodes]
    return tuple(nodes)


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=[
        "tests/core/test_knotvector.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_begin"])
def test_find_span():
    knots = [0, 1, 2, 3, 4]
    assert find_span(0, knots) == 0
    assert find_span(1, knots) == 1
    assert find_span(2, knots) == 2
    assert find_span(3, knots) == 3
    assert find_span(4, knots) == 3

    assert find_span(0.5, knots) == 0
    assert find_span(1.5, knots) == 1
    assert find_span(2.5, knots) == 2
    assert find_span(3.5, knots) == 3


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_begin"])
def test_build():
    x = Polynomial([0, 1])
    polys = [x, 1 - x, x * x, 3 - x * x * x]
    PiecewisePolynomial(polys, range(1 + len(polys)))


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build", "test_find_span"])
def test_evaluate():
    x = Polynomial([0, 1])
    polys = [x, 1 - x, x * x, 3 - x * x * x]
    piece = PiecewisePolynomial(polys, range(1 + len(polys)))

    assert piece(0) == 0
    assert piece(0.5) == 0.5
    assert piece(1) == 0
    assert piece(1.5) == -0.5
    assert piece(2) == 4
    assert piece(2.5) == 6.25
    assert piece(3) == -24
    assert piece(4) == -61


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build"])
def test_compare():
    x = Polynomial([0, 1])
    polysa = [x, 1 - x, x * x, 3 - x * x * x]
    polysb = [-x, x - 1, -x * x, x * x * x - 3]
    piecea = PiecewisePolynomial(polysa, range(1 + len(polysa)))
    pieceb = PiecewisePolynomial(polysb, range(1 + len(polysb)))

    assert piecea == piecea
    assert pieceb == pieceb
    assert piecea != pieceb


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build", "test_evaluate"])
def test_add():
    nsegs, degree = 6, 4

    knotsa = get_random_knots(0, 1, nsegs)
    coefsa = np.random.randint(-10, 11, (nsegs, degree + 1))
    piecea = PiecewisePolynomial(map(Polynomial, coefsa), knotsa)

    knotsb = get_random_knots(0, 1, nsegs)
    coefsb = np.random.randint(-10, 11, (nsegs, degree + 1))
    pieceb = PiecewisePolynomial(map(Polynomial, coefsb), knotsb)

    piecec = piecea + pieceb

    for x in np.linspace(0, 1, 129):
        assert abs(piecea(x) + pieceb(x) - piecec(x)) < 1e-9


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build", "test_evaluate"])
def test_neg():
    nsegs, degree = 6, 4
    knots = get_random_knots(0, 10, nsegs)
    coefs = np.random.randint(-10, 11, (nsegs, degree + 1))
    piecea = PiecewisePolynomial(map(Polynomial, coefs), knots)
    pieceb = -piecea

    for x in np.linspace(knots[0], knots[-1], 129):
        assert pieceb(x) == -piecea(x)


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build", "test_evaluate", "test_add", "test_neg"])
def test_sub():
    nsegs, degree = 6, 4

    knotsa = get_random_knots(0, 1, nsegs)
    coefsa = np.random.randint(-10, 11, (nsegs, degree + 1))
    piecea = PiecewisePolynomial(map(Polynomial, coefsa), knotsa)

    knotsb = get_random_knots(0, 1, nsegs)
    coefsb = np.random.randint(-10, 11, (nsegs, degree + 1))
    pieceb = PiecewisePolynomial(map(Polynomial, coefsb), knotsb)

    piecec = piecea - pieceb
    pieced = piecea + (-pieceb)

    for x in np.linspace(0, 1, 129):
        assert abs(piecea(x) - pieceb(x) - piecec(x)) < 1e-9
        assert abs(piecea(x) - pieceb(x) - pieced(x)) < 1e-9


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=["test_build", "test_evaluate", "test_neg", "test_add", "test_sub"]
)
def test_mul():
    nsegs, degree = 6, 4

    knotsa = get_random_knots(0, 1, nsegs)
    coefsa = np.random.randint(-10, 11, (nsegs, degree + 1))
    piecea = PiecewisePolynomial(map(Polynomial, coefsa), knotsa)

    knotsb = get_random_knots(0, 1, nsegs)
    coefsb = np.random.randint(-10, 11, (nsegs, degree + 1))
    pieceb = PiecewisePolynomial(map(Polynomial, coefsb), knotsb)

    piecec = piecea * pieceb

    for x in np.linspace(0, 1, 129):
        assert abs(piecea(x) * pieceb(x) - piecec(x)) < 1e-9


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=["test_build", "test_neg", "test_add", "test_sub", "test_mul"]
)
def test_matmul():
    nsegs, degree = 6, 4

    knotsa = get_random_knots(0, 1, nsegs)
    coefsa = np.random.randint(-10, 11, (nsegs, degree + 1, 3))
    piecea = PiecewisePolynomial(map(Polynomial, coefsa), knotsa)

    knotsb = get_random_knots(0, 1, nsegs)
    coefsb = np.random.randint(-10, 11, (nsegs, degree + 1, 3))
    pieceb = PiecewisePolynomial(map(Polynomial, coefsb), knotsb)

    piecec = piecea @ pieceb

    for x in np.linspace(0, 1, 129):
        assert abs(piecea(x) @ pieceb(x) - piecec(x)) < 1e-9


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=[
        "test_build",
        "test_neg",
        "test_add",
        "test_sub",
        "test_mul",
        "test_matmul",
    ]
)
def test_scalar_operation():
    nsegs, degree = 6, 4

    knotsa = get_random_knots(0, 1, nsegs)
    nodes = np.linspace(0, 1, 129)

    for _ in range(10):  # number of tests
        const = np.random.randint(-10, 11)
        coefsa = np.random.randint(-10, 11, (nsegs, degree + 1))
        piecea = PiecewisePolynomial(map(Polynomial, coefsa), knotsa)

        pieceb = piecea + const
        piecec = const + piecea
        for node in nodes:
            assert abs(piecea(node) + const - pieceb(node)) < 1e-9
            assert abs(const + piecea(node) - piecec(node)) < 1e-9

        pieceb = piecea - const
        piecec = const - piecea
        for node in nodes:
            assert abs(piecea(node) - const - pieceb(node)) < 1e-9
            assert abs(const - piecea(node) - piecec(node)) < 1e-9

        pieceb = piecea * const
        piecec = const * piecea
        for node in nodes:
            assert abs(piecea(node) * const - pieceb(node)) < 1e-9
            assert abs(const * piecea(node) - piecec(node)) < 1e-9

    ndim = 3
    for _ in range(10):  # number of tests
        const = np.random.randint(-10, 11, (ndim,))
        coefsa = np.random.randint(-10, 11, (nsegs, degree + 1, ndim))
        piecea = PiecewisePolynomial(map(Polynomial, coefsa), knotsa)

        pieceb = piecea @ const
        for node in nodes:
            assert abs(piecea(node) @ const - pieceb(node)) < 1e-9


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_build",
        "test_evaluate",
        "test_neg",
        "test_add",
        "test_sub",
        "test_mul",
        "test_matmul",
        "test_scalar_operation",
    ]
)
def test_all():
    pass
