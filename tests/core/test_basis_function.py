from fractions import Fraction

import numpy as np
import pytest

from pynurbs.core.basisfunction import ImmutableBasisFunction
from pynurbs.core.knotvector import ImmutableKnotVector


def binom(n: int, i: int):
    """
    Returns binomial (n, i)
    """
    assert isinstance(n, int)
    assert isinstance(i, int)
    prod = 1
    if i <= 0 or i >= n:
        return 1
    for j in range(i):
        prod *= (n - j) / (i - j)
    return int(prod)


@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=[
        "tests/core/test_knotvector.py::test_end",
        "tests/core/test_polynomial.py::test_all",
    ],
    scope="session",
)
def test_begin():
    pass


class TestBezier:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestBezier::test_begin"])
    def test_creation(self):
        knotvector = ImmutableKnotVector([0, 0, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert callable(bezier)
        assert bezier.degree == 1
        assert bezier.npts == 2

        knotvector = ImmutableKnotVector([0, 0, 0, 1, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert callable(bezier)
        assert bezier.degree == 2
        assert bezier.npts == 3

        knotvector = ImmutableKnotVector([0, 0, 0, 0, 1, 1, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert callable(bezier)
        assert bezier.degree == 3
        assert bezier.npts == 4

        for degree in range(0, 8):
            npts = degree + 1
            knotvector = ImmutableKnotVector([0] * npts + [1] * npts)
            bezier = ImmutableBasisFunction(knotvector)
            assert callable(bezier)
            assert bezier.degree == degree
            assert bezier.npts == npts

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_creation"])
    def test_sum_equal_to_1(self):
        knotvector = ImmutableKnotVector([0, 0, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert sum(bezier(0.00)) == 1
        assert sum(bezier(0.25)) == 1
        assert sum(bezier(0.50)) == 1
        assert sum(bezier(0.75)) == 1
        assert sum(bezier(1.00)) == 1

        knotvector = ImmutableKnotVector([0, 0, 0, 1, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert sum(bezier(0.00)) == 1
        assert sum(bezier(0.25)) == 1
        assert sum(bezier(0.50)) == 1
        assert sum(bezier(0.75)) == 1
        assert sum(bezier(1.00)) == 1

        knotvector = ImmutableKnotVector([0, 0, 0, 0, 1, 1, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert sum(bezier(0.00)) == 1
        assert sum(bezier(0.25)) == 1
        assert sum(bezier(0.50)) == 1
        assert sum(bezier(0.75)) == 1
        assert sum(bezier(1.00)) == 1

        divisions = 32
        for degree in range(0, 6):
            npts = degree + 1
            knotvector = ImmutableKnotVector([0] * npts + [1] * npts)
            bezier = ImmutableBasisFunction(knotvector)
            assert bezier.degree == degree
            assert bezier.npts == npts

            for i in range(divisions + 1):
                results = bezier(i / divisions)
                assert len(results) == npts
                assert all(result >= 0 for result in results)
                assert sum(results) == 1

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_creation",
            "TestBezier::test_sum_equal_to_1",
        ]
    )
    def test_single_values(self):
        # degree = 1, npts = 2
        knotvector = ImmutableKnotVector([0, 0, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert bezier(0) == (1, 0)
        assert bezier(0.5) == (0.5, 0.5)
        assert bezier(1) == (0, 1)

        # degree = 2, npts = 3
        knotvector = ImmutableKnotVector([0, 0, 0, 1, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert bezier(0) == (1, 0, 0)
        assert bezier(0.5) == (0.25, 0.5, 0.25)
        assert bezier(1) == (0, 0, 1)

        # degree = 3, npts = 3
        knotvector = ImmutableKnotVector([0, 0, 0, 0, 1, 1, 1, 1])
        bezier = ImmutableBasisFunction(knotvector)
        assert bezier(0) == (1, 0, 0, 0)
        assert bezier(0.25) == (27 / 64, 27 / 64, 9 / 64, 1 / 64)
        assert bezier(0.5) == (1 / 8, 3 / 8, 3 / 8, 1 / 8)
        assert bezier(0.75) == (1 / 64, 9 / 64, 27 / 64, 27 / 64)
        assert bezier(1) == (0, 0, 0, 1)

        divisions = 8
        for degree in range(0, 6):
            npts = degree + 1
            knotvector = [Fraction(0)] * npts + [Fraction(1)] * npts
            knotvector = ImmutableKnotVector(knotvector)
            bezier = ImmutableBasisFunction(knotvector)
            for j in range(divisions + 1):
                node = Fraction(j, divisions)
                minu = 1 - node
                goods = (
                    binom(degree, i) * minu ** (degree - i) * node**i
                    for i in range(degree + 1)
                )
                assert bezier(node) == tuple(goods)

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_begin",
            "TestBezier::test_creation",
            "TestBezier::test_sum_equal_to_1",
            "TestBezier::test_single_values",
        ]
    )
    def test_all(self):
        pass


class TestSpline:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestBezier::test_all"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestSpline::test_begin"])
    def test_creation(self):
        knotvector = ImmutableKnotVector([0, 0, 1, 1])
        spline = ImmutableBasisFunction(knotvector)
        assert callable(spline)
        assert spline.degree == 1
        assert spline.npts == 2

        knotvector = ImmutableKnotVector([0, 0, 0.5, 1, 1])
        spline = ImmutableBasisFunction(knotvector)
        assert callable(spline)
        assert spline.degree == 1
        assert spline.npts == 3

        knotvector = ImmutableKnotVector([0, 0, 0, 1, 1, 1])
        spline = ImmutableBasisFunction(knotvector)
        assert callable(spline)
        assert spline.degree == 2
        assert spline.npts == 3

        knotvector = ImmutableKnotVector([0, 0, 0, 0.5, 1, 1, 1])
        spline = ImmutableBasisFunction(knotvector)
        assert callable(spline)
        assert spline.degree == 2
        assert spline.npts == 4

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestSpline::test_creation"])
    def test_tablevalues_degree1npts3(self):
        knotvector = ImmutableKnotVector([0, 0, 0.5, 1, 1])
        spline = ImmutableBasisFunction(knotvector)
        assert spline.degree == 1
        assert spline.npts == 3

        nodes_test = np.linspace(0, 1, 11)

        matrix_good = [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.6, 0.4, 0.0],
            [0.4, 0.6, 0.0],
            [0.2, 0.8, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.6, 0.4],
            [0.0, 0.4, 0.6],
            [0.0, 0.2, 0.8],
            [0.0, 0.0, 1.0],
        ]
        for node, good in zip(nodes_test, matrix_good):
            np.testing.assert_allclose(spline(node), good)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestSpline::test_tablevalues_degree1npts3"])
    def test_tablevalues_degree2npts4(self):
        knotvector = [0, 0, 0, 0.5, 1, 1, 1]
        knotvector = ImmutableKnotVector(knotvector)
        spline = ImmutableBasisFunction(knotvector)
        assert spline.degree == 2
        assert spline.npts == 4
        nodes_test = np.linspace(0, 1, 11)

        matrix_good = [
            [1.0, 0.0, 0.0, 0.0],
            [0.64, 0.34, 0.02, 0.0],
            [0.36, 0.56, 0.08, 0.0],
            [0.16, 0.66, 0.18, 0.0],
            [0.04, 0.64, 0.32, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.32, 0.64, 0.04],
            [0.0, 0.18, 0.66, 0.16],
            [0.0, 0.08, 0.56, 0.36],
            [0.0, 0.02, 0.34, 0.64],
            [0.0, 0.0, 0.0, 1.0],
        ]

        for node, good in zip(nodes_test, matrix_good):
            np.testing.assert_allclose(spline(node), good)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestSpline::test_tablevalues_degree2npts4"])
    def test_tablevalues_degree3npts5(self):
        knotvector = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
        knotvector = ImmutableKnotVector(knotvector)
        spline = ImmutableBasisFunction(knotvector)
        assert spline.degree == 3
        assert spline.npts == 5
        nodes_test = np.linspace(0, 1, 11)

        matrix_good = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.512, 0.434, 0.052, 0.002, 0.0],
            [0.216, 0.592, 0.176, 0.016, 0.0],
            [0.064, 0.558, 0.324, 0.054, 0.0],
            [0.008, 0.416, 0.448, 0.128, 0.0],
            [0.0, 0.25, 0.5, 0.25, 0.0],
            [0.0, 0.128, 0.448, 0.416, 0.008],
            [0.0, 0.054, 0.324, 0.558, 0.064],
            [0.0, 0.016, 0.176, 0.592, 0.216],
            [0.0, 0.002, 0.052, 0.434, 0.512],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        for node, good in zip(nodes_test, matrix_good):
            np.testing.assert_allclose(spline(node), good)

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestSpline::test_begin",
            "TestSpline::test_creation",
            "TestSpline::test_tablevalues_degree1npts3",
            "TestSpline::test_tablevalues_degree2npts4",
            "TestSpline::test_tablevalues_degree3npts5",
        ]
    )
    def test_all(self):
        pass


@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestBezier::test_all",
        "TestSpline::test_all",
    ]
)
def test_all():
    pass
