import math
from fractions import Fraction

import numpy as np
import pytest

from pynurbs.heavy import IntegratorArray, LeastSquare, Linalg, Math, NodeSample


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
    pass


class TestMath:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestMath::test_begin"])
    def test_gcd(self):
        assert Math.gcd(0) == 0
        assert Math.gcd(1) == 1
        assert Math.gcd(*[0] * 5) == 0
        assert Math.gcd(*[1] * 5) == 1
        assert Math.gcd(0, 1) == 1
        assert Math.gcd(0, 2) == 2
        assert Math.gcd(1, 1000) == 1
        assert Math.gcd(10, 1000) == 10
        assert Math.gcd(100, 1000) == 100
        assert Math.gcd(1000, 1000) == 1000
        assert Math.gcd(2, 3, 4) == 1
        assert Math.gcd(6, 9, 12) == 3

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestMath::test_begin", "TestMath::test_gcd"])
    def test_lcm(self):
        assert Math.lcm(0) == 0
        assert Math.lcm(1) == 1
        assert Math.lcm(*[0] * 5) == 0
        assert Math.lcm(*[1] * 5) == 1
        assert Math.lcm(0, 1) == 1
        assert Math.lcm(0, 2) == 2
        assert Math.lcm(1, 1000) == 1000
        assert Math.lcm(10, 1000) == 1000
        assert Math.lcm(100, 1000) == 1000
        assert Math.lcm(1000, 1000) == 1000
        assert Math.lcm(10000, 1000) == 10000
        assert Math.lcm(2, 3, 4) == 12
        assert Math.lcm(6, 9, 12) == 36

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestMath::test_begin"])
    def test_comb(self):
        assert Math.comb(1, 0) == 1
        assert Math.comb(1, 1) == 1
        assert Math.comb(2, 0) == 1
        assert Math.comb(2, 1) == 2
        assert Math.comb(2, 2) == 1
        assert Math.comb(3, 0) == 1
        assert Math.comb(3, 1) == 3
        assert Math.comb(3, 2) == 3
        assert Math.comb(3, 3) == 1

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestMath::test_begin",
            "TestMath::test_gcd",
            "TestMath::test_lcm",
            "TestMath::test_comb",
        ]
    )
    def test_end(self):
        pass


class TestLinalg:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin", "TestMath::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestLinalg::test_begin"])
    def test_invert_float(self):
        identit = np.eye(4)
        inverse = Linalg.invert(identit)
        np.testing.assert_allclose(inverse, identit)

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=["TestLinalg::test_begin", "TestLinalg::test_invert_float"]
    )
    def test_invert_integer(self):
        matrix = ((1, 0), (0, 1))
        good = ((1, 0), (0, 1))
        test = Linalg.invert(matrix)
        test = np.array(test, dtype="int16")
        np.testing.assert_allclose(test, good)

        matrix = ((1, 1), (2, 3))
        good = ((3, -1), (-2, 1))
        test = Linalg.invert(matrix)
        test = np.array(test, dtype="int16")
        np.testing.assert_allclose(test, good)

        matrix = ((1, 1), (11, 12))
        good = ((12, -1), (-11, 1))
        test = Linalg.invert(matrix)
        test = np.array(test, dtype="int16")
        np.testing.assert_allclose(test, good)

        matrix = (
            (1, 1, 1, 1),
            (11, 12, 13, 14),
            (55, 66, 78, 91),
            (165, 220, 286, 364),
        )
        good = (
            (364, -78, 12, -1),
            (-1001, 221, -35, 3),
            (924, -209, 34, -3),
            (-286, 66, -11, 1),
        )
        test = Linalg.invert(matrix)
        test = np.array(test, dtype="int64")
        np.testing.assert_allclose(test, good)

        # Ericksen matrix
        for side in range(1, 10):
            matrix = np.zeros((side, side), dtype="int64")
            for n in range(side, side + 10):
                for i in range(side):
                    for j in range(side):
                        matrix[i, j] = Math.comb(n + j, i)
            inverse = Linalg.invert(matrix)
            inverse = np.array(inverse, dtype="int64")
            np.testing.assert_allclose(np.dot(inverse, matrix), np.eye(side))
            np.testing.assert_allclose(np.dot(matrix, inverse), np.eye(side))

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestLinalg::test_begin",
            "TestLinalg::test_invert_float",
            "TestLinalg::test_invert_integer",
        ]
    )
    def test_invert_fraction(self):
        # Identity
        for side in range(1, 10):
            zero, one = Fraction(0), Fraction(1)
            matrix = [
                [one if i == j else zero for j in range(side)] for i in range(side)
            ]
            test = Linalg.invert(matrix)
            test = np.array(test, dtype="int64")
            good = np.eye(side, dtype="int64")
            np.testing.assert_allclose(test, good)

        # Ericksen matrix
        for side in range(1, 10):
            matrix = np.zeros((side, side), dtype="object")
            for n in range(side, side + 10):
                for i in range(side):
                    for j in range(side):
                        matrix[i, j] = Fraction(Math.comb(n + j, i))
            inverse = Linalg.invert(matrix)
            inverse = np.array(inverse, dtype="int64")
            matrix = np.array(matrix, dtype="int64")
            np.testing.assert_allclose(np.dot(inverse, matrix), np.eye(side))
            np.testing.assert_allclose(np.dot(matrix, inverse), np.eye(side))

        matrix = [[3, 4], [1, 2]]
        inverse = Linalg.invert(matrix)
        testident = np.dot(inverse, matrix)
        testident = testident.astype("float64")
        np.testing.assert_allclose(testident, np.eye(len(matrix)))

        matrix = [[1, 2], [2, 3]]
        inverse = Linalg.invert(matrix)
        testident = np.dot(inverse, matrix)
        testident = testident.astype("float64")
        np.testing.assert_allclose(testident, np.eye(len(matrix)))

        frac = Fraction
        matrix = [[frac(1), frac(-2)], [frac(-1, 2), frac(3, 2)]]
        inverse = Linalg.invert(matrix)
        testident = np.dot(inverse, matrix)
        testident = testident.astype("float64")
        np.testing.assert_allclose(testident, np.eye(len(matrix)))

        size = 3
        while True:
            floatmatrix = np.random.uniform(-1, 1, (size, size))
            fracmatrix = np.empty((size, size), dtype="object")
            for i, line in enumerate(floatmatrix):
                for j, elem in enumerate(line):
                    fracmatrix[i, j] = Fraction(elem).limit_denominator(10)
            fracmatrix += np.transpose(fracmatrix)
            if abs(np.linalg.det(np.array(fracmatrix, dtype="float64"))) > 1e-6:
                break
        invfracmatrix = Linalg.invert(fracmatrix)
        product = fracmatrix @ invfracmatrix
        product = np.array(product, dtype="float64")
        np.testing.assert_allclose(product, np.eye(size))

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=["TestLinalg::test_begin", "TestLinalg::test_invert_fraction"]
    )
    def test_solve_float(self):
        side, nsols = 4, 4
        matrix = np.eye(side)
        force = np.random.rand(side, nsols)
        solution = Linalg.solve(matrix, force)
        np.testing.assert_allclose(np.dot(matrix, solution), force)

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=["TestLinalg::test_begin", "TestLinalg::test_solve_float"]
    )
    def test_solve_integer(self):
        side, nsols = 2, 4
        force = [[np.random.randint(-5, 6) for j in range(nsols)] for i in range(side)]
        matrix = ((1, 0), (0, 1))
        solution = Linalg.solve(matrix, force)
        mult = np.dot(matrix, solution)
        np.testing.assert_allclose(mult, force)

        side, nsols = 2, 4
        force = [[np.random.randint(-5, 6) for j in range(nsols)] for i in range(side)]
        matrix = ((1, 1), (2, 3))
        solution = Linalg.solve(matrix, force)
        mult = np.dot(matrix, solution)
        np.testing.assert_allclose(mult, force)

        side, nsols = 2, 4
        force = [[np.random.randint(-5, 6) for j in range(nsols)] for i in range(side)]
        matrix = ((1, 1), (11, 12))
        solution = Linalg.solve(matrix, force)
        mult = np.dot(matrix, solution)
        np.testing.assert_allclose(mult, force)

        side, nsols = 4, 4
        force = [[np.random.randint(-5, 6) for j in range(nsols)] for i in range(side)]
        matrix = (
            (1, 1, 1, 1),
            (11, 12, 13, 14),
            (55, 66, 78, 91),
            (165, 220, 286, 364),
        )
        solution = Linalg.solve(matrix, force)
        mult = np.dot(matrix, solution)
        np.testing.assert_allclose(mult, force)

        # Ericksen matrix
        for side in range(1, 10):
            nsols = side + 1
            force = np.random.randint(-10, 10, (side, nsols))
            matrix = np.zeros((side, side), dtype="int64")
            for n in range(side, side + 10):
                for i in range(side):
                    for j in range(side):
                        matrix[i, j] = Math.comb(n + j, i)
            solution = Linalg.solve(matrix, force)
            mult = np.dot(matrix, solution)
            np.testing.assert_allclose(mult, force)

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestLinalg::test_begin",
            "TestLinalg::test_solve_float",
            "TestLinalg::test_solve_integer",
        ]
    )
    def test_solve_fraction(self):
        # Identity
        for side in range(1, 10):
            zero, one = Fraction(0), Fraction(1)
            matrix = [
                [one if i == j else zero for j in range(side)] for i in range(side)
            ]
            test = Linalg.invert(matrix)
            test = np.array(test, dtype="int64")
            good = np.eye(side, dtype="int64")
            np.testing.assert_allclose(test, good)

        # Ericksen matrix
        for side in range(1, 10):
            matrix = np.zeros((side, side), dtype="object")
            for n in range(side, side + 10):
                for i in range(side):
                    for j in range(side):
                        matrix[i, j] = Fraction(Math.comb(n + j, i))
            inverse = Linalg.invert(matrix)
            inverse = np.array(inverse, dtype="int64")
            matrix = np.array(matrix, dtype="int64")
            np.testing.assert_allclose(np.dot(inverse, matrix), np.eye(side))
            np.testing.assert_allclose(np.dot(matrix, inverse), np.eye(side))

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestLinalg::test_begin",
            "TestLinalg::test_solve_float",
            "TestLinalg::test_solve_fraction",
        ]
    )
    def test_specific_case(self):
        f = Fraction
        B = [
            [f(1, 9), f(1, 12), f(5, 84), f(5, 126), f(1, 42), f(1, 84), f(17, 4235)],
            [
                f(1, 36),
                f(1, 21),
                f(5, 84),
                f(4, 63),
                f(5, 84),
                f(204, 4235),
                f(653, 21780),
            ],
            [
                f(1, 252),
                f(1, 84),
                f(1, 42),
                f(5, 126),
                f(51, 847),
                f(653, 7260),
                f(493, 4356),
            ],
        ]
        C = [
            [f(1, 5), f(1, 10), f(727, 21780)],
            [f(1, 10), f(727, 5445), f(1117, 10890)],
            [f(727, 21780), f(1117, 10890), f(1501, 7260)],
        ]
        B = np.array(B)
        C = np.array(C)

        Cinv = Linalg.invert(C)
        solution = np.dot(Cinv, B)
        mult = np.dot(C, solution)
        diff = np.array(mult - B, dtype="float64")
        np.testing.assert_allclose(diff, np.zeros(B.shape))

        solution = Linalg.solve(C, B)
        mult = np.dot(C, solution)
        diff = np.array(mult - B, dtype="float64")
        np.testing.assert_allclose(diff, np.zeros(B.shape))

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestLinalg::test_begin",
            "TestLinalg::test_solve_float",
            "TestLinalg::test_solve_fraction",
        ]
    )
    def test_specific_case2(self):
        f = Fraction
        matrix = [[0, -54, -5], [-9, -20, -3], [-25, -90, 216]]
        good = [
            [f(2295, 55288), f(-6057, 55288), f(-31, 55288)],
            [f(-2019, 110576), f(125, 110576), f(-45, 110576)],
            [f(-155, 55288), f(-675, 55288), f(243, 55288)],
        ]
        prod = np.dot(matrix, np.array(good, dtype="float64"))
        np.testing.assert_allclose(prod, np.eye(3), atol=1e-9)
        test = Linalg.invert(matrix)
        diff = np.array(good - test, dtype="float64")
        np.testing.assert_array_equal(diff, np.zeros(diff.shape))

        matrix = [
            [f(0, 1), f(-3, 4), f(-5, 72)],
            [f(-3, 4), f(-5, 3), f(-1, 4)],
            [f(-5, 72), f(-1, 4), f(3, 5)],
        ]
        inverse = Linalg.invert(matrix)
        prod = np.dot(matrix, inverse).astype("float64")
        np.testing.assert_allclose(prod, np.eye(3))
        prod = np.dot(inverse, matrix).astype("float64")
        np.testing.assert_allclose(prod, np.eye(3))

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestLinalg::test_begin",
            "TestLinalg::test_invert_float",
            "TestLinalg::test_invert_integer",
            "TestLinalg::test_invert_fraction",
            "TestLinalg::test_solve_float",
            "TestLinalg::test_solve_integer",
            "TestLinalg::test_solve_fraction",
            "TestLinalg::test_specific_case",
        ]
    )
    def test_end(self):
        pass


class TestNodeSample:
    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=["test_begin", "TestMath::test_end", "TestLinalg::test_end"]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestNodeSample::test_begin"])
    def test_closed_linspace(self):
        nodes = NodeSample.closed_linspace(2)
        good = (0, 1)
        assert nodes == good

        nodes = NodeSample.closed_linspace(3)
        good = (0, 1 / 2, 1)
        assert nodes == good

        nodes = NodeSample.closed_linspace(4)
        nodes = np.array(nodes, dtype="float64")
        good = (0, 1 / 3, 2 / 3, 1)
        np.testing.assert_allclose(nodes, good)

        nodes = NodeSample.closed_linspace(5)
        good = (0, 1 / 4, 2 / 4, 3 / 4, 1)
        assert nodes == good

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestNodeSample::test_begin"])
    def test_open_linspace(self):
        nodes = NodeSample.open_linspace(1)
        good = (1 / 2,)
        assert nodes == good

        nodes = NodeSample.open_linspace(2)
        good = (1 / 4, 3 / 4)
        assert nodes == good

        nodes = NodeSample.open_linspace(3)
        nodes = np.array(nodes, dtype="float64")
        good = (1 / 6, 3 / 6, 5 / 6)
        np.testing.assert_allclose(nodes, good)

        nodes = NodeSample.open_linspace(4)
        good = (1 / 8, 3 / 8, 5 / 8, 7 / 8)
        assert nodes == good

        nodes = NodeSample.open_linspace(5)
        nodes = np.array(nodes, dtype="float64")
        good = (1 / 10, 3 / 10, 5 / 10, 7 / 10, 9 / 10)
        np.testing.assert_allclose(nodes, good)

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestNodeSample::test_begin"])
    def test_chebyshev(self):
        nodes = NodeSample.chebyshev(1)
        assert nodes == (1 / 2,)

        nodes = NodeSample.chebyshev(2)
        left = (2 - np.sqrt(2)) / 4
        right = (2 + np.sqrt(2)) / 4
        good = (left, right)
        np.testing.assert_allclose(nodes, good)

        nodes = NodeSample.chebyshev(3)
        left = (2 - np.sqrt(3)) / 4
        right = (2 + np.sqrt(3)) / 4
        good = (left, 1 / 2, right)
        np.testing.assert_allclose(nodes, good)

        nodes = NodeSample.chebyshev(4)
        good = np.sin(np.pi * np.array([1 / 16, 3 / 16, 5 / 16, 7 / 16])) ** 2
        np.testing.assert_allclose(nodes, good)

        nodes = NodeSample.chebyshev(5)
        good = np.sin(np.pi * np.array([1 / 20, 3 / 20, 5 / 20, 7 / 20, 9 / 20])) ** 2
        np.testing.assert_allclose(nodes, good)

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestNodeSample::test_begin"])
    def test_gauss_legendre(self):
        nodes = NodeSample.gauss_legendre(1)
        assert nodes == (1 / 2,)

        nodes = NodeSample.gauss_legendre(2)
        minor = 1 / np.sqrt(3)
        good = [(1 - minor) / 2, (1 + minor) / 2]
        np.testing.assert_allclose(nodes, good)

        nodes = NodeSample.gauss_legendre(3)
        minor = np.sqrt(3 / 5)
        good = [(1 - minor) / 2, 1 / 2, (1 + minor) / 2]
        np.testing.assert_allclose(nodes, good)

        nodes = NodeSample.gauss_legendre(4)
        minor = np.sqrt(3 / 7 + 2 * np.sqrt(6 / 5) / 7)
        middl = np.sqrt(3 / 7 - 2 * np.sqrt(6 / 5) / 7)
        good = [(1 - minor) / 2, (1 - middl) / 2, (1 + middl) / 2, (1 + minor) / 2]
        np.testing.assert_allclose(nodes, good)

        nodes = NodeSample.gauss_legendre(5)
        minor = np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3
        middl = np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3
        good = [
            (1 - minor) / 2,
            (1 - middl) / 2,
            1 / 2,
            (1 + middl) / 2,
            (1 + minor) / 2,
        ]
        np.testing.assert_allclose(nodes, good)

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestNodeSample::test_begin",
            "TestNodeSample::test_closed_linspace",
            "TestNodeSample::test_open_linspace",
            "TestNodeSample::test_chebyshev",
            "TestNodeSample::test_gauss_legendre",
        ]
    )
    def test_end(self):
        pass


class TestUnidimentionIntegral:
    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestMath::test_end",
            "TestLinalg::test_end",
            "TestNodeSample::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestUnidimentionIntegral::test_begin"])
    def test_closed_newton_cotes(self):
        a, b = Fraction(3), Fraction(5)
        for degree in range(0, 7):
            npts = max(2, degree + 1)  # Number integration points
            numers = np.random.randint(-5, 5, degree + 1)
            denoms = np.random.randint(2, 8, degree + 1)
            coefs = [Fraction(int(num), int(den)) for num, den in zip(numers, denoms)]
            good = sum(
                ci * (b ** (i + 1) - a ** (i + 1)) / (i + 1)
                for i, ci in enumerate(coefs)
            )

            nodes = NodeSample.closed_linspace(npts)
            weights = IntegratorArray.closed_newton_cotes(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)

            assert test == good

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestUnidimentionIntegral::test_begin"])
    def test_open_newton_cotes(self):
        a, b = Fraction(3), Fraction(5)
        for degree in range(0, 7):
            npts = degree + 1  # Number integration points
            numers = np.random.randint(-5, 5, degree + 1)
            denoms = np.random.randint(2, 8, degree + 1)
            coefs = [Fraction(int(num), int(den)) for num, den in zip(numers, denoms)]
            good = sum(
                ci * (b ** (i + 1) - a ** (i + 1)) / (i + 1)
                for i, ci in enumerate(coefs)
            )

            nodes = NodeSample.open_linspace(npts)
            weights = IntegratorArray.open_newton_cotes(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)

            assert test == good

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestUnidimentionIntegral::test_begin"])
    def test_chebyshev(self):
        a, b = Fraction(3), Fraction(7)
        for degree in range(0, 7):
            npts = degree + 1  # Number integration points
            numers = np.random.randint(-5, 5, degree + 1)
            denoms = np.random.randint(2, 8, degree + 1)
            coefs = [Fraction(int(num), int(den)) for num, den in zip(numers, denoms)]
            good = sum(
                ci * (b ** (i + 1) - a ** (i + 1)) / (i + 1)
                for i, ci in enumerate(coefs)
            )

            nodes = NodeSample.chebyshev(npts)
            weights = IntegratorArray.chebyshev(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)

            assert abs(test - good) < 1e-9

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestUnidimentionIntegral::test_begin"])
    def test_gauss_legendre(self):
        a, b = Fraction(3), Fraction(7)
        for degree in range(0, 7):
            npts = degree + 1  # Number integration points
            numers = np.random.randint(-5, 5, degree + 1)
            denoms = np.random.randint(2, 8, degree + 1)
            coefs = [Fraction(int(num), int(den)) for num, den in zip(numers, denoms)]
            good = sum(
                ci * (b ** (i + 1) - a ** (i + 1)) / (i + 1)
                for i, ci in enumerate(coefs)
            )

            nodes = NodeSample.gauss_legendre(npts)
            weights = IntegratorArray.gauss_legendre(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)

            assert abs(test - good) < 1e-9

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestUnidimentionIntegral::test_begin",
            "TestUnidimentionIntegral::test_closed_newton_cotes",
            "TestUnidimentionIntegral::test_open_newton_cotes",
            "TestUnidimentionIntegral::test_chebyshev",
            "TestUnidimentionIntegral::test_gauss_legendre",
        ]
    )
    def test_exact_integral_fraction(self):
        a, b = Fraction(3), Fraction(7)
        for degree in range(0, 7):
            npts = 1 + 2 * math.floor(degree / 2)  # Number integration points
            numers = np.random.randint(-5, 5, degree + 1)
            denoms = np.random.randint(2, 8, degree + 1)
            coefs = [Fraction(int(num), int(den)) for num, den in zip(numers, denoms)]
            good = sum(
                ci * (b ** (i + 1) - a ** (i + 1)) / (i + 1)
                for i, ci in enumerate(coefs)
            )

            if npts > 1:
                nodes = NodeSample.open_linspace(npts)
                weights = IntegratorArray.open_newton_cotes(npts)
                nodes = tuple(a + (b - a) * node for node in nodes)
                funcvals = tuple(
                    sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
                )
                test = (b - a) * np.inner(weights, funcvals)
                assert test == good

            nodes = NodeSample.open_linspace(npts)
            weights = IntegratorArray.open_newton_cotes(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)
            assert test == good

            nodes = NodeSample.chebyshev(npts)
            weights = IntegratorArray.chebyshev(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)
            assert abs(test - good) < 1e-9

            nodes = NodeSample.gauss_legendre(npts)
            weights = IntegratorArray.gauss_legendre(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)
            assert abs(test - good) < 1e-9

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestUnidimentionIntegral::test_begin",
            "TestUnidimentionIntegral::test_closed_newton_cotes",
            "TestUnidimentionIntegral::test_open_newton_cotes",
            "TestUnidimentionIntegral::test_chebyshev",
            "TestUnidimentionIntegral::test_gauss_legendre",
            "TestUnidimentionIntegral::test_exact_integral_fraction",
        ]
    )
    def test_approx_integral(self):
        a, b = Fraction(3), Fraction(7)
        for degree in range(0, 7):
            npts = 1 + 2 * math.floor(degree / 2)  # Number integration points
            coefs = np.random.uniform(-5, 6, degree + 1)
            good = sum(
                ci * (b ** (i + 1) - a ** (i + 1)) / (i + 1)
                for i, ci in enumerate(coefs)
            )

            if npts > 1:
                nodes = NodeSample.open_linspace(npts)
                weights = IntegratorArray.open_newton_cotes(npts)
                nodes = tuple(a + (b - a) * node for node in nodes)
                funcvals = tuple(
                    sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
                )
                test = (b - a) * np.inner(weights, funcvals)
                assert abs(test - good) < 1e-9

            nodes = NodeSample.open_linspace(npts)
            weights = IntegratorArray.open_newton_cotes(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)
            assert abs(test - good) < 1e-9

            nodes = NodeSample.chebyshev(npts)
            weights = IntegratorArray.chebyshev(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)
            assert abs(test - good) < 1e-9

            nodes = NodeSample.gauss_legendre(npts)
            weights = IntegratorArray.gauss_legendre(npts)
            nodes = tuple(a + (b - a) * node for node in nodes)
            funcvals = tuple(
                sum([cj * xi**j for j, cj in enumerate(coefs)]) for xi in nodes
            )
            test = (b - a) * np.inner(weights, funcvals)
            assert abs(test - good) < 1e-9

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestUnidimentionIntegral::test_begin",
            "TestUnidimentionIntegral::test_closed_newton_cotes",
            "TestUnidimentionIntegral::test_open_newton_cotes",
            "TestUnidimentionIntegral::test_chebyshev",
            "TestUnidimentionIntegral::test_gauss_legendre",
            "TestUnidimentionIntegral::test_exact_integral_fraction",
            "TestUnidimentionIntegral::test_approx_integral",
        ]
    )
    def test_end(self):
        pass


class TestLeastSquare:
    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestMath::test_end",
            "TestLinalg::test_end",
            "TestNodeSample::test_end",
            "TestUnidimentionIntegral::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestLeastSquare::test_begin"])
    def test_leastsquarespline_identity(self):
        U0 = [0, 0, 1, 1]
        U1 = [0, 0, 1, 1]
        T, E = LeastSquare.spline2spline(U0, U1)
        np.testing.assert_almost_equal(T, np.eye(2))
        assert np.all(np.abs(E) < 1e-9)

        U0 = [0, 0, 0, 1, 1, 1]
        U1 = [0, 0, 0, 1, 1, 1]
        T, E = LeastSquare.spline2spline(U0, U1)
        np.testing.assert_almost_equal(T, np.eye(3))
        assert np.all(np.abs(E) < 1e-9)

        U0 = [0, 0, 0, 0.5, 1, 1, 1]
        U1 = [0, 0, 0, 0.5, 1, 1, 1]
        T, E = LeastSquare.spline2spline(U0, U1)
        np.testing.assert_almost_equal(T, np.eye(4))
        assert np.all(np.abs(E) < 1e-9)

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestLeastSquare::test_begin"])
    def test_leastsquarespline_eval_error(self):
        # knot insertion
        U0 = [0, 0, 0, 1, 1, 1]
        U1 = [0, 0, 0, 0.5, 1, 1, 1]
        _, E = LeastSquare.spline2spline(U0, U1)
        assert np.all(np.abs(E) < 1e-9)

        # degree elevate
        U0 = [0, 0, 1, 1]
        U1 = [0, 0, 0, 1, 1, 1]
        _, E = LeastSquare.spline2spline(U0, U1)
        assert np.all(np.abs(E) < 1e-9)

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestLeastSquare::test_begin",
            "TestLeastSquare::test_leastsquarespline_identity",
            "TestLeastSquare::test_leastsquarespline_eval_error",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=[
        "TestMath::test_end",
        "TestLinalg::test_end",
        "TestNodeSample::test_end",
        "TestUnidimentionIntegral::test_end",
        "TestLeastSquare::test_end",
    ]
)
def test_end():
    pass
