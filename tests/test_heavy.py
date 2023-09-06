import math
from fractions import Fraction

import numpy as np
import pytest

from compmec.nurbs.heavy import LeastSquare, Linalg, Math


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
        print("-" * 20)
        assert Math.lcm(2, 3, 4) == 12
        assert Math.lcm(6, 9, 12) == 36

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestMath::test_begin",
            "TestMath::test_gcd",
            "TestMath::test_lcm",
        ]
    )
    def test_end(self):
        pass


class TestLinalg:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin"])
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
                        matrix[i, j] = math.comb(n + j, i)
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
                        matrix[i, j] = Fraction(math.comb(n + j, i))
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
        floatmatrix = np.random.uniform(-1, 1, (size, size))
        fracmatrix = np.empty((size, size), dtype="object")
        for i, line in enumerate(floatmatrix):
            for j, elem in enumerate(line):
                fracmatrix[i, j] = Fraction(elem).limit_denominator(10)
        fracmatrix += np.transpose(fracmatrix)
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
        print("mult = ")
        print(mult)
        print("force = ")
        print(force)
        np.testing.assert_allclose(mult, force)

        side, nsols = 2, 4
        force = [[np.random.randint(-5, 6) for j in range(nsols)] for i in range(side)]
        matrix = ((1, 1), (2, 3))
        solution = Linalg.solve(matrix, force)
        mult = np.dot(matrix, solution)
        print("mult = ")
        print(mult)
        print("force = ")
        print(force)
        np.testing.assert_allclose(mult, force)

        side, nsols = 2, 4
        force = [[np.random.randint(-5, 6) for j in range(nsols)] for i in range(side)]
        matrix = ((1, 1), (11, 12))
        solution = Linalg.solve(matrix, force)
        mult = np.dot(matrix, solution)
        print("mult = ")
        print(mult)
        print("force = ")
        print(force)
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
        print("mult = ")
        print(mult)
        print("force = ")
        print(force)
        np.testing.assert_allclose(mult, force)

        # Ericksen matrix
        for side in range(1, 10):
            nsols = side + 1
            force = [
                [np.random.randint(-10, 10) for j in range(nsols)] for i in range(side)
            ]
            matrix = np.zeros((side, side), dtype="int64")
            for n in range(side, side + 10):
                for i in range(side):
                    for j in range(side):
                        matrix[i, j] = math.comb(n + j, i)
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
                        matrix[i, j] = Fraction(math.comb(n + j, i))
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


class TestLeastSquare:
    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=["test_begin", "TestMath::test_end", "TestLinalg::test_end"]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestLeastSquare::test_begin"])
    def test_chebyshev_nodes(self):
        nodes = LeastSquare.chebyshev_nodes(1)
        assert nodes[0] == 0.5

        nodes = LeastSquare.chebyshev_nodes(2)
        assert abs(nodes[0] - (2 - np.sqrt(2)) / 4) < 1e-9
        assert abs(nodes[1] - (2 + np.sqrt(2)) / 4) < 1e-9

        nodes = LeastSquare.chebyshev_nodes(3)
        assert abs(nodes[0] - (2 - np.sqrt(3)) / 4) < 1e-9
        assert nodes[1] == 0.5
        assert abs(nodes[2] - (2 + np.sqrt(3)) / 4) < 1e-9

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestLeastSquare::test_begin"])
    def test_integral_value(self):
        """
        This is for testing the array LeastSquare.integrator_array
        """
        degree = 2  # Polynomial degree to integrate
        coeffs = np.random.uniform(-1, 1, degree + 1)

        def func(x: float) -> float:
            return sum([coef * x**i for i, coef in enumerate(coeffs)])

        # Symbolic integral, since it's polynomial
        symbintegral = sum([coef / (i + 1) for i, coef in enumerate(coeffs)])

        npts = 10  # Number of integration points
        nodes = LeastSquare.chebyshev_nodes(npts)
        fvals = [func(xi) for xi in nodes]
        integrator = LeastSquare.integrator_array(nodes)
        numeintegral = np.array(integrator) @ fvals
        assert np.abs(symbintegral - numeintegral) < 1e-9

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestLeastSquare::test_begin"])
    def test_leastsquarespline_identity(self):
        U0 = [0, 0, 1, 1]
        U1 = [0, 0, 1, 1]
        T, _ = LeastSquare.spline2spline(U0, U1)
        np.testing.assert_almost_equal(T, np.eye(2))

        U0 = [0, 0, 0, 1, 1, 1]
        U1 = [0, 0, 0, 1, 1, 1]
        T, _ = LeastSquare.spline2spline(U0, U1)
        np.testing.assert_almost_equal(T, np.eye(3))

        U0 = [0, 0, 0, 0.5, 1, 1, 1]
        U1 = [0, 0, 0, 0.5, 1, 1, 1]
        T, _ = LeastSquare.spline2spline(U0, U1)
        np.testing.assert_almost_equal(T, np.eye(4))

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestLeastSquare::test_begin"])
    def test_leastsquarespline_error(self):
        U0 = [0, 0, 1, 1]
        U1 = [0, 0, 1, 1]  # Same curve
        _, E = LeastSquare.spline2spline(U0, U1)
        assert np.all(np.abs(E) < 1e-9)

        U0 = [0, 0, 0, 1, 1, 1]
        U1 = [0, 0, 0, 1, 1, 1]
        _, E = LeastSquare.spline2spline(U0, U1)
        assert np.all(np.abs(E) < 1e-9)

        U0 = [0, 0, 0, 0.5, 1, 1, 1]
        U1 = [0, 0, 0, 0.5, 1, 1, 1]
        _, E = LeastSquare.spline2spline(U0, U1)
        assert np.all(np.abs(E) < 1e-9)

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
            "TestLeastSquare::test_chebyshev_nodes",
            "TestLeastSquare::test_integral_value",
            "TestLeastSquare::test_leastsquarespline_identity",
            "TestLeastSquare::test_leastsquarespline_error",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["TestMath::test_end", "TestLeastSquare::test_end"])
def test_end():
    pass
