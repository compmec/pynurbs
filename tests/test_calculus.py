"""
This file is responsible to testing the code inside the file ```calculus.py```
Its functions are getting derivatives, computing integrals along curves and so on
"""

from fractions import Fraction

import numpy as np
import pytest

from pynurbs.calculus import Derivate, Integrate
from pynurbs.curves import Curve
from pynurbs.knotspace import GeneratorKnotVector, KnotVector


@pytest.mark.order(7)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_functions.py::test_end",
        "tests/test_beziercurve.py::test_end",
        "tests/test_splinecurve.py::test_end",
        "tests/test_rationalcurve.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestDerivBezier:
    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestDerivBezier::test_begin"])
    def test_bezier_degree1(self):
        curve = Curve(GeneratorKnotVector.bezier(1))
        P = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = P

        test_curve = Derivate(curve)
        good_curve = Curve(GeneratorKnotVector.bezier(0))
        good_curve.ctrlpoints = [P[1] - P[0]]

        assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestDerivBezier::test_bezier_degree1"])
    def test_bezier_degree2(self):
        curve = Curve(GeneratorKnotVector.bezier(2))
        P = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = P

        test_curve = Derivate(curve)

        good_curve = Curve(GeneratorKnotVector.bezier(1))
        good_curve.ctrlpoints = [2 * (P[1] - P[0]), 2 * (P[2] - P[1])]

        assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestDerivBezier::test_bezier_degree2"])
    def test_bezier_degree3(self):
        curve = Curve(GeneratorKnotVector.bezier(3))
        P = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = P

        test_curve = Derivate(curve)

        good_curve = Curve(GeneratorKnotVector.bezier(2))
        good_curve.ctrlpoints = [
            3 * (P[1] - P[0]),
            3 * (P[2] - P[1]),
            3 * (P[3] - P[2]),
        ]

        assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestDerivBezier::test_bezier_degree2"])
    def test_random_degree(self):
        for degree in range(1, 7):
            vector = GeneratorKnotVector.bezier(degree)
            curve = Curve(vector)
            points = np.random.uniform(-1, 1, curve.npts)
            curve.ctrlpoints = points

            test_curve = Derivate(curve)

            good_vector = GeneratorKnotVector.bezier(degree - 1)
            good_curve = Curve(good_vector)
            good_curve.ctrlpoints = [
                degree * (points[i + 1] - points[i]) for i in range(degree)
            ]

            assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestDerivBezier::test_begin",
            "TestDerivBezier::test_bezier_degree1",
            "TestDerivBezier::test_bezier_degree2",
            "TestDerivBezier::test_bezier_degree3",
            "TestDerivBezier::test_random_degree",
        ]
    )
    def test_end(self):
        pass


class TestNumericalDeriv:
    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestNumericalDeriv::test_begin"])
    def test_bezier(self):
        deltau = 1e-6
        usample = np.linspace(2 * deltau, 1 - 2 * deltau, 5)
        for degree in range(0, 4):
            vector = GeneratorKnotVector.bezier(degree)
            curve = Curve(vector)
            points = np.random.uniform(-1, 1, curve.npts)
            curve.ctrlpoints = points

            dcurve = Derivate(curve)
            for node in usample:
                dnumer = (curve(node + deltau) - curve(node - deltau)) / (2 * deltau)
                assert np.abs(dcurve(node) - dnumer) < 1e-6

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestNumericalDeriv::test_begin", "TestNumericalDeriv::test_bezier"]
    )
    def test_spline(self):
        deltau = 1e-6
        for degree in range(0, 4):
            for npts in range(degree + 3, degree + 9):
                vector = GeneratorKnotVector.uniform(degree, npts)
                knots = vector.knots

                curve = Curve(vector)
                points = np.random.uniform(-1, 1, curve.npts)
                curve.ctrlpoints = points

                dcurve = Derivate(curve)
                for start, end in zip(knots[:-1], knots[1:]):
                    usample = np.linspace(start + 2 * deltau, end - 2 * deltau, 5)
                    for node in usample:
                        dnumer = (curve(node + deltau) - curve(node - deltau)) / (
                            2 * deltau
                        )
                        assert np.abs(dcurve(node) - dnumer) < 1e-6

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestNumericalDeriv::test_begin",
            "TestNumericalDeriv::test_bezier",
            "TestNumericalDeriv::test_spline",
        ]
    )
    def test_rationalbezier(self):
        deltau = 1e-6
        usample = np.linspace(2 * deltau, 1 - 2 * deltau, 5)
        for degree in range(0, 4):
            vector = GeneratorKnotVector.bezier(degree)
            curve = Curve(vector)
            points = np.random.uniform(-1, 1, curve.npts)
            weighs = np.random.uniform(1, 2, curve.npts)
            curve.ctrlpoints = points
            curve.weights = weighs

            dcurve = Derivate(curve)
            for node in usample:
                dnumer = curve(node + deltau) - curve(node - deltau)
                dnumer /= 2 * deltau
                assert np.abs(dcurve(node) - dnumer) < 1e-6

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestNumericalDeriv::test_begin",
            "TestNumericalDeriv::test_bezier",
            "TestNumericalDeriv::test_spline",
            "TestNumericalDeriv::test_rationalbezier",
        ]
    )
    def test_rationalspline(self):
        deltau = 1e-6
        for degree in range(0, 4):
            for npts in range(degree + 3, degree + 9):
                vector = GeneratorKnotVector.uniform(degree, npts)
                knots = vector.knots
                curve = Curve(vector)
                points = np.random.uniform(-1, 1, curve.npts)
                weighs = np.random.uniform(1, 2, curve.npts)
                curve.ctrlpoints = points
                curve.weights = weighs

                dcurve = Derivate(curve)
                for start, end in zip(knots[:-1], knots[1:]):
                    usample = np.linspace(start + 2 * deltau, end - 2 * deltau, 5)
                    for node in usample:
                        dnumer = curve(node + deltau) - curve(node - deltau)
                        dnumer /= 2 * deltau
                        assert np.abs(dcurve(node) - dnumer) < 1e-6

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestNumericalDeriv::test_begin",
            "TestNumericalDeriv::test_bezier",
            "TestNumericalDeriv::test_spline",
            "TestNumericalDeriv::test_rationalbezier",
            "TestNumericalDeriv::test_rationalspline",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(7)
@pytest.mark.dependency(depends=["TestDerivBezier::test_end"])
def test_derivate_integers_knotvector():
    knotvector = [0, 0, 1, 2, 3, 4, 5, 5]
    knotvector = KnotVector(knotvector)
    assert knotvector.degree == 1
    points = np.random.uniform(-1, 1, knotvector.npts)
    curve = Curve(knotvector, points)

    dcurve = Derivate(curve)
    knots = np.array(curve.knotvector.knots, dtype="float64")
    midnodes = (knots[:-1] + knots[1:]) / 2
    for i, node in enumerate(midnodes):
        assert np.abs((points[i + 1] - points[i]) - dcurve(node)) < 1e-9


@pytest.mark.order(7)
@pytest.mark.dependency(depends=["TestDerivBezier::test_end"])
def test_example31page94nurbsbook():
    # Example 3.1 at page 94 of Nurbs book
    knotvector = KnotVector([0, 0, 0, 2 / 5, 3 / 5, 1, 1, 1])
    P = np.random.uniform(-1, 1, (knotvector.npts, 2))
    curve = Curve(knotvector)
    curve.ctrlpoints = P

    Q = np.empty((4, 2), dtype="float64")
    newknotvector = KnotVector([0, 0, 2 / 5, 3 / 5, 1, 1])
    Q[0] = 5 * (P[1] - P[0])
    Q[1] = 10 * (P[2] - P[1]) / 3
    Q[2] = 10 * (P[3] - P[2]) / 3
    Q[3] = 5 * (P[4] - P[3])
    good_curve = Curve(newknotvector, Q)

    test_curve = Derivate(curve)
    assert test_curve == good_curve


class TestIntegBezier:
    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestIntegBezier::test_begin"])
    def test_scalar_integral(self):
        curve = Curve(GeneratorKnotVector.bezier(1))
        points = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = points
        test = Integrate.scalar(curve)
        good = sum(points) / 2
        assert abs(test - good) < 1e-9

        curve = Curve(GeneratorKnotVector.bezier(2))
        points = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = points
        test = Integrate.scalar(curve)
        good = sum(points) / 3
        assert abs(test - good) < 1e-9

        curve = Curve(GeneratorKnotVector.bezier(3))
        points = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = points
        test = Integrate.scalar(curve)
        good = sum(points) / 4
        assert abs(test - good) < 1e-9

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestIntegBezier::test_begin"])
    def test_lenght_integral(self):
        curve = Curve(GeneratorKnotVector.bezier(1))
        points = np.random.uniform(-1, 1, (curve.npts, 2))
        curve.ctrlpoints = points

        test = Integrate.lenght(curve)
        good = np.linalg.norm(points[1] - points[0])

        assert abs(test - good) < 1e-9

        curve = Curve(GeneratorKnotVector.uniform(1, 3))
        points = np.random.uniform(-1, 1, (curve.npts, 2))
        curve.ctrlpoints = points

        test = Integrate.lenght(curve)
        good = sum(
            np.linalg.norm(points[i + 1] - points[i]) for i in range(curve.npts - 1)
        )

        assert abs(test - good) < 1e-9

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestIntegBezier::test_begin"])
    def test_winding_number(self):
        knotvector = GeneratorKnotVector.uniform(1, 5)
        curvex = Curve(knotvector, [1, 0, -1, 0, 1])
        curvey = Curve(knotvector, [0, 1, 0, -1, 0])
        dcurvex = Derivate(curvex)
        dcurvey = Derivate(curvey)

        numer = lambda u: curvex(u) * dcurvey(u) - curvey(u) * dcurvex(u)
        denom = lambda u: 2 * np.pi * (curvex(u) ** 2 + curvey(u) ** 2)
        function = lambda u: numer(u) / denom(u)

        test = Integrate.function(knotvector, function, "chebyshev")
        assert abs(test - 1) < 2e-1
        test = Integrate.function(knotvector, function, nnodes=5)
        assert abs(test - 1) < 1e-3
        test = Integrate.function(knotvector, function, "chebyshev", 5)
        assert abs(test - 1) < 1e-3
        test = Integrate.function(knotvector, function, "gauss-legendre", 5)
        assert abs(test - 1) < 1e-3
        test = Integrate.function(knotvector, function, "chebyshev", 7)
        assert abs(test - 1) < 1e-3
        test = Integrate.function(knotvector, function, "gauss-legendre", 7)
        assert abs(test - 1) < 1e-3

        knotvector = GeneratorKnotVector.uniform(1, 5, Fraction)
        curvex = Curve(knotvector, [1, 0, -1, 0, 1])
        curvey = Curve(knotvector, [0, 1, 0, -1, 0])
        new_knots = [Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8)]
        curvex.knot_insert(new_knots)
        curvey.knot_insert(new_knots)
        knotvector += new_knots
        dcurvex = Derivate(curvex)
        dcurvey = Derivate(curvey)

        numer = lambda u: curvex(u) * dcurvey(u) - curvey(u) * dcurvex(u)
        denom = lambda u: curvex(u) ** 2 + curvey(u) ** 2
        function = lambda u: numer(u) / denom(u)

        test = Integrate.function(knotvector, function, "closed-newton-cotes", 6)
        assert abs(test - 2 * np.pi) < 1e-3
        test = Integrate.function(knotvector, function, "open-newton-cotes", 6)
        assert abs(test - 2 * np.pi) < 1e-3

        test = Integrate.function(knotvector, function, "closed-newton-cotes")
        assert abs(test - 2 * np.pi) < 1
        test = Integrate.function(knotvector, function, "open-newton-cotes")
        assert abs(test - 2 * np.pi) < 1

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestIntegBezier::test_begin",
            "TestIntegBezier::test_scalar_integral",
            "TestIntegBezier::test_lenght_integral",
            "TestIntegBezier::test_winding_number",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(7)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestDerivBezier::test_end",
        "TestNumericalDeriv::test_end",
        "test_derivate_integers_knotvector",
        "test_example31page94nurbsbook",
        "TestIntegBezier::test_end",
    ]
)
def test_end():
    pass
