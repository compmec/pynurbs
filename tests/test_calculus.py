"""
This file is responsible to testing the code inside the file ```calculus.py```
Its functions are getting derivatives, computing integrals along curves and so on
"""

import numpy as np
import pytest

from compmec.nurbs import calculus
from compmec.nurbs.curves import Curve
from compmec.nurbs.functions import Function
from compmec.nurbs.knotspace import GeneratorKnotVector, KnotVector


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


class TestBezier:
    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestBezier::test_begin"])
    def test_derivative_bezier_degree1(self):
        curve = Curve(GeneratorKnotVector.bezier(1))
        P = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = P

        test_curve = calculus.derivate_curve(curve)
        good_curve = Curve(GeneratorKnotVector.bezier(0))
        good_curve.ctrlpoints = [P[1] - P[0]]

        assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestBezier::test_derivative_bezier_degree1"])
    def test_derivative_bezier_degree2(self):
        curve = Curve(GeneratorKnotVector.bezier(2))
        P = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = P

        test_curve = calculus.derivate_curve(curve)

        good_curve = Curve(GeneratorKnotVector.bezier(1))
        good_curve.ctrlpoints = [2 * (P[1] - P[0]), 2 * (P[2] - P[1])]

        assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestBezier::test_derivative_bezier_degree2"])
    def test_derivative_bezier_degree3(self):
        curve = Curve(GeneratorKnotVector.bezier(3))
        P = np.random.uniform(-1, 1, curve.npts)
        curve.ctrlpoints = P

        test_curve = calculus.derivate_curve(curve)

        good_curve = Curve(GeneratorKnotVector.bezier(2))
        good_curve.ctrlpoints = [
            3 * (P[1] - P[0]),
            3 * (P[2] - P[1]),
            3 * (P[3] - P[2]),
        ]

        assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_begin",
            "TestBezier::test_derivative_bezier_degree1",
            "TestBezier::test_derivative_bezier_degree2",
            "TestBezier::test_derivative_bezier_degree3",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(7)
@pytest.mark.dependency(depends=["TestBezier::test_end"])
def test_derivate_uniform_knotvector():
    knotvector = [0, 0, 1, 2, 3, 4, 5, 5]
    knotvector = KnotVector(knotvector)
    assert knotvector.degree == 1
    points = np.random.uniform(-1, 1, knotvector.npts)
    curve = Curve(knotvector, points)

    dcurve = calculus.derivate_curve(curve)
    knots = np.array(curve.knotvector.knots, dtype="float64")
    midnodes = (knots[:-1] + knots[1:]) / 2
    for i, node in enumerate(midnodes):
        assert np.abs((points[i + 1] - points[i]) - dcurve(node)) < 1e-9


@pytest.mark.order(7)
@pytest.mark.dependency(depends=["TestBezier::test_end"])
def test_derivative_curve():
    # Example 3.1 at page 94 of Nurbs book
    knotvector = KnotVector([0, 0, 0, 2 / 5, 3 / 5, 1, 1, 1])
    P = np.random.uniform(-1, 1, (knotvector.npts, 2))
    curve = Curve(knotvector)
    curve.ctrlpoints = P

    Q = np.empty((4, 2), dtype="float64")
    newknotvector = KnotVector([0, 0, 2 / 5, 3 / 5, 1, 1])
    Q[0] = 5 * (P[1] - P[0])
    Q[1] = 10 * (P[1] - P[0]) / 3
    Q[2] = 10 * (P[1] - P[0]) / 3
    Q[3] = 5 * (P[1] - P[0])
    basisfunc = Function(newknotvector)

    test_curve = calculus.derivate_curve(curve)


@pytest.mark.order(7)
@pytest.mark.dependency(depends=["test_begin"])
def test_multcurve():
    knotvector = [0, 0, 1, 1]
    curve = Curve(knotvector)
    curve.ctrlpoints = [1, 0]  # curve(u) = 1-u

    goodknotvector = [0, 0, 0, 1, 1, 1]
    goodcurve = Curve(goodknotvector)
    goodcurve.ctrlpoints = [1, 0, 0]  # curve(u) = (1-u)^2

    testcurve = calculus.MathOperations.mult_spline(curve, curve)
    assert testcurve == goodcurve

    knotvectora = [0, 0, 0, 1, 1, 1]
    knotvectorb = [0, 0, 1, 1]
    curvea = Curve(knotvectora)
    curveb = Curve(knotvectorb)
    curvea.ctrlpoints = [1, 0, 0]  # curvea(u) = (1-u)^2
    curveb.ctrlpoints = [1, 0]  # curveb(u) = 1-u

    goodknotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    goodcurve = Curve(goodknotvector)
    goodcurve.ctrlpoints = [1, 0, 0, 0]  # curve(u) = (1-u)^3

    testcurve = calculus.MathOperations.mult_spline(curvea, curveb)
    assert testcurve == goodcurve


@pytest.mark.order(7)
@pytest.mark.dependency(depends=["test_begin"])
def test_divcurve():
    knotvectora = [0, 0, 0, 1, 1, 1]
    ctrlpointsa = [1, 0, 0]
    knotvectorb = [0, 0, 0, 1, 1, 1]
    ctrlpointsb = [1, 1, 2]
    curvea = Curve(knotvectora, ctrlpointsa)
    curveb = Curve(knotvectorb, ctrlpointsb)

    goodcurve = Curve([0, 0, 0, 1, 1, 1])
    goodcurve.ctrlpoints = ctrlpointsa
    goodcurve.weights = ctrlpointsb

    testcurve = calculus.MathOperations.div_spline(curvea, curveb, False)
    assert testcurve == goodcurve

    knotvectora = [0, 0, 0, 1, 1, 1]
    curvea = Curve(knotvectora)
    curvea.ctrlpoints = [1, 0, 0]  # curve(u) = (1-u)^2

    knotvectorb = [0, 0, 1, 1]
    curveb = Curve(knotvectorb)
    curveb.ctrlpoints = [1, 0]  # curve(u) = (1-u)

    goodcurve = curveb.deepcopy()
    testcurve = calculus.MathOperations.div_spline(curvea, curveb)

    assert testcurve == goodcurve


@pytest.mark.order(7)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestBezier::test_end",
        "test_derivative_curve",
        "test_multcurve",
        "test_divcurve",
    ]
)
def test_end():
    pass
