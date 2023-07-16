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
        # "tests/test_knotspace.py::test_end",
        # "tests/test_functions.py::test_end",
        # "tests/test_beziercurve.py::test_end",
        # "tests/test_splinecurve.py::test_end",
        # "tests/test_rationalcurve.py::test_end",
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
            "TestBezier::test_derivative_bezier_degree1",
            "TestBezier::test_derivative_bezier_degree2",
            "TestBezier::test_derivative_bezier_degree3",
        ]
    )
    def test_end(self):
        pass


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
@pytest.mark.dependency(depends=["test_begin", "test_derivative_curve"])
def test_end():
    pass
