from fractions import Fraction as frac

import numpy as np
import pytest

from compmec.nurbs.curves import Curve
from compmec.nurbs.knotspace import GeneratorKnotVector


@pytest.mark.order(6)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_functions.py::test_end",
        "tests/test_beziercurve.py::test_end",
        "tests/test_splinecurve.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestBuild:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestBuild::test_begin"])
    def test_failbuild(self):
        for degree in range(1, 6):
            npts = np.random.randint(degree + 1, degree + 9)
            knotvector = GeneratorKnotVector.random(degree, npts)
            curve = Curve(knotvector)
            with pytest.raises(ValueError):
                curve.weights = 1
            with pytest.raises(ValueError):
                curve.weights = "asd"

    @pytest.mark.order(6)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestBuild::test_failbuild"])
    def test_print(self):
        knotvector = GeneratorKnotVector.uniform(2, 4)
        rational = Curve(knotvector)
        str(rational)
        rational.ctrlpoints = [2, 4, 3, 1]
        str(rational)
        rational.weights = (2, 3, 1, 4)
        str(rational)

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestBuild::test_begin",
            "TestBuild::test_failbuild",
            "TestBuild::test_print",
        ]
    )
    def test_end(self):
        pass


class TestCircle:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestBuild::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestCircle::test_begin"])
    def test_quarter_circle_standard(self):
        knotvector = [0, 0, 0, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 1), (0, 1)]
        weights = [1, 1, 2]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = weights
        nodes_sample = np.linspace(0, 1, 129)
        points = curve(nodes_sample)
        for point in points:
            assert abs(np.linalg.norm(point) - 1) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestCircle::test_quarter_circle_standard"])
    def test_quarter_circle_symmetric(self):
        knotvector = [0, 0, 0, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 1), (0, 1)]
        weights = [2, np.sqrt(2), 2]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = weights
        nodes_sample = np.linspace(0, 1, 129)
        points = curve(nodes_sample)
        for point in points:
            assert abs(np.linalg.norm(point) - 1) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestCircle::test_quarter_circle_standard",
            "TestCircle::test_quarter_circle_symmetric",
        ]
    )
    def test_half_circle(self):
        knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 2), (-1, 2), (-1, 0)]
        weights = [3, 1, 1, 3]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = weights
        nodes_sample = np.linspace(0, 1, 129)
        points = curve(nodes_sample)
        for point in points:
            assert abs(np.linalg.norm(point) - 1) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestCircle::test_quarter_circle_standard",
            "TestCircle::test_quarter_circle_symmetric",
        ]
    )
    def test_full_circle(self):
        knotvector = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 2), (-1, 2), (-1, 0), (-1, -2), (1, -2), (1, 0)]
        weights = [3, 1, 1, 3, 1, 1, 3]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = weights
        nodes_sample = np.linspace(0, 1, 129)
        points = curve(nodes_sample)
        for point in points:
            assert abs(np.linalg.norm(point) - 1) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestCircle::test_begin",
            "TestCircle::test_quarter_circle_standard",
            "TestCircle::test_quarter_circle_symmetric",
            "TestCircle::test_half_circle",
            "TestCircle::test_full_circle",
        ]
    )
    def test_end(self):
        pass


class TestInsKnotCircle:
    @pytest.mark.order(6)
    # @pytest.mark.skip(reason="Needs knot insertion correction")
    @pytest.mark.dependency(depends=["TestCircle::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.skip(reason="Insert knot gives problem")
    @pytest.mark.dependency(depends=["TestInsKnotCircle::test_begin"])
    def test_quarter_circle_standard(self):
        zero = frac(0, 1)
        one = frac(1, 1)
        knotvector = [zero, zero, zero, one, one, one]
        ctrlpoints = [(one, zero), (one, one), (zero, one)]
        weights = [one, one, 2 * one]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = np.array(weights)

        newcurve = curve.deepcopy()
        newcurve.knot_insert([one / 2])

        nodes_sample = [frac(i, 128) for i in range(129)]
        points_old = curve(nodes_sample)
        points_new = newcurve(nodes_sample)
        for oldpt, newpt in zip(points_old, points_new):
            diff = np.array(oldpt) - newpt
            distsquare = sum(diff**2)
            assert abs(distsquare) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInsKnotCircle::test_quarter_circle_standard"])
    def test_quarter_circle_symmetric(self):
        knotvector = [0, 0, 0, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 1), (0, 1)]
        weights = [2, np.sqrt(2), 2]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints, dtype="float64")
        curve.weights = np.array(weights, dtype="float64")

        newcurve = curve.deepcopy()
        newcurve.knot_insert([0.5])

        nodes_sample = np.linspace(0, 1, 129)
        points_old = curve(nodes_sample)
        points_new = newcurve(nodes_sample)
        for oldpt, newpt in zip(points_old, points_new):
            assert abs(np.linalg.norm(oldpt - newpt)) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestInsKnotCircle::test_quarter_circle_standard",
            "TestInsKnotCircle::test_quarter_circle_symmetric",
        ]
    )
    def test_half_circle(self):
        knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 2), (-1, 2), (-1, 0)]
        weights = [3, 1, 1, 3]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints, dtype="float64")
        curve.weights = np.array(weights, dtype="float64")

        newcurve = curve.deepcopy()
        newcurve.knot_insert([0.5])

        nodes_sample = np.linspace(0, 1, 129)
        points_old = curve(nodes_sample)
        points_new = newcurve(nodes_sample)
        for oldpt, newpt in zip(points_old, points_new):
            assert abs(np.linalg.norm(oldpt - newpt)) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestInsKnotCircle::test_quarter_circle_standard",
            "TestInsKnotCircle::test_quarter_circle_symmetric",
        ]
    )
    def test_full_circle(self):
        knotvector = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 2), (-1, 2), (-1, 0), (-1, -2), (1, -2), (1, 0)]
        weights = [3, 1, 1, 3, 1, 1, 3]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints, dtype="float64")
        curve.weights = np.array(weights, dtype="float64")

        newcurve = curve.deepcopy()
        newcurve.knot_insert([0.25, 0.75])

        nodes_sample = np.linspace(0, 1, 129)
        points_old = curve(nodes_sample)
        points_new = newcurve(nodes_sample)
        for oldpt, newpt in zip(points_old, points_new):
            assert abs(np.linalg.norm(oldpt - newpt)) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestInsKnotCircle::test_begin",
            "TestInsKnotCircle::test_quarter_circle_standard",
            "TestInsKnotCircle::test_quarter_circle_symmetric",
            "TestInsKnotCircle::test_half_circle",
            "TestInsKnotCircle::test_full_circle",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(6)
@pytest.mark.dependency(
    depends=["test_begin", "TestCircle::test_end", "TestInsKnotCircle::test_end"]
)
def test_end():
    pass
