"""
This file is responsible to testing the code inside the file ```calculus.py```
Its functions are getting derivatives, computing integrals along curves and so on
"""

import numpy as np
import pytest

from pynurbs.advanced import Intersection, Projection
from pynurbs.curves import Curve


@pytest.mark.order(8)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_functions.py::test_end",
        "tests/test_beziercurve.py::test_end",
        "tests/test_splinecurve.py::test_end",
        "tests/test_calculus.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestProjection:
    @pytest.mark.order(8)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(8)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(depends=["TestProjection::test_begin"])
    def test_point_on_curve(self):
        knotvector = [0, 0, 1, 1]
        points = [(0, 0), (1, 0)]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(points, dtype="float64")

        project = lambda point: Projection.point_on_curve(point, curve)
        np.testing.assert_allclose(project((0, 0)), (0,))
        np.testing.assert_allclose(project((1, 0)), (1,))
        np.testing.assert_allclose(project((0.5, 0)), (0.5,))

        knotvector = [0, 0, 1, 2, 2]
        points = [(1, -1), (0, 0), (1, 1)]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(points, dtype="float64")

        project = lambda point: Projection.point_on_curve(point, curve)
        np.testing.assert_allclose(project((1, -1)), (0,))
        np.testing.assert_allclose(project((0, 0)), (1,))
        np.testing.assert_allclose(project((1, 1)), (2,))
        np.testing.assert_allclose(project((1, 0)), (0.5, 1.5))

        knotvector = [0, 0, 1, 2, 3, 4, 4]
        points = [(1, -2), (1, -1), (0, 0), (1, 1), (1, 2)]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(points, dtype="float64")

        project = lambda point: Projection.point_on_curve(point, curve)
        np.testing.assert_allclose(project((1, -2)), (0,))
        np.testing.assert_allclose(project((1, -1)), (1,))
        np.testing.assert_allclose(project((0, 0)), (2,))
        np.testing.assert_allclose(project((1, 1)), (3,))
        np.testing.assert_allclose(project((1, 2)), (4,))
        np.testing.assert_allclose(project((1, 0)), (1.5, 2.5))

    @pytest.mark.order(8)
    @pytest.mark.dependency(
        depends=["TestProjection::test_begin", "TestProjection::test_point_on_curve"]
    )
    def test_end(self):
        pass


class TestIntersection:
    @pytest.mark.order(8)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(8)
    @pytest.mark.timeout(4)
    @pytest.mark.dependency(depends=["TestIntersection::test_begin"])
    def test_bcurve_and_bcurve(self):
        beziera = Curve([0, 0, 1, 1])
        beziera.ctrlpoints = np.array([(0, 0), (1, 1)])
        bezierb = Curve([0, 0, 1, 1])
        bezierb.ctrlpoints = np.array([(0, 1), (1, 0)])
        inters = Intersection.bcurve_and_bcurve(beziera, bezierb)

        assert len(inters) == 1
        np.testing.assert_allclose(inters[0], (0.5, 0.5))

        beziera.knot_insert([0.2])
        bezierb.knot_insert([0.7])
        inters = Intersection.curve_and_curve(beziera, bezierb)

        assert len(inters) == 1
        np.testing.assert_allclose(inters[0], (0.5, 0.5))

    @pytest.mark.order(8)
    @pytest.mark.timeout(50)
    @pytest.mark.dependency(
        depends=[
            "TestIntersection::test_begin",
            "TestIntersection::test_bcurve_and_bcurve",
        ]
    )
    def test_quarter_circles(self):
        knotvector = [0, 0, 0, 1, 1, 1]
        pointsa = [(1, 0), (1, 1), (0, 1)]
        pointsb = [(0, 0), (0, 1), (1, 1)]
        circlea = Curve(knotvector, np.array(pointsa))
        circleb = Curve(knotvector, np.array(pointsb))

        inters = Intersection.bcurve_and_bcurve(circlea, circleb)
        assert len(inters) == 1
        root = 1 / np.sqrt(2)
        np.testing.assert_allclose(inters[0], (root, root))

        circlea.weights = (1, 1, 1)
        circleb.weights = (1, 1, 1)
        inters = Intersection.bcurve_and_bcurve(circlea, circleb)
        assert len(inters) == 1
        np.testing.assert_allclose(inters[0], (root, root))

        circlea.weights = (1, 1, 2)
        circleb.weights = (1, 1, 2)
        inters = Intersection.bcurve_and_bcurve(circlea, circleb)
        assert len(inters) == 1
        root = 1 / np.sqrt(3)
        np.testing.assert_allclose(inters[0], (root, root))

    @pytest.mark.order(8)
    @pytest.mark.timeout(50)
    @pytest.mark.dependency(
        depends=[
            "TestIntersection::test_begin",
            "TestIntersection::test_bcurve_and_bcurve",
            "TestIntersection::test_quarter_circles",
        ]
    )
    def test_half_circles(self):
        knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
        weights = [3, 1, 1, 3]
        pointsa = [(1, 0), (1, 2), (-1, 2), (-1, 0)]
        pointsb = [(0, 0), (0, 2), (2, 2), (2, 0)]
        circlea = Curve(knotvector, np.array(pointsa), weights)
        circleb = Curve(knotvector, np.array(pointsb), weights)

        inters = Intersection.bcurve_and_bcurve(circlea, circleb)
        assert len(inters) == 1
        root = (np.sqrt(3) - 1) / 2
        np.testing.assert_allclose(inters[0], (root, root))

    @pytest.mark.order(8)
    @pytest.mark.timeout(50)
    @pytest.mark.dependency(
        depends=[
            "TestIntersection::test_begin",
            "TestIntersection::test_bcurve_and_bcurve",
            "TestIntersection::test_quarter_circles",
            "TestIntersection::test_half_circles",
        ]
    )
    def test_circle_and_circle(self):
        knotvector = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]
        weights = [3, 1, 1, 3, 1, 1, 3]
        ctrlpointsa = [
            (1, 0),
            (1, 2),
            (-1, 2),
            (-1, 0),
            (-1, -2),
            (1, -2),
            (1, 0),
        ]
        ctrlpointsa = np.array(ctrlpointsa, dtype="float64")
        circlea = Curve(knotvector, ctrlpointsa, weights)

        ctrlpointsb = np.copy(ctrlpointsa)
        ctrlpointsb[:, 0] += 1
        circleb = Curve(knotvector, ctrlpointsb, weights)

        inters = Intersection.curve_and_curve(circlea, circleb)
        for ua, ub in inters:
            pointa = circlea(ua)
            pointb = circleb(ub)
            distance = np.abs(pointa - pointb)
            assert np.all(distance < 1e-9)

    @pytest.mark.order(8)
    @pytest.mark.dependency(
        depends=[
            "TestIntersection::test_begin",
            "TestIntersection::test_bcurve_and_bcurve",
            "TestIntersection::test_quarter_circles",
            "TestIntersection::test_half_circles",
            "TestIntersection::test_circle_and_circle",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(8)
@pytest.mark.dependency(
    depends=["test_begin", "TestProjection::test_end", "TestIntersection::test_end"]
)
def test_end():
    pass
