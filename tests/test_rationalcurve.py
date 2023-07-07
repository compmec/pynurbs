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
    @pytest.mark.dependency(depends=["TestCircle::test_begin"])
    def test_failbuild(self):
        for degree in range(1, 6):
            npts = np.random.randint(degree + 1, degree + 9)
            knotvector = GeneratorKnotVector.random(degree, npts)
            curve = Curve(knotvector)
            with pytest.raises(ValueError):
                curve.weights = 1
            with pytest.raises(ValueError):
                curve.weights = -1 * np.ones(npts)
            with pytest.raises(ValueError):
                curve.weights = "asd"

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


class TestCircle:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
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
        for i, point in enumerate(points):
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


@pytest.mark.order(6)
@pytest.mark.dependency(depends=["test_begin", "TestCircle::test_end"])
def test_end():
    pass
