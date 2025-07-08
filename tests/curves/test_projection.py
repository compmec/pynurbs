"""
This file is responsible to testing the code inside the file ```calculus.py```
Its functions are getting derivatives, computing integrals along curves and so on
"""

import numpy as np
import pytest

from pynurbs.curves.curves import Curve
from pynurbs.curves.projection import point_on_curve


@pytest.mark.order(42)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_basis_functions.py::test_end",
        "tests/curves/test_bezier.py::test_end",
        "tests/curves/test_spline.py::test_end",
        "tests/operations/test_calculus.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(42)
@pytest.mark.timeout(4)
@pytest.mark.dependency(depends=["test_begin"])
def test_point_on_curve():
    knotvector = [0, 0, 1, 1]
    points = [(0, 0), (1, 0)]
    curve = Curve(knotvector)
    curve.ctrlpoints = np.array(points, dtype="float64")

    project = lambda point: point_on_curve(point, curve)
    np.testing.assert_allclose(project((0, 0)), (0,))
    np.testing.assert_allclose(project((1, 0)), (1,))
    np.testing.assert_allclose(project((0.5, 0)), (0.5,))

    knotvector = [0, 0, 1, 2, 2]
    points = [(1, -1), (0, 0), (1, 1)]
    curve = Curve(knotvector)
    curve.ctrlpoints = np.array(points, dtype="float64")

    project = lambda point: point_on_curve(point, curve)
    np.testing.assert_allclose(project((1, -1)), (0,))
    np.testing.assert_allclose(project((0, 0)), (1,))
    np.testing.assert_allclose(project((1, 1)), (2,))
    np.testing.assert_allclose(project((1, 0)), (0.5, 1.5))

    knotvector = [0, 0, 1, 2, 3, 4, 4]
    points = [(1, -2), (1, -1), (0, 0), (1, 1), (1, 2)]
    curve = Curve(knotvector)
    curve.ctrlpoints = np.array(points, dtype="float64")

    project = lambda point: point_on_curve(point, curve)
    np.testing.assert_allclose(project((1, -2)), (0,))
    np.testing.assert_allclose(project((1, -1)), (1,))
    np.testing.assert_allclose(project((0, 0)), (2,))
    np.testing.assert_allclose(project((1, 1)), (3,))
    np.testing.assert_allclose(project((1, 2)), (4,))
    np.testing.assert_allclose(project((1, 0)), (1.5, 2.5))


@pytest.mark.order(42)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_point_on_curve",
    ]
)
def test_end():
    pass
