"""
This file is responsible to testing the code inside the file ```calculus.py```
Its functions are getting derivatives, computing integrals along curves and so on
"""

import numpy as np
import pytest

from pynurbs.curves.curves import Curve
from pynurbs.curves.intersection import bcurve_and_bcurve, curve_and_curve


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
def test_bcurve_and_bcurve():
    beziera = Curve([0, 0, 1, 1])
    beziera.ctrlpoints = np.array([(0, 0), (1, 1)])
    bezierb = Curve([0, 0, 1, 1])
    bezierb.ctrlpoints = np.array([(0, 1), (1, 0)])
    inters = bcurve_and_bcurve(beziera, bezierb)

    assert len(inters) == 1
    np.testing.assert_allclose(inters[0], (0.5, 0.5))

    beziera.knot_insert([0.2])
    bezierb.knot_insert([0.7])
    inters = curve_and_curve(beziera, bezierb)

    assert len(inters) == 1
    np.testing.assert_allclose(inters[0], (0.5, 0.5))


@pytest.mark.order(42)
@pytest.mark.timeout(50)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_bcurve_and_bcurve",
    ]
)
def test_quarter_circles():
    knotvector = [0, 0, 0, 1, 1, 1]
    pointsa = [(1, 0), (1, 1), (0, 1)]
    pointsb = [(0, 0), (0, 1), (1, 1)]
    circlea = Curve(knotvector, np.array(pointsa))
    circleb = Curve(knotvector, np.array(pointsb))

    inters = bcurve_and_bcurve(circlea, circleb)
    assert len(inters) == 1
    root = 1 / np.sqrt(2)
    np.testing.assert_allclose(inters[0], (root, root))

    circlea.weights = (1, 1, 1)
    circleb.weights = (1, 1, 1)
    inters = bcurve_and_bcurve(circlea, circleb)
    assert len(inters) == 1
    np.testing.assert_allclose(inters[0], (root, root))

    circlea.weights = (1, 1, 2)
    circleb.weights = (1, 1, 2)
    inters = bcurve_and_bcurve(circlea, circleb)
    assert len(inters) == 1
    root = 1 / np.sqrt(3)
    np.testing.assert_allclose(inters[0], (root, root))


@pytest.mark.order(42)
@pytest.mark.timeout(50)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_bcurve_and_bcurve",
        "test_quarter_circles",
    ]
)
def test_half_circles():
    knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    weights = [3, 1, 1, 3]
    pointsa = [(1, 0), (1, 2), (-1, 2), (-1, 0)]
    pointsb = [(0, 0), (0, 2), (2, 2), (2, 0)]
    circlea = Curve(knotvector, np.array(pointsa), weights)
    circleb = Curve(knotvector, np.array(pointsb), weights)

    inters = bcurve_and_bcurve(circlea, circleb)
    assert len(inters) == 1
    root = (np.sqrt(3) - 1) / 2
    np.testing.assert_allclose(inters[0], (root, root))


@pytest.mark.order(42)
@pytest.mark.timeout(50)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_bcurve_and_bcurve",
        "test_quarter_circles",
        "test_half_circles",
    ]
)
def test_circle_and_circle():
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

    inters = curve_and_curve(circlea, circleb)
    for ua, ub in inters:
        pointa = circlea(ua)
        pointb = circleb(ub)
        distance = np.abs(pointa - pointb)
        assert np.all(distance < 1e-9)


@pytest.mark.order(42)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_bcurve_and_bcurve",
        "test_quarter_circles",
        "test_half_circles",
        "test_circle_and_circle",
    ]
)
def test_end():
    pass
