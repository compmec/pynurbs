import numpy as np
import pytest

from pynurbs.operations.least_square import spline2spline


@pytest.mark.order(21)
@pytest.mark.dependency(
    depends=["tests/core/test_custom_math.py::test_end"], scope="session"
)
def test_begin():
    pass


@pytest.mark.order(21)
@pytest.mark.dependency(depends=["test_begin"])
def test_leastsquarespline_identity():
    U0 = [0, 0, 1, 1]
    U1 = [0, 0, 1, 1]
    T, E = spline2spline(U0, U1)
    np.testing.assert_almost_equal(T, np.eye(2))
    assert np.all(np.abs(E) < 1e-9)

    U0 = [0, 0, 0, 1, 1, 1]
    U1 = [0, 0, 0, 1, 1, 1]
    T, E = spline2spline(U0, U1)
    np.testing.assert_almost_equal(T, np.eye(3))
    assert np.all(np.abs(E) < 1e-9)

    U0 = [0, 0, 0, 0.5, 1, 1, 1]
    U1 = [0, 0, 0, 0.5, 1, 1, 1]
    T, E = spline2spline(U0, U1)
    np.testing.assert_almost_equal(T, np.eye(4))
    assert np.all(np.abs(E) < 1e-9)


@pytest.mark.order(21)
@pytest.mark.dependency(depends=["test_begin"])
def test_leastsquarespline_eval_error():
    # knot insertion
    U0 = [0, 0, 0, 1, 1, 1]
    U1 = [0, 0, 0, 0.5, 1, 1, 1]
    _, E = spline2spline(U0, U1)
    assert np.all(np.abs(E) < 1e-9)

    # degree elevate
    U0 = [0, 0, 1, 1]
    U1 = [0, 0, 0, 1, 1, 1]
    _, E = spline2spline(U0, U1)
    assert np.all(np.abs(E) < 1e-9)


@pytest.mark.order(21)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_leastsquarespline_identity",
        "test_leastsquarespline_eval_error",
    ]
)
def test_end():
    pass
