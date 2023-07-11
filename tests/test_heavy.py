import numpy as np
import pytest

from compmec.nurbs.heavy import LeastSquare


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
    pass


class TestLeastSquare:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin"])
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
        chebynodes = LeastSquare.chebyshev_nodes(npts)
        fvals = [func(xi) for xi in chebynodes]
        integrator = LeastSquare.integrator_array(npts)
        numeintegral = integrator @ fvals
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
@pytest.mark.dependency(depends=["TestLeastSquare::test_end"])
def test_end():
    pass
