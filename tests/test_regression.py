import numpy as np
import pytest
from numpy import linalg as la

from compmec.nurbs.approx import curve_spline


@pytest.mark.order(4)
@pytest.mark.dependency(depends=["tests/test_curve.py::test_end"], scope="session")
def test_begin():
    pass


@pytest.mark.order(4)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_curvecossin():
    a, b = 0, 2 * np.pi
    fx = np.cos
    fy = np.sin

    p, n = 2, 15
    nsample = 101
    nplot = 1024
    theta = np.linspace(a, b, nsample)
    x = fx(theta)
    y = fy(theta)
    ubar = (theta - min(theta)) / (max(theta) - min(theta))
    xplot = np.linspace(a, b, 1025)
    thetaplot = np.linspace(a, b, nplot)
    uplot = (thetaplot - min(thetaplot)) / (max(thetaplot) - min(thetaplot))
    xplot = fx(thetaplot)
    yplot = fy(thetaplot)

    F = curve_spline(ubar, (x, y), p, n)
    v = F(uplot)
    xsuposed = v[:, 0]
    ysuposed = v[:, 1]

    L2x = la.norm(xsuposed - xplot)
    L2y = la.norm(ysuposed - yplot)
    # assert L2x < 1
    # assert L2y < 1

    print("Error L2 x = %.3f" % L2x)
    print("Error L2 y = %.3f" % L2y)


def main():
    test_curvecossin()


if __name__ == "__main__":
    main()
