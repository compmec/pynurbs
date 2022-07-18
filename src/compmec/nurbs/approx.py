import numpy as np
from numpy import linalg as la
from typing import Iterable, Optional
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs.curves import SplineCurve

def transform2U(u: Iterable[float], n: float, p: float, algorithm: int = 1):
    U = np.zeros(n+p+1)
    if algorithm == 1:
        for j in range(1, n-p):
            U[j+p] = np.sum(u[j:j+p])/p
    else:
        ValueError("Algorithm is nod valid!")
    U[n:] = 1
    return U

def curve_control_points(L: np.ndarray, points: np.ndarray):
    """
    t is a vector of the position of each point.
        path(t[i]) = points[i]
    points is a 2D numpy matrix of shape (npts, dim)
    """
    n = L.shape[0]
    points = np.array(points).T
    npts, dim = points.shape
    controlpts = np.zeros((n, dim))
    for i in range(dim):
        controlpts[:, i] = compute_control_points(L, points[:, i])
    return controlpts

def curve_spline(t: Iterable[float], points: np.ndarray, p: Optional[int] = None, n: Optional[int] = None):
    t = (t - min(t))/(max(t) - min(t))  # Normalize
    U = transform2U(t, n, p)
    N = SplineBaseFunction(U)
    L = N(t)
    controlpts = curve_control_points(L, points)
    return SplineCurve(N, controlpts)


def function_control_points(x: Iterable[float], y: Iterable[float], p: Optional[int] = None, n: Optional[int] = None):
    ubar = (x-x[0])/(x[-1]-x[0])
    U = transform2U(ubar, n, p)
    N = SplineBaseFunction(U)
    L = N(ubar)
    controlpts = compute_control_points(L, y)
    return controlpts


def compute_control_points(L, y): 
    return la.solve(L @ L.T, L @ y)