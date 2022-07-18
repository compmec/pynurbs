import numpy as np
from numpy import iterable, linalg as la
from typing import Iterable, Optional
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs.curves import SplineCurve, SplineXYFunction

def transform2U(u: Iterable[float], n: float, p: float, algorithm: int = 1):
    U = np.zeros(n+p+1)
    if algorithm == 1:
        for j in range(1, n-p):
            U[j+p] = np.sum(u[j:j+p])/p
    else:
        ValueError("Algorithm is nod valid!")
    U[n:] = 1
    return U



def curve_spline(u: Iterable[float], points: np.ndarray, p: int, dpoints: Optional[np.ndarray] = None):
    n = len(u)
    U = transform2U(u, n, p)
    N = SplineBaseFunction(U)
    dN = N.derivate()
    L = N(u)
    dL = dN(u)
    dim = len(points[0])
    controlpts = curve_control_points(L, points)
    return SplineCurve(N, controlpts)


def function_spline(x: Iterable[float], y: Iterable[float], p: int):
    ubar = (np.array(x)-min(x))/(max(x)-min(x))
    n = len(ubar)
    U = transform2U(ubar, n, p)
    N = SplineBaseFunction(U)
    L = N(ubar)
    X = curve_control_points(L, x)
    Y = curve_control_points(L, y)
    return SplineXYFunction(N, X, Y)


def curve_control_points(L: np.ndarray, y: Iterable[float]):
    return la.solve(L.T, y)