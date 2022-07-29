import numpy as np
from numpy import linalg as la
from typing import Iterable, Optional, Any
from compmec.nurbs import SplineBaseFunction, SplineCurve
from compmec.nurbs.curves import SplineXYFunction

def transform2U(u: Iterable[float], n: float, p: int, algorithm: int = 1):
    if p < 1:
        raise ValueError(f"p must be > 0. Received {p}")
    if n <= p:
        raise ValueError(f"({n} = n must be greater than p = {p}")
    U = np.zeros(n+p+1)
    if algorithm == 1:
        for j in range(1, n-p):
            U[j+p] = np.sum(u[j:j+p])/p
    else:
        ValueError("Algorithm is nod valid!")
    U[n:] = 1
    return U


def curve_spline(p: int, u: Iterable[Iterable[float]], points: Iterable[Iterable[Any]]):
    p = int(p)
    for i, ui in enumerate(u):
        u[i] = np.array(ui)
    for i, pi in enumerate(points):
        points[i] = np.array(pi)
    number_derivatives = len(u)-1
    ndofs = []
    for ui in u:
        ndofs.append(len(ui))
    totalndofs = np.sum(ndofs)
    if totalndofs <=  p:
        raise ValueError(f"The number of informations {totalndofs} very small to interpolate a curve with p = {p}")
    uequaly = np.linspace(0, 1, totalndofs)
    U = transform2U(uequaly, totalndofs, p)
    N = SplineBaseFunction(U)
    functions = [N]
    for i in range(number_derivatives):
        functions.append(functions[-1].derivate())

    Ls = []
    for ui, function in zip(u, functions):
        Ls.append(function(ui))

    dim = points[0].shape[1]
    M = np.zeros((totalndofs, totalndofs))
    B = np.zeros((totalndofs, dim))
    summe = 0
    for i, ndof in enumerate(ndofs):
        M[summe:summe+ndof, :] = Ls[i].T
        B[summe:summe+ndof, :] = points[i][:, :]
        summe += ndof
    controlpoints = np.linalg.solve(M, B)
    return SplineCurve(N, controlpoints)


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