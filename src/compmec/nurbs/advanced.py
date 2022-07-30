from compmec.nurbs.knotspace import KnotVector
from compmec.nurbs.basefunctions import BaseFunction, SplineBaseFunction
from compmec.nurbs.curves import BaseCurve
from compmec.nurbs.knotspace import KnotVector
from typing import List, Optional
import numpy as np
from geomdl import BSpline
from geomdl.operations import remove_knot



def insert_knot_basefunction(F: BaseFunction, knot: float, times: Optional[int] = 1) -> BaseFunction:
    """
    Receives a base function (like N) and returns a new basefuncion such that has the 
    """
    if not isinstance(F, BaseFunction):
        raise TypeError("F must be a BaseFunction instance")
    knotvector = F.U
    spot = knotvector.spot(knot)
    newU = list(knotvector)
    for i in range(times):
        newU.insert(spot+1, knot)
    return F.__class__(newU)


def insert_knot_controlpoints(F: BaseFunction, P: np.ndarray, knot: float, times: Optional[int]=1) -> np.ndarray:
    """
    Receives a curve and returns the controlpoints of the new curve
        * the new base function and the control points
    """
    if not isinstance(F, BaseFunction):
        raise TypeError("F must be a BaseFunction instance")
    U = KnotVector(F.U)
    if not isinstance(U, KnotVector):
        raise TypeError(f"Curve must be a KnotVector instance. Not {type(U)}")
    if times != 1:
        raise ValueError("For the moment, can insert knot only once at the time")
    n, p = U.n, U.p
    k = U.spot(knot)
    a = np.zeros(p)
    for j in range(p):
        i = k - p + 1 + j
        a[j] = (knot - U[i])/(U[i+p] - U[i])
    Q = np.zeros((n+1, P.shape[1]))
    for i in range(k-p+1): # Eq 5.5 at pag 142
        Q[i] = P[i]
    for i in range(k, n):
        Q[i+1] = P[i]
    for i in range(k-p+1, k+1):
        j = i - k + p - 1
        Q[i] = a[j]*P[i] + (1-a[j])*P[i-1]
    return Q

def remove_knot_basefunction(F: BaseFunction, knot: float, times: Optional[int] = 1) -> BaseFunction:
    if not isinstance(F, BaseFunction):
        raise TypeError("F must be a BaseFunction instance")
    U = list(F.U)
    if knot not in U:
        raise ValueError("The knot {knot} is not inside knot vector {U}")
    if times != 1:
        raise ValueError("Can only remove one knot at the time")
    U.remove(knot)
    newF = F.__class__(U)
    return newF

def remove_knot_controlpoints(F: BaseFunction, P: np.ndarray, knot: float, times: Optional[int]=1) -> np.ndarray:
    if not isinstance(F, BaseFunction):
        raise TypeError("F must be a BaseFunction instance")
    U = list(F.U)
    curve = BSpline.Curve()
    curve.degree = F.p
    Ps = []
    for i, Pi in enumerate(P):
        Ps.append(list(Pi))
    curve.ctrlpts = Ps
    curve.knotvector = list(U)
    remove_knot(curve, [knot], [times])
    return np.array(curve.ctrlpts)