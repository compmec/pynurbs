from compmec.nurbs.knotspace import KnotVector
from compmec.nurbs.basefunctions import BaseFunction, BaseFunction
from compmec.nurbs.curves import BaseCurve
from compmec.nurbs.knotspace import KnotVector
from typing import List, Optional
import numpy as np



def insert_knot_basefunction(basefunction: BaseFunction, knot: float, times: Optional[int] = 1) -> BaseFunction:
    """
    Receives a base function (like N) and returns a new basefuncion such that has the 
    """
    knotvector = basefunction.U
    spot = knotvector.spot(knot)
    newU = list(knotvector)
    for i in range(times):
        newU.insert(spot+1, knot)
    return basefunction.__class__(newU)


def insert_knot_controlpoints(U: KnotVector, P: np.ndarray, knot: float, times: Optional[int]=1) -> np.ndarray:
    """
    Receives a curve and returns the controlpoints of the new curve
        * the new base function and the control points
    """
    U = KnotVector(U)
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

def remove_knot_basefunction(basefunction: BaseFunction, knot: float, times: Optional[int] = 1) -> BaseFunction:
    raise NotImplementedError("Stop")

def remove_knot_controlpoints(U: KnotVector, P: np.ndarray, knot: float, times: Optional[int]=1) -> np.ndarray:
    raise NotImplementedError("Stop")