from compmec.nurbs.knotspace import KnotVector
from compmec.nurbs.basefunctions import BaseFunction
from compmec.nurbs.knotspace import KnotVector
from typing import List, Optional, Dict
import numpy as np


def getMultiplicities(U: KnotVector) -> Dict:
    s = {}
    for i, ui in enumerate(U):
        if ui not in s:
            s[ui] = 0
        s[ui] += 1
    return s


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
    n, p = F.n, F.p
    Pw = np.copy(P)
    r = U.index(knot)
    s = getMultiplicities(U)
    if knot not in s.keys():
        raise ValueError("The given knot is not in the KnotVector")
    t, newU, newP = RemoveCurveKnot(n, p, U, Pw, knot, r, s[knot], times)
    if t == 0:
        newP = newP.tolist()
        newP.pop(r-p+1)
        newP = np.array(newP)
    else:
        raise ValueError("Cannot knot what happens here. Needs testing")
    return np.array(newP)

def Distance4D(P1, P2):
    return np.linalg.norm(np.array(P1)-np.array(P2))

def RemoveCurveKnot(n, p, U, Pw, u, r, s, num):
    """
    Algorith A5.8 from Nurbs Book at page 185
    Remove knot u (index r) num times.
    Input: n, p, U, Pw, u, r, s, num
    Output: t, nw knots & ctrl pts in U & Pw
    s is the multiplicity of of knot u
    """
    Pw = np.array(Pw)
    TOLERANCE = 1e-7
    m = n + p + 1
    ord = p+1
    fout = (2*r-s-p)/2
    last = r-s
    first = r-p
    temp = np.zeros(Pw.shape)
    for t in range(num):  # This loop is Eq. (5.28)
        off = first - 1  # Diff in index between temp and P
        temp[0] = Pw[off]
        temp[last+1-off] = Pw[last+1]
        i = first
        j = last
        ii = 1
        jj = last - off
        remflag = 0
        while j - i > t:  # Compute new control points for one removal step
            alfi = (u-U[i])/(U[i+ord+t]-U[i])
            alfj = (u-U[j-t])/(U[j+ord]-U[j-t])
            temp[ii] = (Pw[i] - (1-alfi)*temp[ii-1])/alfi
            temp[jj] = (Pw[j] - alfj*temp[jj+1])/(1-alfj)
            i += 1
            ii += 1
            j -= 1
            jj -= 1
        if j - i < t:  # Check if knot removable
            if Distance4D(temp[ii-1], temp[jj+1]) <= TOLERANCE:
                remflag = 1
        else:
            alfi = (u-U[i])/(U[i+ord+t]-U[i])
            if Distance4D(Pw[i], alfi*temp[ii+t+1] + (1-alfi)*temp[ii-1]) <= TOLERANCE:
                remflag = 1
        if remflag == 0:  # Cannot remove any more knots
            break
        else:  # Successful removal. Save new cont. pts.
            i = first
            j = last
            while j - i > t:
                Pw[i] = temp[i-off]
                Pw[j] = temp[j-off]
                i += 1
                j -= 1
        first -= 1
        last += 1
    if t == 0:
        return t, U, Pw
    for k in range(r+1, m+1):
        U[k-t] = U[k]  # Shift knots
    j = fout
    i = j  # Pj thru Pi will be overwritten
    for k in range(1, t):
        if k % 2 == 1:  # k modulo 2
            i += 1
        else:
            j -= 1
    for k in range(i+1, n+1):  # Shift
        Pw[j] = Pw[k]
        j += 1
    return t, U, Pw

