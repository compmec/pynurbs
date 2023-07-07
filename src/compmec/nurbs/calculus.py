from typing import Optional, Tuple

import numpy as np

from compmec.nurbs import heavy
from compmec.nurbs.curves import Curve
from compmec.nurbs.functions import Function
from compmec.nurbs.knotspace import KnotVector


def knotvector_derivated(knotvector: Tuple[float]) -> Tuple[float]:
    degree = heavy.KnotVector.find_degree(knotvector)
    knots = heavy.KnotVector.find_knots(knotvector)
    for knot in knots:
        mult = heavy.KnotVector.find_mult(knot, knotvector)
        if mult == degree + 1:
            knotvector = heavy.KnotVector.remove_knots(knotvector, [knot])
    return knotvector


def difference_vector(knotvector: Tuple[float]) -> Tuple[float]:
    degree = heavy.KnotVector.find_degree(knotvector)
    npts = heavy.KnotVector.find_npts(knotvector)
    avals = np.zeros(npts, dtype="float64")
    for i in range(npts):
        diff = knotvector[i + degree] - knotvector[i]
        if diff != 0:
            avals[i] = degree / diff
    return avals


def difference_matrix(knotvector: Tuple[float]) -> np.ndarray:
    avals = difference_vector(knotvector)
    npts = len(avals)
    matrix = np.diag(avals)
    for i in range(npts - 1):
        matrix[i, i + 1] = -avals[i + 1]
    return matrix


def derivate_curve(curve: Curve) -> Curve:
    if curve.degree == 0:
        knots = curve.knots
        newknotvector = [knots[0], knots[-1]]
        anypoint = curve.ctrlpoints[0]
        newctrlpoints = [0 * anypoint]
        return Curve(newknotvector, newctrlpoints)
    knotvector = tuple(curve.knotvector)
    matrix = difference_matrix(knotvector)

    newvector = knotvector_derivated(knotvector)
    newcurve = Curve(newvector)
    newctrlpoints = np.transpose(matrix) @ curve.ctrlpoints
    newcurve.ctrlpoints = newctrlpoints[1:]
    return newcurve
