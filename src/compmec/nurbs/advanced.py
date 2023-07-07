"""
This file contains Advanced Geometric Algorithms
In Nurbs book, it correspond to chapter 6
"""
from typing import Tuple

import numpy as np

from compmec.nurbs.calculus import derivate_curve
from compmec.nurbs.curves import Curve


def find_projection_point_on_curve(point: Tuple[float], curve: Curve) -> Tuple[float]:
    """
    This function finds the parameter tstar in [tmin, tmax] such
    minimizes the distance abs(curve(tstar) - point).
    Trully, it minimizes the distance square, related to the inner
    product < C(u) - P, C(u) - P > = abs(C(u)-P)^2
    This function finds the solution of
        f(u) = < C'(u), C(u) - P > = 0
    Since it's possible to have more than one solution:
        for example, the center of a circle is at equal distance always
    then we return a list of parameters

    First, we decompose the curve in beziers, and try to find
    the minimum distance of each bezier curve.
    We use Newton's method
    """
    point = np.array(point)
    beziers = curve.split()
    tvalues = set()
    for bez in beziers:
        umin, umax = bez.knots[0], bez.knots[-1]
        dbez = derivate_curve(bez)
        ddbez = derivate_curve(dbez)
        tparams = np.linspace(umin, umax, 5)
        for tparam in tparams:
            newt = newtons(point, bez, dbez, ddbez, tparam)
            tvalues |= set(newt)
    tvalues = tuple(tvalues)
    tvalues = np.array(tvalues)
    distances = [np.linalg.norm(point - curve(t)) for t in tvalues]
    minimaldistance = np.min(distances)
    indexs = np.where(abs(distances - minimaldistance) < 1e-6)[0]
    tvalues = tvalues[indexs]
    tvalues.sort()
    return tuple(tvalues)


def newtons(
    point: Tuple[float], bez: Curve, dbez: Curve, ddbez: Curve, initparam: float
) -> float:
    tolerance1 = 1e-6
    umin, umax = bez.knots[0], bez.knots[-1]
    niter = 0
    while True:
        bezui = bez(initparam) - point
        dbezui = dbez(initparam)
        ddbezui = ddbez(initparam)
        upper = np.inner(dbezui, bezui)
        lower = np.inner(ddbezui, bezui)
        lower += np.inner(dbezui, dbezui)
        diff = upper / lower
        initparam -= diff
        if initparam < umin:
            return (umin,)
        if initparam > umax:
            return (umax,)
        if np.abs(diff) < tolerance1:
            return [initparam]
        niter += 1
