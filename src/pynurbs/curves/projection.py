"""
File that contains the algorithms to find the nearest point/curve with respect
to another point/curve
"""

from typing import Tuple

import numpy as np

from ..operations.calculus import Derivate
from .curves import Curve


def __newton_point_on_curve(
    point: Tuple[float], curves: Tuple[Curve], initparam: float
) -> float:
    """
    Returns the parameter ui from newton's iteration
        u_{i+1} = u_{i} - f(u_{i})/f'(u_{i})
    The point is

    """
    tolerance1 = 1e-6
    umin, umax = curves[0].knotvector.limits
    niter = 0
    while True:
        bezui = curves[0](initparam) - point
        dbezui = curves[1](initparam)
        ddbezui = curves[2](initparam)
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


def point_on_bezier(point: Tuple[float], bezier: Curve) -> Tuple[float]:
    """Finds the parameters t* such
    bezier(t*) is the near point
    """
    umin, umax = bezier.knotvector.limits
    curves = [bezier]
    curves.append(Derivate(curves[0]))
    curves.append(Derivate(curves[1]))
    tparams = np.linspace(umin, umax, 5)
    tvalues = set()
    for tparam in tparams:
        newt = __newton_point_on_curve(point, curves, tparam)
        tvalues |= set(newt)
    return tuple(tvalues)


def point_on_curve(point: Tuple[float], curve: Curve) -> Tuple[float]:
    """Finds the parameters t* such curve(t*) is near point

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
    for bez in beziers:
        bez.clean()
    tvalues = set()
    for bezier in beziers:
        newtvalues = point_on_bezier(point, bezier)
        tvalues |= set(newtvalues)
    tvalues = tuple(tvalues)
    tvalues = np.array(tvalues)
    distances = [np.linalg.norm(point - curve(t)) for t in tvalues]
    minimaldistance = np.min(distances)
    indexs = np.where(abs(distances - minimaldistance) < 1e-6)[0]
    tvalues = tvalues[indexs]
    tvalues.sort()
    return tuple(tvalues)
