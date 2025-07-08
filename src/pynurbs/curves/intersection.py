"""
File that contains the algorithms to compute the
intersection between two curves
"""

from typing import Any, Tuple

import numpy as np

from ..operations import heavy
from ..operations.calculus import Derivate
from .curves import Curve


def _inse_retangle_float(avals: Tuple[float], bvals: Tuple[float]) -> bool:
    """
    Given two array of floats, if verifies if the region
        [min(avals), max(avals)] cap [min(bvals), max(bvals)]
    is not empty

    """
    mina, maxa = min(avals), max(avals)
    minb, maxb = min(bvals), max(bvals)
    avals = (mina, (mina + maxa) / 2, maxa)
    bvals = (minb, (minb + maxb) / 2, maxb)
    for aval in avals:
        if (aval - minb) * (aval - maxb) < 0:
            return True
    for bval in bvals:
        if (bval - mina) * (bval - maxa) < 0:
            return True
    return False


def _inse_retangle(ctrlptsa: Tuple[Any], ctrlptsb: Tuple[Any]) -> bool:
    """Given two curves A(u) and B(t), we test if the rectangular
    region made by points A intersects the retangular region
    made by points of B.

    - If A control points are scalars, it verifies if the region

    """
    try:
        nsuba = len(ctrlptsa[0])
        assert nsuba == len(ctrlptsb[0])
        for i in range(nsuba):
            valsa = [pt[i] for pt in ctrlptsa]
            valsb = [pt[i] for pt in ctrlptsb]
            inside = _inse_retangle(valsa, valsb)
            if not inside:
                return False
        return True
    except TypeError:
        return _inse_retangle_float(ctrlptsa, ctrlptsb)


def filter_pairs(pairs: Tuple[Tuple[float]], tolerance: float = 1e-9):
    """Filter the repeted knots within a given tolerance"""
    pairs = np.array(pairs, dtype="float64")
    filteredpairs = []
    for pair in pairs:
        inside = False
        for filtpair in filteredpairs:
            if np.linalg.norm(pair - filtpair) < tolerance:
                inside = True
        if not inside:
            filteredpairs.append(pair)
    filteredpairs = tuple(map(tuple, filteredpairs))
    return filteredpairs


def pairs_min_distance(
    pairs: Tuple[float],
    curvea: Curve,
    curveb: Curve,
    tolerance: float = 1e-9,
):
    """
    Filter the pairs (t*, u*) such abs(curvea(t*) - curveb(u*)) > tolerance
    """
    pairs = heavy.totuple(pairs)
    distances = np.empty(len(pairs), dtype="float64")
    for k, (pti, puj) in enumerate(pairs):
        pointati = curvea.eval(pti)
        pointbuj = curveb.eval(puj)
        distances[k] = np.linalg.norm(pointati - pointbuj)
    distances = np.abs(distances)
    matchs = np.abs(distances - np.min(distances)) < tolerance
    pairs = np.array(pairs, dtype="float64")[matchs]
    return heavy.totuple(pairs)


def __newton_bcurve_and_bcurve(
    pair: Tuple[float],
    curvesa: Tuple[Curve],
    curvesb: Tuple[Curve],
    limits: Tuple[float],
):
    """
    Uses newton iterations to get the intersection
    between two bezier curves.

    We supose pair is inside limits
    """
    tmin, tmax = limits[0]
    umin, umax = limits[1]
    for _ in range(10):
        diff = curvesa[0].eval(pair[0])
        dati = curvesa[1].eval(pair[0])
        ddati = curvesa[2].eval(pair[0])
        diff -= curvesb[0].eval(pair[1])
        dbuj = curvesb[1].eval(pair[1])
        ddbuj = curvesb[2].eval(pair[1])
        grad = np.array([np.inner(dati, diff), -np.inner(dbuj, diff)])
        ggrad = np.zeros((2, 2), dtype="float64")
        ggrad[0, 0] = np.inner(ddati, diff)
        ggrad[0, 0] += np.linalg.norm(dati) ** 2
        ggrad[1, 1] = -np.inner(ddbuj, diff)
        ggrad[1, 1] += np.linalg.norm(dbuj) ** 2
        ggrad[0, 1] = -np.inner(dati, dbuj)
        ggrad[1, 0] = ggrad[0, 1]
        denom = np.linalg.det(ggrad)
        if np.abs(denom) < 1e-9:
            return tuple()  # no convergence
        deltapair = np.linalg.solve(ggrad, grad)
        pair -= deltapair
        if pair[0] < tmin:
            pair[0] = tmin
        elif tmax < pair[0]:
            pair[0] = tmax
        if pair[1] < umin:
            pair[1] = umin
        elif umax < pair[1]:
            pair[1] = umax
        if np.linalg.norm(deltapair) < 1e-9:
            return tuple(pair)  # convergence
    return tuple(pair)


def bcurve_and_bcurve(beziera: Curve, bezierb: Curve) -> Tuple[float, float]:
    """Return the parameters t*, u* such beziera(t*) = bezierb(u*)

    Given two bezier curves, A(t) and B(u), this function returns the
    intersections between A and B. It can be:

    - If A(t) don't touch B(u), returns empty tuple
    - If A(t) touches B(u) in a finite number of points, it returns
        the pairs [(ta, ua), (tb, ub), ..., (tk, uk)]
    - If A(t) overlaps B(u) in some interval, it returns
        The interval [(ta, tb), (ua, ub)]
        Still needs implementation

    """
    assert isinstance(beziera, Curve)
    assert isinstance(bezierb, Curve)
    assert beziera.degree + 1 == beziera.npts
    assert beziera.degree + 1 == beziera.npts
    if not _inse_retangle(beziera.ctrlpoints, bezierb.ctrlpoints):
        return tuple()

    curvesa = [beziera]
    curvesa.append(Derivate(curvesa[0]))
    curvesa.append(Derivate(curvesa[1]))
    curvesb = [bezierb]
    curvesb.append(Derivate(curvesb[0]))
    curvesb.append(Derivate(curvesb[1]))
    dega, degb = beziera.degree, bezierb.degree
    nsma, nsmb = dega + 1, degb + 1  # Number of samples
    uamin, uamax = beziera.knotvector.limits
    ubmin, ubmax = bezierb.knotvector.limits
    limits = ((uamin, uamax), (ubmin, ubmax))
    nodes_a_sample = (
        [0] + [(2 * i + 1) / (2 * nsma) for i in range(nsma)] + [1]
    )
    nodes_b_sample = (
        [0] + [(2 * i + 1) / (2 * nsmb) for i in range(nsmb)] + [1]
    )
    uasample = [uamin + (uamax - uamin) * node for node in nodes_a_sample]
    ubsample = [ubmin + (ubmax - ubmin) * node for node in nodes_b_sample]
    pairs = set()
    for nodea in uasample:
        for nodeb in ubsample:
            # Newton's iteration
            pair = np.array((nodea, nodeb), dtype="float64")
            pair = __newton_bcurve_and_bcurve(pair, curvesa, curvesb, limits)
            if len(pair) != 0:
                pairs |= set((pair,))
    if len(pairs) == 0:
        return tuple()
    pairs = tuple(pairs)
    pairs = filter_pairs(pairs)
    pairs = pairs_min_distance(pairs, curvesa[0], curvesb[0])
    return heavy.totuple(pairs)


def curve_and_curve(curvea: Curve, curveb: Curve) -> Tuple[Curve]:
    """Return the parameters t*, u* such curvea(t*) = curveb(u*)

    Given two curves, A(t) and B(u), this function returns the
    intersections between A and B. It can be:

    - If A(t) don't touch B(u), returns empty tuple
    - If A(t) touches B(u) in a finite number of points, it returns
        the pairs [(ta, ua), (tb, ub), ..., (tk, uk)]
    - If A(t) overlaps B(u) in some interval, it returns
        The interval [(ta, tb), (ua, ub)]

    """
    beziersa = curvea.split()
    beziersb = curveb.split()
    for bez in beziersa:
        bez.clean()
    for bez in beziersb:
        bez.clean()
    pairs = set()
    for beziera in beziersa:
        for bezierb in beziersb:
            newpair = bcurve_and_bcurve(beziera, bezierb)
            pairs |= set(newpair)
    pairs = tuple(pairs)
    pairs = filter_pairs(pairs)
    pairs = pairs_min_distance(pairs, curvea, curveb)
    return pairs
