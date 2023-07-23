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


def derivate_spline_curve(curve: Curve) -> Curve:
    assert curve.weights is None
    if curve.degree == 0:
        newknotvector = curve.knotvector.limits
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


def derivate_curve(curve: Curve) -> Curve:
    if curve.weights is None:
        return derivate_spline_curve(curve)
    numer = Curve(curve.knotvector)
    numer.ctrlpoints = [wi * pt for pt, wi in zip(curve.ctrlpoints, curve.weights)]
    denom = Curve(curve.knotvector)
    denom.ctrlpoints = curve.weights

    dnumer = derivate_spline_curve(numer)
    ddenom = derivate_spline_curve(denom)
    deriva = (dnumer * denom - numer * ddenom) / (ddenom * ddenom)
    deriva.clean()
    return deriva


class MathOperations:
    def add_spline(curvea: Curve, curveb: Curve, simplify: bool = True) -> Curve:
        knotvectc = curvea.knotvector | curveb.knotvector
        curveacopy = curvea.deepcopy()
        curveacopy.update_knotvector(knotvectc)
        curvebcopy = curveb.deepcopy()
        curvebcopy.update_knotvector(knotvectc)
        curvec = Curve(knotvectc)
        curvec.ctrlpoints = curveacopy.ctrlpoints + curvebcopy.ctrlpoints
        del curveacopy
        del curvebcopy
        if simplify:
            curvec.degree_clean()
            curvec.knot_clean()
        return curvec

    def mult_spline(curvea: Curve, curveb: Curve, simplify: bool = True) -> Curve:
        """
        Given two spline curves, called A(u) and B(u), it computes and returns
        a new curve C(u) such C(u) = A(u) * B(u) for every u
        Restrictions: The limits of B(u) must be the same as the limits of A(u)
        The parameter `simplify` shows if the function try to reduce at maximum
        the degree and the knots inside.
        """
        degreea, nptsa = curvea.degree, curvea.npts
        degreeb, nptsb = curveb.degree, curveb.npts
        allknots = (curvea.knotvector | curveb.knotvector).knots
        classes = np.zeros(len(allknots) - 2, dtype="int16")
        for i, knot in enumerate(allknots[1:-1]):
            multa = curvea.knotvector.mult(knot)
            multb = curveb.knotvector.mult(knot)
            classes[i + 1] = min(degreea - multa, degreeb - multb)
        degreec = degreea + degreeb
        knotvectorc = [curvea.knotvector[0]] * (degreec + 1)
        for knot, classe in zip(allknots[1:-1], classes):
            knotvectorc += [knot] * (degreec - classe)
        knotvectorc += [curvea.knotvector[-1]] * (degreec + 1)
        knotvectorc = KnotVector(knotvectorc)
        nptsc = knotvectorc.npts

        basisfunca = Function(curvea.knotvector)
        basisfuncb = Function(curveb.knotvector)
        basisfuncc = Function(knotvectorc)

        nptseval = degreec + 1
        nptstotal = nptseval * (len(allknots) - 1)
        avals = np.zeros((nptsa, nptstotal), dtype="float64")
        bvals = np.zeros((nptsb, nptstotal), dtype="float64")
        cvals = np.zeros((nptsc, nptstotal), dtype="float64")

        for i in range(len(allknots) - 1):
            start, end = allknots[i : i + 2]
            chebynodes = heavy.LeastSquare.chebyshev_nodes(nptseval, start, end)
            avals[i * nptseval : (i + 1) * nptseval] = basisfunca(chebynodes)
            bvals[i * nptseval : (i + 1) * nptseval] = basisfuncb(chebynodes)
            cvals[i * nptseval : (i + 1) * nptseval] = basisfuncc(chebynodes)
        matrix2d = np.einsum("il,jl->ij", cvals, cvals)
        invmatrix2d = np.linalg.inv(matrix2d)
        system3d = np.einsum("il,lz,jz,kz->ijk", invmatrix2d, cvals, avals, bvals)
        ctrlpointsc = np.empty(nptsc, dtype="object")
        for i, matrix in enumerate(system3d):
            ctrlpointsc[i] = curvea.ctrlpoints @ matrix @ curveb.ctrlpoints
        curvec = Curve(knotvectorc)
        curvec.ctrlpoints = ctrlpointsc
        if simplify:
            curvec.clean()
        return curvec

    def div_spline(curvea: Curve, curveb: Curve, simplify: bool = True) -> Curve:
        """
        Given two curves A(u) and B(u), it returns a curve C(u) such
        C(u) = A(u) / B(u) for every u
        Restrictions:
         - A(u) and B(u) must have the same limits for u
         - B(u) cannot have any root inside the limits of u
        """
        knotvectorc = curvea.knotvector | curveb.knotvector
        curveacopy = curvea.deepcopy()
        curvebcopy = curveb.deepcopy()
        curveacopy.update_knotvector(knotvectorc)
        curvebcopy.update_knotvector(knotvectorc)

        curveacopy.weights = curvebcopy.ctrlpoints
        if simplify:
            curveacopy.clean()
        return curveacopy

    def add_rationals(curvea: Curve, curveb: Curve, simplify: bool = True) -> Curve:
        if curvea.weights is None and curveb.weights is None:
            return MathOperations.add_spline(curvea, curveb, simplify)
        numa = curvea.deepcopy()
        numb = curveb.deepcopy()
        if curvea.weights is None:
            dena = Curve([0, 1], [1])
        else:
            numa.ctrlpoints = [
                pt * wi for pt, wi in zip(curvea.ctrlpoints, curvea.weights)
            ]
            numa.weights = None
            dena = Curve(curvea.knotvector)
            dena.ctrlpoints = curvea.weights
        if curveb.weights is None:
            denb = Curve([0, 1], [1])
        else:
            numb.ctrlpoints = [
                pt * wi for pt, wi in zip(curveb.ctrlpoints, curveb.weights)
            ]
            numb.weights = None
            denb = Curve(curveb.knotvector)
            denb.ctrlpoints = curveb.weights

        num_pt0 = MathOperations.mult_spline(numa, denb)
        num_pt1 = MathOperations.mult_spline(numb, dena)
        numerat = MathOperations.add_spline(num_pt0, num_pt1)
        denomin = MathOperations.mult_spline(dena, denb)
        result = MathOperations.div_spline(numerat, denomin)
        if simplify:
            result.clean()
        return result

    def mult_rationals(curvea: Curve, curveb: Curve, simplify: bool = True) -> Curve:
        if curvea.weights is None and curveb.weights is None:
            return MathOperations.div_spline(curvea, curveb, simplify)
        numa = curvea.deepcopy()
        if curvea.weights is None:
            dena = Curve([0, 1], [1])
        if curveb.weights is None:
            curveb.weights = [1] * curveb.npts
        numa = curvea.deepcopy()
        numa.ctrlpoints = tuple(
            [wi * pti for wi, pti in zip(curvea.weights, curvea.ctrlpoints)]
        )
        numa.weights = None
        numb = curveb.deepcopy()
        numb.ctrlpoints = tuple(
            [wi * pti for wi, pti in zip(curveb.weights, curveb.ctrlpoints)]
        )
        numb.weights = None
        dena = Curve(curvea.knotvector)
        dena.ctrlpoints = curvea.weights
        denb = Curve(curveb.knotvector)
        denb.ctrlpoints = curveb.weights

        multnumanumb = MathOperations.mult_spline(numa, numb)
        multdenadenb = MathOperations.mult_spline(dena, denb)
        result = MathOperations.div_spline(multnumanumb, multdenadenb)
        if simplify:
            result.clean()
        return result
