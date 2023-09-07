import numpy as np

from compmec.nurbs import heavy
from compmec.nurbs.curves import Curve


class Derivate:
    def __new__(cls, curve: Curve):
        return Derivate.curve(curve)

    @staticmethod
    def curve(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.ctrlpoints is not None
        if curve.degree == 0:
            limits = curve.knotvector.limits
            zero = 0 * curve.ctrlpoints[0]
            return curve.__class__(limits, [zero])
        if curve.degree + 1 == curve.npts:
            return Derivate.bezier(curve)
        return Derivate.spline(curve)

    @staticmethod
    def bezier(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.degree + 1 == curve.npts
        if curve.weights is None:
            return Derivate.nonrational_bezier(curve)
        return Derivate.rational_bezier(curve)

    @staticmethod
    def spline(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.degree + 1 != curve.npts
        if curve.weights is None:
            return Derivate.nonrational_spline(curve)
        return Derivate.rational_spline(curve)

    @staticmethod
    def nonrational_bezier(curve: Curve) -> Curve:
        """ """
        assert curve.degree + 1 == curve.npts
        assert curve.weights is None
        vector = tuple(curve.knotvector)
        matrix = heavy.Calculus.derivate_nonrational_bezier(vector)
        ctrlpoints = tuple(np.dot(matrix, curve.ctrlpoints))
        newcurve = curve.__class__(vector[1:-1], ctrlpoints)
        newcurve.clean()
        return newcurve

    @staticmethod
    def rational_bezier(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.degree + 1 == curve.npts
        assert curve.weights is not None
        assert np.all(np.array(curve.weights) != 0)

        knotvector = tuple(curve.knotvector)
        matrixup, matrixdo = heavy.Calculus.derivate_rational_bezier(knotvector)
        num, den = curve.fraction()
        matrixup = np.dot(matrixup, den.ctrlpoints)
        matrixdo = np.dot(matrixdo, den.ctrlpoints)

        dennumctrlpts = den.ctrlpoints @ matrixdo
        newnumctrlpts = np.dot(np.transpose(matrixup), num.ctrlpoints)
        newnumctrlpts = [
            point / weight for point, weight in zip(newnumctrlpts, dennumctrlpts)
        ]
        number_bound = 1 + 2 * curve.degree
        newknotvector = number_bound * [knotvector[0]] + number_bound * [knotvector[-1]]
        finalcurve = curve.__class__(newknotvector)
        finalcurve.ctrlpoints = newnumctrlpts
        finalcurve.weights = dennumctrlpts
        return finalcurve

    def nonrational_spline(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.weights is None
        assert curve.degree + 1 != curve.npts
        if curve.degree == 0:
            newknotvector = curve.knotvector.limits
            anypoint = curve.ctrlpoints[0]
            newctrlpoints = [0 * anypoint]
            return Curve(newknotvector, newctrlpoints)

        knotvector = tuple(curve.knotvector)
        matrix = heavy.Calculus.derivate_nonrational_spline(knotvector)
        ctrlpoints = np.dot(matrix, curve.ctrlpoints)
        newvector = heavy.KnotVector.derivate(knotvector)
        newcurve = curve.__class__(newvector, ctrlpoints)
        return newcurve

    @staticmethod
    def rational_spline(curve: Curve) -> Curve:
        numer, denom = curve.fraction()
        dnumer = Derivate.nonrational_spline(numer)
        dnumer.degree_increase(1)  # Shouldn't be necessary
        ddenom = Derivate.nonrational_spline(denom)
        ddenom.degree_increase(1)  # Needs further correction
        deriva = dnumer * denom - numer * ddenom
        deriva /= denom * denom
        return deriva


class Integrate:
    pass
