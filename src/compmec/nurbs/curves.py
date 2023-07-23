from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np

from compmec.nurbs import heavy
from compmec.nurbs.__classes__ import Intface_BaseCurve
from compmec.nurbs.knotspace import KnotVector


class BaseCurve(Intface_BaseCurve):
    def __init__(self, knotvector: KnotVector):
        self.__ctrlpoints = None
        self.__weights = None
        self.__knotvector = KnotVector(knotvector)

    def __call__(self, nodes: np.ndarray) -> np.ndarray:
        return self.eval(nodes)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        if self.knotvector[0] != other.knotvector[0]:
            return False
        if self.knotvector[-1] != other.knotvector[-1]:
            return False
        newknotvec = self.knotvector | other.knotvector
        selfcopy = self.deepcopy()
        selfcopy.knotvector = newknotvec
        othercopy = other.deepcopy()
        othercopy.knotvector = newknotvec
        for poi, qoi in zip(self.ctrlpoints, othercopy.ctrlpoints):
            if np.linalg.norm(poi - qoi) > 1e-9:
                return False
        return True

    def __ne__(self, obj: object):
        return not self.__eq__(obj)

    def __neg__(self):
        newcurve = self.deepcopy()
        newctrlpoints = [-1 * ctrlpt for ctrlpt in newcurve.ctrlpoints]
        newcurve.ctrlpoints = newctrlpoints
        return newcurve

    def __add__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copy = self.deepcopy()
            copy.ctrlpoints = [other + point for point in self.ctrlpoints]
            return copy
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            assert self.knotvector == other.knotvector
            vectadd = self.knotvector | other.knotvector
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            matra, matrb = heavy.MathOperations.add_spline_curve(vecta, vectb)
            curve = Curve(vectadd)
            ctrlpoints = np.array(matra) @ self.ctrlpoints
            ctrlpoints += np.array(matrb) @ other.ctrlpoints
            curve.ctrlpoints = ctrlpoints
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa * denb + numb * dena) / (dena * denb)

    def __radd__(self, other: object):
        return self.__add__(other)

    def __sub__(self, other: object):
        return self + (-other)

    def __rsub__(self, other: object):
        return other + (-self)

    def __mul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copy = self.deepcopy()
            copy.ctrlpoints = [point * other for point in copy.ctrlpoints]
            return copy
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            assert self.knotvector == other.knotvector
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            vectmul = heavy.MathOperations.knotvector_mul(vecta, vectb)
            matrix3d = heavy.MathOperations.mul_spline_curve(vecta, vectb)
            ctrlpoints = []
            for i, matrix in enumerate(matrix3d):
                matrix = np.array(matrix)
                newpoint = self.ctrlpoints @ matrix @ other.ctrlpoints
                ctrlpoints.append(newpoint)
            curve = Curve(vectmul, ctrlpoints)
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa * numb) / (dena * denb)

    def __rmul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        assert not isinstance(other, self.__class__)
        copy = self.deepcopy()
        copy.ctrlpoints = [other * point for point in copy.ctrlpoints]
        return copy

    def __truediv__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copy = self.deepcopy()
            copy.ctrlpoints = [point / other for point in copy.ctrlpoints]
            return copy
        if self.weights is None and other.weights is None:
            assert self.knotvector == other.knotvector
            vectorc = self.knotvector | other.knotvector
            weights = [deepcopy(point) for point in other.ctrlpoints]
            ctrlpts = [deepcopy(pti) for pti in self.ctrlpoints]
            ctrlpts = [pti / wi for pti, wi in zip(ctrlpts, weights)]
            curve = self.__class__(vectorc, ctrlpts, weights)
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa * denb) / (dena * numb)

    def __rtruediv__(self, other: object):
        """
        Example: 1/curve
        """
        if self.ctrlpoints is None:
            raise ValueError
        assert not isinstance(other, self.__class__)
        for point in self.ctrlpoints:
            float(point)
        num, den = self.fraction()
        return other * (den / num)

    def __or__(self, other: object):
        umaxleft = self.knotvector[-1]
        uminright = other.knotvector[0]
        if umaxleft != uminright:
            error_msg = f"max(Uleft) = {umaxleft} != {uminright} = min(Uright)"
            raise ValueError(error_msg)
        pointleft = self.ctrlpoints[-1]
        pointright = other.ctrlpoints[0]
        if np.sum((pointleft - pointright) ** 2) > 1e-9:
            error_msg = "Points are not coincident.\n"
            error_msg += f"  Point left = {pointleft}\n"
            error_msg += f" Point right = {pointright}"
            raise ValueError(error_msg)
        othercopy = other.deepcopy()
        selfcopy = self.deepcopy()
        maxdegree = max(self.degree, other.degree)
        selfcopy.degree = maxdegree
        othercopy.degree = maxdegree
        npts0 = selfcopy.npts
        npts1 = othercopy.npts
        newknotvector = [0] * (maxdegree + npts0 + npts1)
        newknotvector[:npts0] = selfcopy.knotvector[:npts0]
        newknotvector[npts0:] = othercopy.knotvector[1:]
        newknotvector = KnotVector(newknotvector)
        newctrlpoints = [0] * (npts0 + npts1 - 1)
        newctrlpoints[:npts0] = selfcopy.ctrlpoints[:npts0]
        newctrlpoints[npts0:] = othercopy.ctrlpoints[1:]
        return self.__class__(newknotvector, newctrlpoints)

    @property
    def knotvector(self):
        return self.__knotvector

    @property
    def degree(self):
        return self.knotvector.degree

    @property
    def npts(self):
        return self.knotvector.npts

    @property
    def knots(self):
        return self.knotvector.knots

    @property
    def weights(self):
        if self.__weights is None:
            return None
        return tuple(self.__weights)

    @property
    def ctrlpoints(self):
        if self.__ctrlpoints is None:
            return None
        return tuple(self.__ctrlpoints)

    @knotvector.setter
    def knotvector(self, value: KnotVector):
        value = KnotVector(value)
        if self.knotvector == value:
            return
        self.update_knotvector(value)

    @degree.setter
    def degree(self, value: int):
        newknotvector = self.knotvector.deepcopy()
        newknotvector.degree = int(value)
        self.knotvector = newknotvector

    @weights.setter
    def weights(self, value: Tuple[float]):
        if value is None:
            self.__weights = None
            return
        array = np.array(value, dtype="float64")
        if array.shape != (self.npts,):
            error_msg = "Weights must be a 1D array with "
            error_msg += f"{self.npts} points"
            raise ValueError(error_msg)
        self.__weights = tuple(value)

    @ctrlpoints.setter
    def ctrlpoints(self, newpoints: np.ndarray):
        if newpoints is None:
            self.__ctrlpoints = None
            return
        if isinstance(newpoints, str):
            raise TypeError
        try:
            iter(newpoints)
        except Exception:
            raise TypeError
        for point in newpoints:  # Verify if operations are valid for each node
            for knot in self.knotvector.knots:
                knot * point
            for otherpoint in newpoints:
                point + otherpoint  # Verify if we can sum every point, same type

        if len(newpoints) != self.npts:
            error_msg = f"The number of control points ({len(newpoints)}) must be "
            error_msg += f"the same as npts of KnotVector ({self.knotvector.npts})\n"
            error_msg += f"  knotvector.npts = {self.npts}"
            error_msg += f"  len(ctrlpoints) = {len(newpoints)}"
            raise ValueError(error_msg)

        self.__ctrlpoints = tuple(newpoints)

    def deepcopy(self) -> Curve:
        """
        Returns a copy with all the internal elements
        """
        knotvector = [deepcopy(knot) for knot in self.knotvector]
        curve = self.__class__(knotvector)
        if self.ctrlpoints is not None:
            curve.ctrlpoints = [deepcopy(point) for point in self.ctrlpoints]
        if self.weights is not None:
            curve.weights = [deepcopy(weight) for weight in self.weights]
        return curve

    def fraction(self) -> Tuple[BaseCurve]:
        """
        Returns the current curve into the form Spline/Spline
        If it's rational, then returns [Spline, Spline]
        If it's spline, then returns [Spline, 1]
        """
        if self.weights is None:
            numerator = self.deepcopy()
            return numerator, 1
        ctrlpoints = [deepcopy(point) for point in self.ctrlpoints]
        numerator = self.__class__(self.knotvector.deepcopy)
        denominator = self.__class__(self.knotvector.deepcopy)
        numerator.ctrlpoints = [wi * pt for wi, pt in zip(self.weights, ctrlpoints)]
        denominator.ctrlpoints = self.weights
        return numerator, denominator

    def update_knotvector(self, newvector: KnotVector, tolerance: float = 1e-9):
        newvector = KnotVector(newvector)
        if newvector == self.knotvector:
            return
        if self.ctrlpoints is None:
            return self.set_knotvector(newvector)
        oldvec, newvec = tuple(self.knotvector), tuple(newvector)
        T, E = heavy.LeastSquare.spline2spline(oldvec, newvec)
        error = np.moveaxis(self.ctrlpoints, 0, -1) @ E @ self.ctrlpoints
        error = np.max(np.abs(error))
        if tolerance and error > tolerance:
            error_msg = "Cannot update knotvector cause error is "
            error_msg += f" {error} > {tolerance}"
            raise ValueError(error_msg)
        self.set_knotvector(newvec)
        self.apply_lineartrans(T)

    def apply_lineartrans(self, matrix: np.ndarray):
        """ """
        matrix = np.array(matrix)
        if self.weights is None:
            self.ctrlpoints = matrix @ self.ctrlpoints
            return
        oldweights = self.weights
        self.weights = matrix @ oldweights

        if self.ctrlpoints is not None:
            oldctrlpoints = [point for point in self.ctrlpoints]
            for i, weight in enumerate(oldweights):
                oldctrlpoints[i] *= weight
            newctrlpoints = []
            for i, line in enumerate(matrix):
                newctrlpoints.append(0 * self.ctrlpoints[0])
                for j, point in enumerate(oldctrlpoints):
                    newpoint = line[j] * point
                    newpoint /= self.weights[i]
                    newctrlpoints[i] += newpoint
            self.ctrlpoints = newctrlpoints

    def set_knotvector(self, newknotvector: KnotVector):
        self.__knotvector = KnotVector(newknotvector)


class Curve(BaseCurve):
    def __init__(
        self,
        knotvector: KnotVector,
        ctrlpoints: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(knotvector)
        self.ctrlpoints = ctrlpoints
        self.weights = weights

    def __str__(self) -> str:
        if self.npts == self.degree + 1:
            msg = "Bezier"
        elif self.weights is None:
            msg = "Spline"
        else:
            msg = "Rational Spline"
        msg += f" curve of degree {self.degree}"
        msg += f" and {self.npts} control points\n"
        msg += f"KnotVector = {self.knotvector}\n"
        if self.ctrlpoints is None:
            return msg
        msg += "ControlPoints = [\n"
        for point in self.ctrlpoints:
            msg += str(point) + "\n"
        msg += "]\n"
        return msg

    def __eval(self, nodes: Tuple[float]) -> Tuple["Point"]:
        """
        Private method fto evaluate points in the curve
        """
        vector = tuple(self.knotvector)
        nodes = tuple(nodes)
        if self.weights is None:
            matrix = heavy.eval_spline_nodes(vector, nodes)
        else:
            weights = tuple(self.weights)
            matrix = heavy.eval_rational_nodes(vector, weights, nodes)
        result = np.moveaxis(matrix, 0, -1) @ self.ctrlpoints
        return tuple(result)

    def eval(self, nodes: Union[float, Tuple[float]]) -> Union["Point", Tuple["Point"]]:
        if self.ctrlpoints is None:
            error_msg = "Cannot evaluate: There are no control points"
            raise ValueError(error_msg)
        try:
            nodes = tuple(nodes)
            onevalue = False
        except TypeError:
            nodes = (nodes,)
            onevalue = True
        self.knotvector.valid_nodes(nodes)
        result = self.__eval(nodes)
        return result[0] if onevalue else result

    def knot_insert(self, nodes: Tuple[float]) -> None:
        nodes = tuple(nodes)
        oldvector = tuple(self.knotvector)
        newvector = tuple(self.knotvector + tuple(nodes))
        matrix = heavy.Operations.knot_insert(oldvector, nodes)
        self.set_knotvector(newvector)
        self.apply_lineartrans(matrix)

    def knot_remove(self, nodes: Tuple[float], tolerance: float = 1e-9) -> None:
        nodes = tuple(nodes)
        newknotvec = self.knotvector.deepcopy() - tuple(nodes)
        self.update_knotvector(newknotvec, tolerance)

    def knot_clean(
        self, nodes: Optional[Tuple[float]] = None, tolerance: float = 1e-9
    ) -> None:
        """
        Remove all unnecessary knots.
        If no nodes are given, it tries to remove all internal knots
        Nothing happens if removing error by removing certain knot is
        bigger than the tolerance.
        """
        if nodes is None:
            nodes = self.knotvector.knots
        nodes = tuple(set(nodes) - set(self.knotvector.limits))
        for knot in nodes:
            try:
                while True:
                    self.knot_remove((knot,), tolerance)
            except ValueError:
                pass

    def degree_increase(self, times: Optional[int] = 1):
        """
        The same as mycurve.degree += times
        """
        oldvector = tuple(self.knotvector)
        matrix = heavy.Operations.degree_increase(oldvector, times)
        nodes = self.knotvector.knots
        newnodes = times * nodes
        newvector = heavy.KnotVector.insert_knots(oldvector, newnodes)
        self.set_knotvector(newvector)
        self.apply_lineartrans(matrix)

    def clean(self, tolerance: float = 1e-9):
        self.degree_clean(tolerance=tolerance)
        self.knot_clean(tolerance=tolerance)
        if self.weights is None:
            return
        # Try to reduce to spline
        knotvector = tuple(self.knotvector)
        weights = tuple(self.weights)
        ctrlpoints = tuple(self.ctrlpoints)
        T, E = heavy.LeastSquare.func2func(
            knotvector, weights, knotvector, [1] * self.npts
        )
        error = ctrlpoints @ E @ ctrlpoints
        if error < tolerance:
            self.ctrlpoints = T @ ctrlpoints
            self.weights = None
            self.clean(tolerance)

    def degree_decrease(
        self, times: Optional[int] = 1, tolerance: Optional[float] = 1e-9
    ):
        """
        The same as mycurve.degree -= 1
        But this function forces the degree reductions without looking the error
        """
        newknotvec = self.knotvector.deepcopy()
        newknotvec.degree -= times
        self.update_knotvector(newknotvec, tolerance)

    def degree_clean(self, tolerance: float = 1e-9):
        """
        Reduces au maximum the degree of the curve received the tolerance.
        If the reduced degree error is bigger than the tolerance, nothing happens
        """
        try:
            while True:
                self.degree_decrease(1, tolerance)
        except ValueError:
            pass

    def split(self, nodes: Optional[Tuple[float]] = None) -> Tuple[Curve]:
        """
        Separate the current spline at specified knots
        If no arguments are given, it splits at every knot and
            returns a list of bezier curves
        """
        vector = tuple(self.knotvector)
        if nodes is None:
            nodes = self.knotvector.knots
        nodes = tuple(nodes)
        matrices = heavy.Operations.split_curve(vector, nodes)
        newvectors = heavy.KnotVector.split(vector, nodes)
        newcurves = []
        for newvector, matrix in zip(newvectors, matrices):
            matrix = np.array(matrix)
            newcurve = Curve(newvector)
            newcurve.ctrlpoints = matrix @ self.ctrlpoints
            if self.weights is not None:
                newcurve.weights = matrix @ self.weights
            newcurves.append(newcurve)
        return tuple(newcurves)
