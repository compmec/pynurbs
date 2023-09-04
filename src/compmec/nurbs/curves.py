from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from compmec.nurbs import heavy
from compmec.nurbs.__classes__ import Intface_BaseCurve
from compmec.nurbs.knotspace import KnotVector


def norm(object: Union[float, Tuple[float]], L: int = 0) -> float:
    """
    Computes recursively a norm of an object.
    If L = 0, it means infinity norm
    If L = 1, it means abs norm
    If L = 2, it means euclidean norm
    """
    try:
        soma = 0
        for item in object:
            norma = norm(item, L)
            soma = max(soma, norma) if L == 0 else soma + norma**L
        return soma if L == 0 else soma ** (1 / L)
    except TypeError:
        return abs(object)


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
        if (self.ctrlpoints is None) ^ (other.ctrlpoints is None):
            return False
        newknotvec = self.knotvector | other.knotvector
        selfcopy = self.copy()
        selfcopy.knotvector = newknotvec
        othercopy = other.copy()
        othercopy.knotvector = newknotvec
        for poi, qoi in zip(self.ctrlpoints, othercopy.ctrlpoints):
            if norm(poi - qoi) > 1e-9:
                return False
        return True

    def __ne__(self, obj: object):
        return not self.__eq__(obj)

    def __neg__(self):
        if self.ctrlpoints is None:
            raise ValueError
        newcurve = self.copy()
        newctrlpoints = [-1 * ctrlpt for ctrlpt in newcurve.ctrlpoints]
        newcurve.ctrlpoints = newctrlpoints
        return newcurve

    def __add__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copy = self.copy()
            copy.ctrlpoints = [other + point for point in self.ctrlpoints]
            return copy
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            matra, matrb = heavy.MathOperations.add_spline_curve(vecta, vectb)
            curve = Curve(self.knotvector | other.knotvector)
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
            copy = self.copy()
            copy.ctrlpoints = [point * other for point in copy.ctrlpoints]
            return copy
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            vectmul = heavy.MathOperations.knotvector_mul(vecta, vectb)
            matrix3d = heavy.MathOperations.mul_spline_curve(vecta, vectb)
            ctrlpoints = np.tensordot(
                np.moveaxis(self.ctrlpoints, 0, -1), matrix3d, axes=1
            )
            ctrlpoints = ctrlpoints @ other.ctrlpoints
            curve = Curve(vectmul, ctrlpoints)
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa * numb) / (dena * denb)

    def __rmul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        assert not isinstance(other, self.__class__)
        copy = self.copy()
        copy.ctrlpoints = [other * point for point in copy.ctrlpoints]
        return copy

    def __truediv__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copy = self.copy()
            copy.ctrlpoints = [point / other for point in copy.ctrlpoints]
            return copy
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            copyse = self.copy()
            copyot = other.copy()
            vectora, vectorb = tuple(copyse.knotvector), tuple(copyot.knotvector)
            vectorc = tuple(copyse.knotvector | copyot.knotvector)
            transctrlpts = heavy.Operations.matrix_transformation(vectora, vectorc)
            transweights = heavy.Operations.matrix_transformation(vectorb, vectorc)
            weights = np.dot(transweights, copyot.ctrlpoints)
            ctrlpts = np.dot(transctrlpts, copyse.ctrlpoints)
            ctrlpts = [pti / wi for pti, wi in zip(ctrlpts, weights)]
            return self.__class__(vectorc, ctrlpts, weights)

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
        if self.weights is None:
            newcurve = self.__class__(tuple(self.knotvector))
            newcurve.weights = [deepcopy(point) for point in self.ctrlpoints]
            newcurve.ctrlpoints = [1 / w for w in newcurve.weights]
            return newcurve
        num, den = self.fraction()
        frac = den / num
        return other * frac

    def __or__(self, other: object):
        umaxleft = self.knotvector[-1]
        uminright = other.knotvector[0]
        if umaxleft != uminright:
            error_msg = f"max(Uleft) = {umaxleft} != {uminright} = min(Uright)"
            raise ValueError(error_msg)
        othercopy = other.copy()
        selfcopy = self.copy()
        maxdegree = max(self.degree, other.degree)
        selfcopy.degree_increase(maxdegree - self.degree)
        othercopy.degree_increase(maxdegree - other.degree)
        npts0 = selfcopy.npts
        npts1 = othercopy.npts
        newknotvector = [0] * (maxdegree + npts0 + npts1 + 1)
        newknotvector[:npts0] = selfcopy.knotvector[:npts0]
        newknotvector[npts0:] = othercopy.knotvector[1:]
        newknotvector = KnotVector(newknotvector)
        newctrlpoints = [0] * (npts0 + npts1 - 1)
        newctrlpoints[:npts0] = selfcopy.ctrlpoints[:npts0]
        newctrlpoints[npts0:] = othercopy.ctrlpoints[1:]
        newcurve = self.__class__(newknotvector, newctrlpoints)
        newcurve.knot_clean([umaxleft])
        return newcurve

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
        newknotvector = self.knotvector.copy()
        newknotvector.degree = int(value)
        self.knotvector = newknotvector

    @weights.setter
    def weights(self, value: Tuple[float]):
        if value is None:
            self.__weights = None
            return
        try:
            value = tuple(value)
            for val in value:
                float(val)
        except TypeError:
            msg = f"Weights must be a vector of floats, received {value}"
            raise ValueError(msg)
        # Verify if there's roots
        vector = tuple(self.knotvector)
        roots = heavy.find_roots(vector, value)
        if roots:
            error_msg = f"Zero division at nodes {roots}"
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

    def copy(self) -> Curve:
        """
        Returns a copy with all the internal elements
        """
        knotvector = self.knotvector.copy()
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
            numerator = self.copy()
            return numerator, 1
        ctrlpoints = [deepcopy(point) for point in self.ctrlpoints]
        numerator = self.__class__(self.knotvector.copy())
        denominator = self.__class__(self.knotvector.copy())
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
        mattrans, materror = heavy.LeastSquare.spline2spline(oldvec, newvec)

        materror = np.array(materror)
        error = np.moveaxis(self.ctrlpoints, 0, -1) @ materror @ tuple(self.ctrlpoints)
        error = np.max(np.abs(error))
        if self.weights is not None:
            error += tuple(self.weights) @ materror @ tuple(self.ctrlpoints)
        if tolerance and error > tolerance:
            error_msg = "Cannot update knotvector cause error is "
            error_msg += f" {error} > {tolerance}"
            raise ValueError(error_msg)
        self.set_knotvector(newvec)
        self.apply_lineartrans(mattrans)

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

    def __eval(self, nodes: Tuple[float]) -> Tuple[Any]:
        """
        Private method fto evaluate points in the curve
        """
        vector = tuple(self.knotvector)
        nodes = tuple(nodes)
        degree = int(self.knotvector.degree)
        if self.weights is None:
            eval = heavy.eval_spline_nodes
            matrix = eval(vector, nodes, degree)
        else:
            eval = heavy.eval_rational_nodes
            weights = tuple(self.weights)
            matrix = eval(vector, weights, nodes, degree)
        result = np.moveaxis(matrix, 0, -1) @ self.ctrlpoints
        return tuple(result)

    def eval(self, nodes: Union[float, Tuple[float]]) -> Union[Any, Tuple[Any]]:
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
        newknotvec = self.knotvector.copy() - tuple(nodes)
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

    def degree_decrease(
        self, times: Optional[int] = 1, tolerance: Optional[float] = 1e-9
    ):
        """
        The same as mycurve.degree -= 1
        But this function forces the degree reductions without looking the error
        """
        newknotvec = self.knotvector.copy()
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

    def clean(self, tolerance: float = 1e-9):
        self.degree_clean(tolerance=tolerance)
        self.knot_clean(tolerance=tolerance)
        if self.weights is None:
            return
        # Try to reduce to spline
        knotvector = tuple(self.knotvector)
        weights = tuple(self.weights)
        ctrlpoints = tuple(self.ctrlpoints)
        mattrans, materror = heavy.LeastSquare.func2func(
            knotvector, weights, knotvector, [1] * self.npts
        )
        error = np.moveaxis(ctrlpoints, 0, -1) @ materror @ ctrlpoints
        error = np.max(abs(error))
        error = max(error, weights @ np.array(materror) @ weights)
        if error < tolerance:
            self.ctrlpoints = np.array(mattrans) @ ctrlpoints
            self.weights = None
            assert NotImplementedError  # Needs correction
            self.clean(tolerance)

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

    def fit_curve(self, other: Curve, nodes: Tuple[float] = None) -> None:
        """
        Given a 'other' curve, this function finds the best control points
        such keeps as near as possible to 'other'

        If weights is None -> spline fit
        Else: -> rationa spline fit

        If nodes are given
            if len(nodes) < npts
                interpolates all nodes, uses least square in others
            if len(nodes) == npts
                interpolate at all points
            if len(nodes) > npts:
                same as fit_points(other(nodes), nodes)
        """
        assert nodes is None  # Needs implementation
        assert isinstance(other, self.__class__)
        vectora, vectorb = tuple(self.knotvector), tuple(other.knotvector)
        if self.weights is None and other.weights is None:
            lstsq = heavy.LeastSquare.spline2spline
            transmat, _ = lstsq(vectorb, vectora)
        else:
            weightsa = self.weights if self.weights else [1] * self.npts
            weightsb = other.weights if other.weights else [1] * other.npts
            lstsq = heavy.LeastSquare.func2func
            transmat, _ = lstsq(vectorb, weightsb, vectora, weightsa)
        transmat = np.array(transmat)
        ctrlpoints = transmat @ other.ctrlpoints
        if self.weights is not None:
            weights = transmat @ weightsb
            ctrlpoints = [point / weig for point, weig in zip(ctrlpoints, weights)]
            self.weights = weights
        self.ctrlpoints = ctrlpoints

    def fit_function(self, function: Callable, nodes: Tuple[float] = None) -> None:
        """
        Find the control points such fits the given function
        If nodes are not given, it uses least square in many intervals
            for subinterval [uk, u_{k+1}] evaluates on
            max(degree+1, 5*npts/len(subintervals)) using chebyshev nodes
        If nodes are given
            if len(nodes) < npts
                interpolates all nodes and
            if len(nodes) == npts
                interpolate at all points
            if len(nodes) > npts:
                same as fit_points(function(nodes), nodes)
        """
        if nodes is not None:
            raise NotImplementedError
        assert not isinstance(function, self.__class__)
        knots = self.knotvector.knots
        npts_each = 1 + int(np.ceil(self.degree * self.npts / (len(knots) - 1)))
        nodes = []
        numbtype = heavy.number_type(knots)
        if numbtype in (float, np.floating):
            funcnodes = heavy.LeastSquare.chebyshev_nodes
        else:
            funcnodes = heavy.LeastSquare.uniform_nodes
        for start, end in zip(knots[:-1], knots[1:]):
            nodes += list(funcnodes(npts_each, start, end))
        nodes = tuple(nodes)
        funcvals = [function(node) for node in nodes]
        return self.fit_points(funcvals, nodes)

    def fit_points(self, points: Tuple[Any], nodes: Tuple[float] = None) -> None:
        """
        Fit the curve into given points
        If the quantity of points are not enough to
        """
        assert len(points) >= self.npts
        fitfunc = heavy.LeastSquare.fit_function
        if nodes is None:
            umin, umax = self.knotvector.limits
            nodes = heavy.LeastSquare.linspace(umin, umax, len(points))
        knotvector = tuple(self.knotvector)
        nodes = tuple(nodes)
        weights = None if self.weights is None else tuple(self.weights)
        matrix = fitfunc(knotvector, nodes, weights)
        ctrlpoints = np.dot(matrix, points)
        self.ctrlpoints = tuple(ctrlpoints)

    def fit(
        self,
        param: Union[Curve, Callable[[float], float], Tuple[Any]],
        nodes: Tuple[float] = None,
    ) -> None:
        if nodes is not None:
            iter(nodes)
            for node in nodes:
                assert self.knotvector[0] <= node
                assert node <= self.knotvector[-1]
        if isinstance(param, self.__class__):
            return self.fit_curve(param, nodes)
        if callable(param):
            return self.fit_function(param, nodes)
        return self.fit_points(param, nodes)
