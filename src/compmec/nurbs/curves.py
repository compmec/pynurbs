from typing import Optional, Tuple, Union

import numpy as np

from compmec.nurbs import algorithms as algo
from compmec.nurbs.__classes__ import Intface_BaseCurve
from compmec.nurbs.functions import RationalFunction, SplineFunction
from compmec.nurbs.knotspace import KnotVector


class BaseCurve(Intface_BaseCurve):
    def __init__(self, knotvector: KnotVector):
        self.knotvector = KnotVector(knotvector)

    def deepcopy(self) -> Intface_BaseCurve:
        curve = self.__class__(self.knotvector)
        curve.ctrlpoints = self.ctrlpoints
        return curve

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.evaluate(u)

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        if self.ctrlpoints is None:
            error_msg = "Cannot evaluate: There are no control points"
            raise ValueError(error_msg)
        L = self.F(u)
        return np.moveaxis(L, 0, -1) @ self.ctrlpoints

    @property
    def knotvector(self):
        return KnotVector(self.__knotvector)

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
    def ctrlpoints(self):
        return self.__ctrlpoints

    @knotvector.setter
    def knotvector(self, value: KnotVector):
        self.__knotvector = KnotVector(value)

    @degree.setter
    def degree(self, value: int):
        value = int(value)
        if value < 1:
            raise ValueError("The degree must be 1 or higher!")
        if value == self.degree:
            return
        if value > self.degree:
            self.__degree_increase(value - self.degree)
        else:
            self.__degree_decrease(self.degree - value)

    @ctrlpoints.setter
    def ctrlpoints(self, value: np.ndarray):
        if value is None:
            self.__ctrlpoints = None
            return
        try:
            iter(value)
        except Exception:
            raise ValueError
        if len(value) != self.npts:
            error_msg = "The number of control points must be the same as"
            error_msg += " degrees of freedom of KnotVector.\n"
            error_msg += f"  knotvector.npts = {self.npts}"
            error_msg += f"  len(ctrlpoints) = {len(value)}"
            raise ValueError(error_msg)
        try:
            1.0 * value[0] - 1.5 * value[0] + 3.1 * value[0]
        except Exception:
            error_msg = "For each control point P, it's needed operations"
            error_msg = "with floats like 1.0*P - 1.5*P + 3.1*P"
            raise ValueError(error_msg)
        self.__ctrlpoints = value

    def __eq__(self, other: object) -> bool:
        if type(self) != type(other):
            return False
        if self.knotvector[0] != other.knotvector[0]:
            return False
        if self.knotvector[-1] != other.knotvector[-1]:
            return False
        selfcopy = self.deepcopy()
        othercopy = other.deepcopy()
        maxdegree = max(self.degree, other.degree)
        selfcopy.degree = maxdegree
        othercopy.degree = maxdegree
        allknots = list(selfcopy.knots) + list(othercopy.knots)
        allknots = list(set(allknots))
        allknots.sort()
        for knot in allknots[1:-1]:
            multself = selfcopy.knotvector.mult(knot)
            multother = othercopy.knotvector.mult(knot)
            diff = multself - multother
            if diff > 0:
                othercopy.knot_insert(diff * [knot])
            elif diff < 0:
                selfcopy.knot_insert((-diff) * [knot])
        # Now the knotvectors are equal
        diffctrlpoints = selfcopy.ctrlpoints - othercopy.ctrlpoints
        if np.all(np.abs(diffctrlpoints) > 1e-9):
            return False
        return True

    def __ne__(self, obj: object):
        return not self.__eq__(obj)

    def __neg__(self):
        return self.__class__(self.knotvector, np.copy(-self.ctrlpoints))

    def __add__(self, obj: object):
        if type(self) != type(obj):
            error_msg = f"Cannot sum a {type(obj)} object with"
            error_msg += f" a {self.__class__} object"
            raise TypeError(error_msg)
        if self.knotvector != obj.knotvector:
            raise ValueError("The knotvectors of curves are not the same!")
        if self.ctrlpoints.shape != obj.ctrlpoints.shape:
            raise ValueError("The shape of control points are not the same!")
        newP = np.copy(self.ctrlpoints) + obj.ctrlpoints
        return self.__class__(self.knotvector, newP)

    def __sub__(self, obj: object):
        return self + (-obj)

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
        newknotvector = np.empty(maxdegree + npts0 + npts1, dtype="float64")
        newknotvector.fill(np.nan)
        newknotvector[:npts0] = selfcopy.knotvector[:npts0]
        newknotvector[npts0:] = othercopy.knotvector[1:]
        newknotvector = KnotVector(newknotvector)
        newshape = [npts0 + npts1 - 1] + list(self.ctrlpoints.shape[1:])
        newctrlpoints = np.zeros(newshape, dtype="float64")
        newctrlpoints[:npts0] = selfcopy.ctrlpoints[:npts0]
        newctrlpoints[npts0:] = othercopy.ctrlpoints[1:]
        return self.__class__(newknotvector, newctrlpoints)

    def knot_insert(self, nodes: Union[float, Tuple[float]]) -> None:
        nodes = np.array(nodes, dtype="float64").flatten()
        newknotvector = np.array(self.knotvector).tolist()

        newknotvector.extend(nodes)
        newknotvector.sort()
        newknotvector = KnotVector(newknotvector)
        T, _ = algo.LeastSquare.spline(self.knotvector, newknotvector)
        newcontrolpoints = T @ self.ctrlpoints
        self.__knotvector = KnotVector(newknotvector)
        self.ctrlpoints = newcontrolpoints

    def knot_remove(
        self, nodes: Union[float, Tuple[float]], tolerance: float = 1e-9
    ) -> None:
        nodes = np.array(nodes, dtype="float64").flatten()
        nodes_requested = list(set(list(nodes)))
        newknotvector = list(self.knotvector)
        for node in nodes_requested:
            if node not in self.knots:
                error_msg = f"Requested remove ({node:.3f}),"
                error_msg += " which it's not in knotvector"
                raise ValueError(error_msg)
        times_requested = [sum(abs(nodes - node) < 1e-9) for node in nodes]
        for node, times in zip(nodes_requested, times_requested):
            mult = self.knotvector.mult(node)
            if times > mult:
                error_msg = f"Requested remove {times} times the node "
                error_msg += f"{node:.3f}, but only {mult} available."
                raise ValueError(error_msg)
            for i in range(times):
                newknotvector.remove(node)
        newknotvector = KnotVector(newknotvector)
        T, E = algo.LeastSquare.spline(self.knotvector, newknotvector)
        error = np.moveaxis(self.ctrlpoints, 0, -1) @ E @ self.ctrlpoints
        error = np.max(np.abs(error))
        if error > tolerance:
            error_msg = f"Cannot remove the nodes {nodes} cause the "
            error_msg += f" error ({error:.1e}) is > {tolerance:.1e} "
            raise ValueError(error_msg)
        newctrlpoints = T @ self.ctrlpoints
        self.__knotvector = KnotVector(newknotvector)
        self.ctrlpoints = newctrlpoints

    def knot_clean(self, tolerance: float = 1e-9) -> None:
        """
        Remove all unecessary knots.
        If removing the knot the error is bigger than the tolerance,
        nothing happens
        """
        internal_knots = self.knotvector.knots[1:-1]
        for knot in internal_knots:
            try:
                while True:
                    self.knot_remove(knot, tolerance)
            except ValueError:
                pass

    def __degree_increase(self, times: int):
        newknotvector = list(self.knotvector) + times * list(self.knots)
        newknotvector.sort()
        T, _ = algo.LeastSquare.spline(self.knotvector, newknotvector)
        newctrlpoints = T @ self.ctrlpoints
        self.__knotvector = KnotVector(newknotvector)
        self.ctrlpoints = newctrlpoints

    def __degree_decrease(self, times: int = 1, tolerance: float = 1e-9):
        newknotvector = list(self.knotvector)
        for knot in self.knots:
            mult = min(times, self.knotvector.mult(knot))
            for i in range(mult):
                newknotvector.remove(knot)
        T, E = algo.LeastSquare.spline(self.knotvector, newknotvector)
        error = np.moveaxis(self.ctrlpoints, 0, -1) @ E @ self.ctrlpoints
        error = np.max(np.abs(error))
        if error > tolerance:
            error_msg = f"Cannot reduce degree {times} times "
            error_msg += f"cause the error ({error:.1e}) is > {tolerance:.1e} "
            raise ValueError(error_msg)
        newctrlpoints = T @ self.ctrlpoints
        self.__knotvector = newknotvector
        self.ctrlpoints = newctrlpoints

    def degree_decrease(self, times: Optional[int] = 1):
        """
        The same as mycurve.degree -= 1
        But this function forces the degree reductions without looking the error
        """
        self.degree -= times

    def degree_increase(self, times: Optional[int] = 1):
        """
        The same as mycurve.degree += times
        But this function forces the degree reductions without looking the error
        """
        self.degree += times

    def degree_clean(self, tolerance: float = 1e-9):
        """
        Reduces au maximum the degree of the curve received the tolerance.
        If the reduced degree error is bigger than the tolerance, nothing happen
        """
        try:
            while True:
                self.__degree_decrease(1, tolerance)
        except ValueError:
            pass

    def split(
        self, nodes: Optional[Union[float, np.ndarray]] = None
    ) -> Tuple[Intface_BaseCurve]:
        """
        Separate the current spline at specified knots
        If no arguments are given, it splits at every knot and
            returns a list of bezier curves
        """
        if nodes is None:  # split into beziers
            if self.degree + 1 == self.npts:  # already bezier
                return (self.deepcopy(),)
            return self.split(self.knots)
        nodes = np.array(nodes, dtype="float64").flatten()
        nodes = list(nodes) + [self.knotvector[0], self.knotvector[-1]]
        nodes = list(set(nodes))
        nodes.sort()
        copycurve = self.deepcopy()
        for node in nodes[1:-1]:
            mult = self.knotvector.mult(node)
            copycurve.knot_insert((self.degree - mult) * [node])

        allknotvec = algo.KnotVector.split(self.knotvector, nodes)
        listcurves = [0] * len(allknotvec)
        for i, newknotvec in enumerate(allknotvec):
            nodeinf, nodesup = nodes[i], nodes[i + 1]
            lowerind = copycurve.knotvector.span(nodeinf) - self.degree
            if nodesup < self.knotvector[-1]:
                upperind = copycurve.knotvector.span(nodesup) - self.degree + 1
            else:
                upperind = None
            newctrlpts = copycurve.ctrlpoints[lowerind:upperind]
            newcurve = self.__class__(newknotvec, newctrlpts)
            listcurves[i] = newcurve
        del copycurve
        return tuple(listcurves)


class SplineCurve(BaseCurve):
    def __init__(self, knotvector: KnotVector, ctrlpoints: Optional[np.ndarray] = None):
        super().__init__(knotvector)
        if ctrlpoints is not None:
            self.ctrlpoints = ctrlpoints

    @property
    def F(self):
        return SplineFunction(self.knotvector)


class RationalCurve(BaseCurve):
    def __init__(
        self,
        knotvector: KnotVector,
        ctrlpoints: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(knotvector)
        self.ctrlpoints = ctrlpoints
        self.weights = weights

    @property
    def F(self):
        return RationalFunction(self.knotvector)

    def deepcopy(self):
        curve = super().deepcopy()
        curve.weights = self.weights
        return curve

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value: Tuple[float]):
        self.__weights = value
