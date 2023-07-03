from typing import Dict, Optional, Tuple, Union

import numpy as np

from compmec.nurbs import algorithms as algo
from compmec.nurbs.__classes__ import Intface_BaseCurve
from compmec.nurbs.functions import RationalFunction, SplineFunction
from compmec.nurbs.knotspace import GeneratorKnotVector, KnotVector


class BaseCurve(Intface_BaseCurve):
    def __init__(self, knotvector: KnotVector, ctrlpoints: np.ndarray):
        self.__set_UFP(knotvector, ctrlpoints)

    def deepcopy(self) -> Intface_BaseCurve:
        return self.__class__(self.knotvector, self.ctrlpoints)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.evaluate(u)

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        L = self.F(u)
        return L.T @ self.ctrlpoints

    @property
    def degree(self):
        return self.knotvector.degree

    @property
    def npts(self):
        return self.knotvector.npts

    @property
    def knotvector(self):
        return self.F.knotvector

    @property
    def knots(self):
        return self.F.knots

    @property
    def F(self):
        return self.__F

    @property
    def ctrlpoints(self):
        return self.__ctrlpoints

    @degree.setter
    def degree(self, value: int):
        if not isinstance(value, int):
            raise TypeError("To set new degree, it must be an integer")
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
        if not isinstance(value, (list, tuple, np.ndarray)):
            error_msg = f"Control points are invalid! type = {type(value)}."
            raise TypeError(error_msg)
        value = np.array(value, dtype="object")
        if np.any(value == None):
            raise TypeError("None is inside the array of control points")
        try:
            value = np.array(value, dtype="float64")
        except Exception:
            error_msg = "Could not convert control points to array of floats"
            raise TypeError(error_msg)
        if value.shape[0] != self.npts:
            error_msg = "The number of control points must be the same as"
            error_msg += " degrees of freedom of KnotVector.\n"
            error_msg += f"  knotvector.npts = {self.npts}"
            error_msg += f"  len(ctrlpoints) = {len(value)}"
            raise ValueError(error_msg)
        self.__ctrlpoints = value

    def __set_UFP(self, knotvector: KnotVector, ctrlpoints: np.ndarray):
        knotvector = KnotVector(knotvector)
        self.__F = self._create_base_function_instance(knotvector)
        self.ctrlpoints = ctrlpoints

    def __eq__(self, obj: object) -> bool:
        if type(self) != type(obj):
            return False
        knots = self.knotvector.knots
        pairs = list(zip(knots[:-1], knots[1:]))
        utest = [np.linspace(ua, ub, 2 + self.degree) for ua, ub in pairs]
        utest = list(set(np.array(utest).reshape(-1)))
        for i, ui in enumerate(utest):
            Cusel = self.evaluate(ui)
            Cuobj = obj.evaluate(ui)
            if np.any(np.abs(Cusel - Cuobj) > 1e-9):
                return False
        return True

    def __ne__(self, __obj: object):
        return not self.__eq__(__obj)

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

    def __sub__(self, __obj: object):
        return self + (-__obj)

    def knot_insert(self, knots: Union[float, Tuple[float]]) -> None:
        knots = np.array(knots, dtype="float64")
        if knots.ndim == 0:
            knots = [knots]
        elif knots.ndim == 1:
            pass
        else:
            raise ValueError("Invalid input of knots")
        newknotvector = np.array(self.knotvector).tolist()

        newknotvector.extend(knots)
        newknotvector.sort()
        newknotvector = KnotVector(newknotvector)
        T, _ = algo.LeastSquare.spline(self.knotvector, newknotvector)
        newcontrolpoints = T @ self.ctrlpoints
        self.__set_UFP(newknotvector, newcontrolpoints)

    def knot_remove(
        self, nodes: Union[float, Tuple[float]], tolerance: float = 1e-9
    ) -> None:
        nodes = np.array(nodes, dtype="float64")
        if nodes.ndim == 0:
            nodes = np.array([nodes.tolist()])
        elif nodes.ndim != 1:
            raise ValueError("Nodes invalid")
        nodes_requested = list(set(list(nodes)))
        newknotvector = list(self.knotvector)
        for node in nodes_requested:
            if not node in self.knots:
                error_msg = f"Requested remove ({node:.3f}),"
                error_msg += f" which it's not in knotvector"
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
        self.__set_UFP(newknotvector, newctrlpoints)

    def knot_clean(self, tolerance: float = 1e-9) -> None:
        """
        Remove all unecessary knots.
        If removing the knot the error is bigger than the tolerance,
        nothing happens
        """
        intknots = self.knotvector.knots
        while True:
            removed = False
            for knot in intknots:
                try:
                    self.knot_remove(knot)
                    removed = True
                except ValueError:
                    pass
            if not removed:
                break

    def __degree_increase(self, times: int):
        newknotvector = list(self.knotvector) + times * list(self.knots)
        newknotvector.sort()
        T, _ = algo.LeastSquare.spline(self.knotvector, newknotvector)
        newctrlpoints = T @ self.ctrlpoints
        self.__set_UFP(newknotvector, newctrlpoints)

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
        self.__set_UFP(newknotvector, newctrlpoints)

    def degree_decrease(self, times: Optional[int] = 1):
        """
        The same as mycurve.degree -= 1
        But this function forces the degree reductions without looking the error
        """
        self.degree = self.degree - times

    def degree_increase(self, times: Optional[int] = 1):
        """
        The same as mycurve.degree += times
        But this function forces the degree reductions without looking the error
        """
        self.degree = self.degree + times

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
        nodes = np.array(nodes, dtype="float64")
        if nodes.ndim == 0:
            nodes = np.array([nodes])
        elif nodes.ndim != 1:
            raise ValueError("Nodes invalid")
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
    def __init__(self, knotvector: KnotVector, ctrlpoints: np.ndarray):
        super().__init__(knotvector, ctrlpoints)

    def __repr__(self):
        return f"SplineCurve of degree {self.degree} and {self.npts} control points"

    def _create_base_function_instance(self, knotvector: KnotVector):
        return SplineFunction(knotvector)


class RationalCurve(BaseCurve):
    def __init__(self, knotvector: KnotVector, ctrlpoints: np.ndarray):
        super().__init__(knotvector, ctrlpoints)

    def _create_base_function_instance(self, knotvector: KnotVector):
        return RationalFunction(knotvector)

    @property
    def weights(self):
        return self.F.weights

    @weights.setter
    def weights(self, value: Tuple[float]):
        self.F.weights = value
