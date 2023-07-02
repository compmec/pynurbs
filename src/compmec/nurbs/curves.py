from typing import Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Intface_BaseCurve
from compmec.nurbs.algorithms import Chapter5, Custom
from compmec.nurbs.functions import RationalFunction, SplineFunction
from compmec.nurbs.knotspace import KnotVector


class BaseCurve(Intface_BaseCurve):
    def __init__(self, knotvector: KnotVector, ctrlpoints: np.ndarray):
        self.insert_knot_at_call = True
        self.__set_UFP(knotvector, ctrlpoints)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        if self.insert_knot_at_call:
            for i in range(self.degree):
                self.knot_insert(u)
        return self.evaluate(u)

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        L = self.F(u)
        return L.T @ self.ctrlpoints

    def derivate(self):
        newU = list(self.knotvector)
        for knot in list(set(self.knotvector)):
            newU.remove(knot)
        newU = self.knotvector.__class__(newU)
        shape = [newU.npts] + list(self.ctrlpoints.shape[1:])
        newP = np.random.uniform(0, 1, shape)
        newC = self.__class__(newU, newP)
        return newC

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

    def __transform_knots_to_table(self, knots: Union[float, Tuple[float]]):
        try:
            knots = float(knots)
            return {knots: 1}
        except Exception:
            pass
        try:
            iter(knots)
            knots = np.array(knots, dtype="float64")
        except Exception:
            raise TypeError
        if knots.ndim != 1:
            raise ValueError("Argument must be 'float' or a 'Array1D[float]'")
        table = {}
        for knot in list(set(knots)):
            table[knot] = np.sum(knots == knot)
        return table

    def knot_insert(self, knots: Union[float, Tuple[float]]) -> None:
        table = self.__transform_knots_to_table(knots)
        knotvector = np.array(self.knotvector).tolist()
        ctrlpoints = list(self.ctrlpoints)
        for knot, times in table.items():
            mult = self.knotvector.mult(knot)
            times = min(times, self.degree - mult)
            if times < 1:
                continue
            knotvector, ctrlpoints = Chapter5.CurveKnotIns(
                knotvector, ctrlpoints, knot, times
            )
        self.__set_UFP(knotvector, ctrlpoints)

    def knot_remove(self, knots: Union[float, Tuple[float]]) -> None:
        table = self.__transform_knots_to_table(knots)
        knotvector = list(self.knotvector)
        ctrlpoints = list(self.ctrlpoints)
        for knot, times in table.items():
            if knot not in knotvector:
                error_msg = f"Requested remove ({knot:.3f}),"
                error_msg += f" it's not in {knotvector}"
                raise ValueError(error_msg)
            result = Chapter5.RemoveCurveKnot(knotvector, ctrlpoints, knot, times)
            t, knotvector, ctrlpoints = result
            if t != times:
                error_msg = f"Cannot remove knot {knot}"
                error_msg += f" (only {t}/{times} times)"
                raise ValueError(error_msg)
        self.__set_UFP(knotvector, ctrlpoints)

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
        knotvector = list(self.knotvector)
        ctrlpoints = list(self.ctrlpoints)
        if self.degree + 1 == self.npts:  # If is bezier
            ctrlpoints = Custom.BezDegreeIncrease(ctrlpoints, times)
            knotvector = [0] * times + list(knotvector) + [1] * times
        else:
            oldknotvector = list(knotvector)
            knotvector += times * list(self.knotvector.knots)
            knotvector.sort()
            ctrlpoints = Custom.LeastSquareSpline(oldknotvector, ctrlpoints, knotvector)
        self.__set_UFP(knotvector, ctrlpoints)

    def __degree_decrease(self, times: int = 1, tolerance: float = 1e-9):
        knotvector = list(self.knotvector)
        ctrlpoints = list(self.ctrlpoints)
        if self.degree + 1 == self.npts:  # If is bezier
            ctrlpoints, error = Custom.BezDegreeReduce(ctrlpoints, times)
            knotvector = [0] * (self.npts - times) + [1] * (self.npts - times)
        else:
            for t in range(times):
                for knot in self.knotvector.knots:
                    if knot in knotvector:
                        knotvector.remove(knot)
            ctrlpoints = Custom.LeastSquareSpline(
                self.knotvector, ctrlpoints, knotvector
            )
            new_curve = self.__class__(knotvector, ctrlpoints)
            usample = np.linspace(1 / 5, 4 / 5, 129)
            oripoints = self.evaluate(usample)
            newpoints = new_curve.evaluate(usample)
            diff2 = np.abs(oripoints - newpoints)
            error = np.max(diff2)
        if error > tolerance:
            error_msg = f"Cannot reduce degree {times} times"
            error_msg += f"cause the error ({error:.1e}) is > {tolerance:.1e} "
            raise ValueError(error_msg)

        self.__set_UFP(knotvector, ctrlpoints)

    def degree_decrease(self, times: Optional[int] = 1):
        """
        The same as mycurve.degree -= 1
        But this function forces the degree reductions without looking the error
        """
        if self.degree - times < 1:
            error_msg = f"Cannot reduce curve {times} times."
            error_msg += f"Final degree would be {self.degree-times}"
            raise ValueError(error_msg)
        self.__degree_decrease(times, 1e9)

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

    def copy(self) -> Intface_BaseCurve:
        return self.__class__(self.knotvector, self.ctrlpoints)

    @classmethod
    def unite_curves(
        cls, curves: Tuple[Intface_BaseCurve], all_knots: Tuple[float]
    ) -> Intface_BaseCurve:
        for i in range(len(all_knots) - 1):
            if all_knots[i] >= all_knots[i + 1]:
                raise ValueError("Received knots are not sorted!")
        if len(all_knots) - 1 != len(curves):
            error_msg = f"You must have (k-1) curves ({len(curves)})"
            error_msg += f"for (k) knots ({len(all_knots)})"
            raise ValueError(error_msg)
        for i, curve in enumerate(curves):
            if not isinstance(curve, Intface_BaseCurve):
                error_msg = f"Curve[{i}] is type {type(curve)},"
                error_msg += "but it must be BaseCurve instance."
                raise TypeError(error_msg)
        for i in range(len(curves) - 1):
            if np.any(curves[i].ctrlpoints[-1] != curves[i + 1].ctrlpoints[0]):
                error_msg = f"Cannot unite curve[{i}] with curve[{i+1}]"
                error_msg += "cause the control points don't match"
                raise ValueError(error_msg)
        for i, curve in enumerate(curves):
            if curve.npts != curve.degree + 1:
                raise NotImplementedError  # Only bezier curves for now
        maximum_degree = 0
        for curve in curves:
            maximum_degree = max(maximum_degree, curve.degree)
        for curve in curves:
            curve.degree = maximum_degree
        all_ctrlpoints = []
        for curve in curves:
            all_ctrlpoints.append(curve.ctrlpoints)
        knotvector, ctrlpoints = Custom.UniteBezierCurvesSameDegree(
            all_knots, all_ctrlpoints
        )
        return cls(knotvector, ctrlpoints)

    def _split_into_bezier(self) -> Tuple[Intface_BaseCurve]:
        if self.degree + 1 == self.npts:  # is bezier
            return [self.copy()]
        all_knots = self.knotvector.knots
        for i in range(self.degree):  # We will insert knot maximum as possible
            self.knot_insert(all_knots)
        knotvector = list(self.knotvector)
        ctrlpoints = list(self.ctrlpoints)
        allctrls = Chapter5.DecomposeCurve(knotvector, ctrlpoints)
        listcurves = [0] * len(allctrls)
        p = self.degree
        U0, U1 = self.knotvector[0], self.knotvector[-1]
        Ubezier = [U0] * (p + 1) + [U1] * (p + 1)
        for i, ctpt in enumerate(allctrls):
            listcurves[i] = self.__class__(Ubezier, ctpt)
        return tuple(listcurves)

    def split(
        self, knots: Optional[Union[float, np.ndarray]] = None
    ) -> Tuple[Intface_BaseCurve]:
        if knots is None:
            return self._split_into_bezier()
        if isinstance(knots, (int, float)):
            knots = tuple([knots])
        knots = np.array(knots, dtype="float64")
        if knots.ndim != 1:
            raise ValueError
        knots = list(set(knots))
        knots.sort()
        knots = tuple(knots)
        copycurve = self.copy()
        for i in range(copycurve.degree):
            copycurve.knot_insert(knots)
        knotvector = np.array(list(copycurve.knotvector)[1:-1], dtype="float64")
        listcurves = [None] * (len(knots) + 1)
        all_knots = copycurve.knotvector.knots
        pairs = list(zip(all_knots[:-1], all_knots[1:]))
        for i, (ua, ub) in enumerate(pairs):
            middle = knotvector[(ua <= knotvector) * (knotvector <= ub)]
            middle = (middle - ua) / (ub - ua)
            newknotvect = [0] + list(middle) + [1]
            newnpts = len(newknotvect) - copycurve.degree - 1
            lowerind = i * (newnpts - 1)
            upperind = (i + 1) * (newnpts - 1) + 1
            newctrlpts = copycurve.ctrlpoints[lowerind:upperind]
            newcurve = copycurve.__class__(newknotvect, newctrlpts)
            listcurves[i] = newcurve
        return listcurves


class SplineCurve(BaseCurve):
    def __init__(self, knotvector: KnotVector, controlpoints: np.ndarray):
        super().__init__(knotvector, controlpoints)

    def _create_base_function_instance(self, knotvector: KnotVector):
        return SplineFunction(knotvector)


class RationalCurve(BaseCurve):
    def __init__(self, knotvector: KnotVector, controlpoints: np.ndarray):
        super().__init__(knotvector, controlpoints)

    def _create_base_function_instance(self, knotvector: KnotVector):
        return RationalFunction(knotvector)

    @property
    def weights(self):
        return self.F.weights

    @weights.setter
    def weights(self, value: Tuple[float]):
        self.F.weights = value
