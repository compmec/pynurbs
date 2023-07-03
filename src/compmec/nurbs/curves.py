from typing import Dict, Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Intface_BaseCurve
from compmec.nurbs.algorithms import *
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

    def __transform_knots_to_table(self, knots: Union[float, Tuple[float]]) -> Dict:
        """
        Receives a float or an array of floats.
        It returns a dictionary with the number of apearances of knot.
        Example:
            [0, 0.5, 0.5, 1] -> {0: 1, 0.5: 2, 1: 1}
            [0, 0.5, 1, 2, 2, 4] -> {0: 1, 0.5: 1, 2: 2, 4: 1}
        """
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
        T, _ = LeastSquare.spline(self.knotvector, newknotvector)
        newcontrolpoints = T @ self.ctrlpoints
        self.__set_UFP(newknotvector, newcontrolpoints)

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
        self, knots: Optional[Union[float, np.ndarray]] = None
    ) -> Tuple[Intface_BaseCurve]:
        """
        Separate the current spline at specified knots
        If no arguments are given, it splits at every knot and
            returns a list of bezier curves
        """
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
        copycurve = self.deepcopy()
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
