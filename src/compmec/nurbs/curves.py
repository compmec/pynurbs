from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Interface_BaseCurve
from compmec.nurbs.algorithms import Chapter5
from compmec.nurbs.basefunctions import (
    BaseFunction,
    RationalBaseFunction,
    RationalWeightsVector,
    SplineBaseFunction,
)
from compmec.nurbs.knotspace import KnotVector


class BaseCurve(Interface_BaseCurve):
    def __init__(self, knotvector: KnotVector, ctrlpoints: np.ndarray):
        self.__set_UFP(knotvector, ctrlpoints)

    def __call__(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        L = self.F(u)
        return L.T @ self.ctrlpoints

    def derivate(self):
        self.F.derivate()

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

    @ctrlpoints.setter
    def ctrlpoints(self, value: np.ndarray):
        try:
            value = np.array(value, dtype="float64")
        except Exception as e:
            error_msg = f"Received Control Points is type {type(value)}, but it must be an float-array"
            raise TypeError(error_msg)
        if value.ndim == 0:
            error_msg = f"The Control Points must be a array, not a single value"
            raise TypeError(error_msg)
        if value.shape[0] != self.npts:
            error_msg = f"The number of control points must be the same of degrees of freedom of KnotVector.\npts"
            error_msg += (
                f"    knotvector.npts = {self.npts} != {len(value)} = len(ctrlpoints)"
            )
            raise ValueError(error_msg)
        self.__ctrlpoints = value

    def __set_UFP(self, knotvector: KnotVector, ctrlpoints: np.ndarray):
        knotvector = KnotVector(knotvector)
        self.__F = self._create_base_function_instance(knotvector)
        self.ctrlpoints = ctrlpoints

    def __eq__(self, obj: object) -> bool:
        if type(self) != type(obj):
            error_msg = (
                f"Cannot compare a {type(obj)} object with a {self.__class__} object"
            )
            raise TypeError(error_msg)
        knots = list(set(self.knotvector))
        knots.sort()
        utest = [
            np.linspace(a, b, 2 + self.degree) for a, b in zip(knots[:-1], knots[1:])
        ]
        utest = list(set(np.array(utest).reshape(-1)))
        Cusel = self(utest)
        Cuobj = obj(utest)
        return np.all(np.abs(Cusel - Cuobj) < 1e-9)

    def __ne__(self, __obj: object):
        return not self.__eq__(__obj)

    def __neg__(self):
        return self.__class__(self.knotvector, np.copy(-self.ctrlpoints))

    def __add__(self, __obj: object):
        if type(self) != type(__obj):
            error_msg = (
                f"Cannot sum a {type(__obj)} object with a {self.__class__} object"
            )
            raise TypeError(error_msg)
        if self.knotvector != __obj.knotvector:
            raise ValueError("The vectors of curves are not the same!")
        if self.ctrlpoints.shape != __obj.ctrlpoints.shape:
            raise ValueError("The shape of control points are not the same!")
        newP = np.copy(self.ctrlpoints) + __obj.ctrlpoints
        return self.__class__(self.knotvector, newP)

    def __sub__(self, __obj: object):
        return self + (-__obj)

    def __transform_knots_to_table(self, knots: Union[float, Tuple[float]]):
        try:
            knots = float(knots)
            return {knots: 1}
        except Exception as e:
            pass
        try:
            iter(knots)
            knots = np.array(knots, dtype="float64")
        except Exception as e:
            raise TypeError
        if knots.ndim != 1:
            raise ValueError("Argument must be 'float' or a 'Array1D[float]'")
        table = {}
        for knot in list(set(knots)):
            table[knot] = np.sum(knots == knot)
        return table

    def knot_insert(self, knots: Union[float, Tuple[float]]):
        table = self.__transform_knots_to_table(knots)
        knotvector = np.array(self.knotvector).tolist()
        ctrlpoints = list(self.ctrlpoints)
        for knot, times in table.items():
            knotvector, ctrlpoints = Chapter5.CurveKnotIns(
                knotvector, ctrlpoints, knot, times
            )
        self.__set_UFP(knotvector, ctrlpoints)

    def knot_remove(self, knots: Union[float, Tuple[float]]):
        table = self.__transform_knots_to_table(knots)
        knotvector = list(self.knotvector)
        ctrlpoints = list(self.ctrlpoints)
        for knot, times in table.items():
            t, knotvector, ctrlpoints = Chapter5.RemoveCurveKnot(
                knotvector, ctrlpoints, knot, times
            )
            if t != times:
                if t == 0:
                    error_msg = f"Cannot remove the knot {knot}"
                else:
                    error_msg = (
                        f"Can remove knot {knot} only {t} times (requested {times})"
                    )
                raise ValueError(error_msg)
        self.__set_UFP(knotvector, ctrlpoints)

    def degree_increase(self, times: Optional[int] = 1):
        knotvector = list(self.knotvector)
        ctrlpoints = list(self.ctrlpoints)
        knotvector, ctrlpoints = Chapter5.DegreeElevateCurve(
            knotvector, ctrlpoints, times
        )
        self.__set_UFP(knotvector, ctrlpoints)

    def degree_decrease(self, times: Optional[int] = 1):
        knotvector = list(self.knotvector)
        ctrlpoints = list(self.ctrlpoints)
        for i in range(times):
            knotvector, ctrlpoints = Chapter5.DegreeReduceCurve(knotvector, ctrlpoints)
        self.__set_UFP(knotvector, ctrlpoints)


class SplineCurve(BaseCurve):
    def __init__(self, knotvector: KnotVector, controlpoints: np.ndarray):
        super().__init__(knotvector, controlpoints)

    def _create_base_function_instance(self, knotvector: KnotVector):
        return SplineBaseFunction(knotvector)


class RationalCurve(BaseCurve, RationalWeightsVector):
    def __init__(self, knotvector: KnotVector, controlpoints: np.ndarray):
        super().__init__(knotvector, controlpoints)
        self.w = np.ones(self.npts, dtype="float64")

    def _create_base_function_instance(self, knotvector: KnotVector):
        return RationalBaseFunction(knotvector)

    def __eq__(self, obj):
        if type(self) != type(obj):
            raise TypeError
        if np.any(self.w != obj.w):
            return False
        if not super().__eq__(obj):
            return False
        return True
