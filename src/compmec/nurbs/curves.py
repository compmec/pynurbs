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
    def __init__(self, U: KnotVector, P: np.ndarray):
        self.__set_UFP(U, P)

    def __call__(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        L = self.F(u)
        return L.T @ self.P

    def derivate(self):
        self.F.derivate()

    @property
    def degree(self):
        return self.U.degree

    @property
    def npts(self):
        return self.U.npts

    @property
    def U(self):
        return self.F.U

    @property
    def F(self):
        return self.__F

    @property
    def P(self):
        return self.__P

    @P.setter
    def P(self, value: np.ndarray):
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
            error_msg += f"    U.npts = {self.npts} != {len(value)} = len(P)"
            raise ValueError(error_msg)
        self.__P = value

    def __set_UFP(self, U: KnotVector, P: np.ndarray):
        U = KnotVector(U)
        self.__F = self._create_base_function_instance(U)
        self.P = P

    def __eq__(self, obj: object) -> bool:
        if type(self) != type(obj):
            error_msg = (
                f"Cannot compare a {type(obj)} object with a {self.__class__} object"
            )
            raise TypeError(error_msg)
        knots = list(set(self.U))
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
        return self.__class__(self.U, np.copy(-self.P))

    def __add__(self, __obj: object):
        if type(self) != type(__obj):
            error_msg = (
                f"Cannot sum a {type(__obj)} object with a {self.__class__} object"
            )
            raise TypeError(error_msg)
        if self.U != __obj.U:
            raise ValueError("The vectors of curves are not the same!")
        if self.P.shape != __obj.P.shape:
            raise ValueError("The shape of control points are not the same!")
        newP = np.copy(self.P) + __obj.P
        return self.__class__(self.U, newP)

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
        npts, degree = self.npts, self.degree
        U = np.array(self.U).tolist()
        P = list(self.P)
        for knot, times in table.items():
            U = KnotVector(U)
            span = U.span(knot)
            mult = U.mult(knot)
            npts, U, P = Chapter5.CurveKnotIns(
                npts, degree, U, P, knot, span, mult, times
            )
        self.__set_UFP(U, P)

    def knot_remove(self, knots: Union[float, Tuple[float]]):
        table = self.__transform_knots_to_table(knots)
        npts, degree = self.npts, self.degree
        U = np.array(self.U).tolist()
        P = list(self.P)
        for knot, times in table.items():
            U = KnotVector(U)
            span = U.span(knot)
            mult = U.mult(knot)
            t, U, P = Chapter5.RemoveCurveKnot(
                npts, degree, U, P, knot, span, mult, times
            )
            if t != times:
                if t == 0:
                    error_msg = f"Cannot remove the knot {knot}"
                else:
                    error_msg = (
                        f"Can remove knot {knot} only {t} times (requested {times})"
                    )
                raise ValueError(error_msg)
        self.__set_UFP(U, P)

    def degree_increase(self, times: Optional[int] = 1):
        U = list(self.U)
        P = list(self.P)
        nq, Uq, Qw = Chapter5.DegreeElevateCurve(self.npts, self.degree, U, P, times)
        Uq = KnotVector(Uq)
        Qw = np.array(Qw)
        self.__set_UFP(Uq, Qw)

    def degree_decrease(self, times: Optional[int] = 1):
        newP = np.copy(self.P)
        newF = self.F
        for i in range(times):
            newF, newP = degree_decrease(newF, newP)
        self.__set_UFP(newF.U, newP)


class SplineCurve(BaseCurve):
    def __init__(self, U: KnotVector, controlpoints: np.ndarray):
        super().__init__(U, controlpoints)

    def _create_base_function_instance(self, U: KnotVector):
        return SplineBaseFunction(U)


class RationalCurve(BaseCurve, RationalWeightsVector):
    def __init__(self, U: KnotVector, controlpoints: np.ndarray):
        super().__init__(U, controlpoints)
        self.w = np.ones(self.npts, dtype="float64")

    def _create_base_function_instance(self, U: KnotVector):
        return RationalBaseFunction(U)

    def __eq__(self, obj):
        if type(self) != type(obj):
            raise TypeError
        if np.any(self.w != obj.w):
            return False
        if not super().__eq__(obj):
            return False
        return True
