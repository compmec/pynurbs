from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Interface_BaseCurve
from compmec.nurbs.basefunctions import (
    BaseFunction,
    RationalBaseFunction,
    SplineBaseFunction,
)
from compmec.nurbs.degreeoperations import degree_decrease, degree_increase
from compmec.nurbs.knotoperations import insert_knot, remove_knot
from compmec.nurbs.knotspace import KnotVector


class BaseCurve(Interface_BaseCurve):
    def __init__(self, U: KnotVector, P: np.ndarray):
        self.__set_FP(U, P)

    def __call__(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        L = self.F(u)
        return L.T @ self.P

    def derivate(self):
        self.F.derivate()

    @property
    def p(self):
        return self.U.p

    @property
    def n(self):
        return self.U.n

    @property
    def U(self):
        return self.__U

    @property
    def P(self):
        return self.__P

    def __set_FP(self, F: BaseFunction, P: np.ndarray):
        if not isinstance(F, BaseFunction):
            raise TypeError(f"F must be a BaseFunction instance, not {type(F)}")
        P = np.array(P, dtype="float64")
        if F.n != P.shape[0]:
            error_msg = f"The number of control points must be the same of degrees of freedom of {type(F)}."
            error_msg += f"F.n = {self.F.n} != {len(P)} = len(P)"
            raise ValueError(error_msg)

        self.__F = F
        self.__P = P

    def __eq__(self, obj: object) -> bool:
        if type(self) != type(obj):
            raise TypeError(
                f"Cannot compare a {type(obj)} object with a {self.__class__} object"
            )
        knots = []
        for ui in self.F.U:
            if ui not in knots:
                knots.append(ui)
        utest = []
        for i in range(len(knots) - 1):
            utest += list(
                np.linspace(knots[i], knots[i + 1], 1 + self.F.p, endpoint=False)
            )
        utest += [knots[-1]]
        utest = np.array(utest)
        Cusel = self(utest)
        Cuobj = obj(utest)
        return np.all(np.abs(Cusel - Cuobj) < 1e-9)

    def __ne__(self, __obj: object):
        return not self.__eq__(__obj)

    def __neg__(self):
        return self.__class__(self.F, np.copy(-self.P))

    def __add__(self, __obj: object):
        if not isinstance(__obj, self.__class__):
            raise TypeError(
                f"Cannot sum a {type(__obj)} object with a {self.__class__} object"
            )
        if self.U != __obj.U:
            raise ValueError("The vectors of curves are not the same!")
        if self.P.shape != __obj.P.shape:
            raise ValueError("The shape of control points are not the same!")
        newP = np.copy(self.P) + __obj.P
        return self.__class__(self.F, newP)

    def __sub__(self, __obj: object):
        if not isinstance(__obj, self.__class__):
            raise TypeError(
                f"Cannot sum a {type(__obj)} object with a {self.__class__} object"
            )
        return self + (-__obj)

    def knot_insert(self, knots: Tuple[float]):
        newP = np.copy(self.P)
        newF = self.F
        for knot in knots:
            newF, newP = insert_knot(self.F, self.P, knot)
        self.__set_FP(newF, newP)

    def knot_remove(self, knots: Tuple[float]):
        newP = np.copy(self.P)
        newF = self.F
        for knot in knots:
            newF, newP = remove_knot(self.F, self.P, knot)
        self.__set_FP(newF, newP)

    def degree_increase(self, times: Optional[int] = 1):
        newP = np.copy(self.P)
        newF = self.F
        for i in range(times):
            newF, newP = degree_increase(self.F, self.P)
        self.__set_FP(newF, newP)

    def degree_decrease(self, times: Optional[int] = 1):
        newP = np.copy(self.P)
        newF = self.F
        for i in range(times):
            newF, newP = degree_decrease(self.F, self.P)
        self.__set_FP(newF, newP)


class SplineCurve(BaseCurve):
    def __init__(self, F: SplineBaseFunction, controlpoints: np.ndarray):
        super().__init__(F, controlpoints)


class RationalCurve(BaseCurve):
    def __init__(self, f: RationalBaseFunction, controlpoints: np.ndarray):
        super().__init__(f, controlpoints)
