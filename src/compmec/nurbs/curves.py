from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Interface_BaseCurve
from compmec.nurbs.basefunctions import (
    BaseFunction,
    RationalBaseFunction,
    RationalWeightsVector,
    SplineBaseFunction,
)
from compmec.nurbs.degreeoperations import degree_decrease, degree_increase
from compmec.nurbs.knotoperations import insert_knot, remove_knot
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
    def p(self):
        return self.U.p

    @property
    def n(self):
        return self.U.n

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
        value = np.array(value, dtype="float64")
        if value.shape[0] != self.n:
            error_msg = f"The number of control points must be the same of degrees of freedom of KnotVector.\n"
            error_msg += f"    U.n = {self.n} != {len(value)} = len(P)"
            raise ValueError(error_msg)
        self.__P = value

    def __set_UFP(self, U: KnotVector, P: np.ndarray):
        U = KnotVector(U)
        self.__F = self._create_base_function_instance(U)
        self.P = P

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

    def knot_insert(self, knots: Union[float, Tuple[float]]):
        newP = np.copy(self.P)
        newF = self.F
        knots = np.array(knots, dtype="float64")
        if knots.ndim == 0:
            newF, newP = insert_knot(newF, newP, float(knots))
        elif knots.ndim == 1:
            for knot in knots:
                newF, newP = insert_knot(newF, newP, knot)
        else:
            raise ValueError
        self.__set_UFP(newF.U, newP)

    def knot_remove(self, knots: Union[float, Tuple[float]]):
        newP = np.copy(self.P)
        newF = self.F
        knots = np.array(knots, dtype="float64")
        if knots.ndim == 0:
            newF, newP = remove_knot(newF, newP, float(knots))
        elif knots.ndim == 1:
            for knot in knots:
                newF, newP = remove_knot(newF, newP, float(knot))
        else:
            raise ValueError
        self.__set_UFP(newF.U, newP)

    def degree_increase(self, times: Optional[int] = 1):
        newP = np.copy(self.P)
        newF = self.F
        for i in range(times):
            newF, newP = degree_increase(newF, newP)
            print("newF = ", newF)
            print("newP = ", newP)
        self.__set_UFP(newF.U, newP)

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
        self.w = np.ones(self.n, dtype="float64")

    def _create_base_function_instance(self, U: KnotVector):
        return RationalBaseFunction(U)

    def __eq__(self, obj):
        if not super().__eq__(obj):
            return False
        if np.any(self.w != obj.w):
            return False
        return True
