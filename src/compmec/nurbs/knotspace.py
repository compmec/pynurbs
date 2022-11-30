from typing import Iterable, Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Interface_KnotVector


class VerifyKnotVector(object):

    minU = 0
    maxU = 1

    @staticmethod
    def isFloatArray1D(U: Tuple[float]) -> None:
        if not isinstance(U, (tuple, list, np.ndarray)):
            error_msg = f"Cannot convert U (type={type(U)}) into a numpy float array"
            raise TypeError(error_msg)
        try:
            Ud = np.array(U, dtype="float16")
        except TypeError as e:
            error_msg = f"Cannot convert U (type={type(U)}) into a numpy float array"
            raise TypeError(error_msg)
        except ValueError as e:
            raise TypeError(f"All the elements inside U must be floats! Received {U}")
        if Ud.ndim == 0:
            raise TypeError(
                f"Received U is type {type(U)}, but it's required a Tuple[float]"
            )
        if Ud.ndim != 1:
            raise ValueError(f"U is not a 1D array")

    @staticmethod
    def isOrdenedVector(U: Tuple[float]) -> None:
        npts = len(U)
        for i in range(npts - 1):
            if U[i] > U[i + 1]:
                raise ValueError("The given KnotVector must be ordened")

    @staticmethod
    def Limits(U: Tuple[float]) -> None:
        if VerifyKnotVector.minU is not None:
            VerifyKnotVector.InferiorLimit(U)
        if VerifyKnotVector.maxU is not None:
            VerifyKnotVector.SuperiorLimit(U)

    @staticmethod
    def InferiorLimit(U: Tuple[float]) -> None:
        for u in U:
            if u < VerifyKnotVector.minU:
                raise ValueError(
                    f"All the values in U must be bigger than {VerifyKnotVector.minU}"
                )

    @staticmethod
    def SuperiorLimit(U: Tuple[float]) -> None:
        for u in U:
            if u > VerifyKnotVector.maxU:
                raise ValueError(
                    f"All the values in U must be less than {VerifyKnotVector.maxU}"
                )

    @staticmethod
    def SameQuantityBoundary(U: Tuple[float]) -> None:
        U = np.array(U)
        minU = np.min(U)
        maxU = np.max(U)
        if np.sum(U == minU) != np.sum(U == maxU):
            raise ValueError("U must contain the same quantity of 0 and 1. U = ", U)

    @staticmethod
    def CountInternalValues(U: Tuple[float]) -> None:
        setU = list(set(U))
        U = np.array(U)
        minU = np.min(U)
        maxU = np.max(U)
        setU.remove(minU)
        setU.remove(maxU)
        qttmin = np.sum(U == minU)
        for u in setU:
            if np.sum(U == u) > qttmin:
                raise ValueError

    @staticmethod
    def isInteger(val: int):
        if not isinstance(val, int):
            try:
                int(val)
            except Exception as e:
                raise TypeError(f"The value must be an integer. Received {type(val)}")

    @staticmethod
    def isNonNegative(val: int):
        if val < 0:
            raise ValueError(f"The value must be > 0. Received {val}")

    @staticmethod
    def isIntegerNonNegative(value: int):
        VerifyKnotVector.isInteger(value)
        VerifyKnotVector.isNonNegative(value)

    @staticmethod
    def PN(degree: int, npts: int):
        VerifyKnotVector.isIntegerNonNegative(degree)
        VerifyKnotVector.isIntegerNonNegative(npts)
        if npts <= degree:
            raise ValueError(
                "Must npts > degree. Received npts=%d, degree=%d" % (npts, degree)
            )

    @staticmethod
    def all(U: Tuple[float]) -> None:
        if isinstance(U, Interface_KnotVector):
            return
        VerifyKnotVector.isFloatArray1D(U)
        VerifyKnotVector.isOrdenedVector(U)
        VerifyKnotVector.Limits(U)
        VerifyKnotVector.SameQuantityBoundary(U)
        VerifyKnotVector.CountInternalValues(U)


class KnotVector(list):
    def __init__(self, U: Tuple[float]):
        VerifyKnotVector.all(U)
        degree, npts = self.compute_pn(U)
        VerifyKnotVector.PN(degree, npts)
        self.__degree = degree
        self.__npts = npts
        super().__init__(U)

    @property
    def degree(self) -> int:
        return self.__degree

    @property
    def npts(self) -> int:
        return self.__npts

    @staticmethod
    def compute_pn(U: Tuple[float]):
        """
        We have that U = [0, ..., 0, ?, ..., ?, 1, ..., 1]
        And that U[degree] = 0, but U[degree+1] != 0
        The same way, U[npts] = 1, but U[npts-1] != 0

        Using that, we know that
            len(U) = m + 1 = npts + degree + 1
        That means that
            m = npts + degree
        """
        minU = min(U)
        degree = 0
        while U[degree + 1] == minU:
            degree += 1
        npts = len(U) - degree - 1
        return degree, npts

    def span_onevalue(self, u: float) -> int:
        try:
            u = float(u)
        except Exception as e:
            raise TypeError
        U = np.array(self)
        minU = np.min(self)
        maxU = np.max(self)
        if u < minU:
            raise ValueError(f"Received u = {u} < minU = {minU}")
        if maxU < u:
            raise ValueError(f"Received u = {u} > maxU = {maxU}")
        lower = int(np.max(np.where(U == minU)))
        upper = int(np.min(np.where(U == maxU)))
        if u == minU:
            return lower
        if u == maxU:
            return upper
        mid = (lower + upper) // 2
        while True:
            if u < U[mid]:
                upper = mid
            elif U[mid + 1] <= u:
                lower = mid
            else:
                return mid
            mid = (lower + upper) // 2

    def span(self, u: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        u = np.array(u)
        if u.ndim == 0:
            return self.span_onevalue(u)
        npts = u.shape[0]
        result = [0] * (npts)
        for i in range(npts):
            result[i] = self.span(u[i])
        return np.array(result, dtype="int16")

    def mult_onevalue(self, u: float) -> int:
        if not (min(self) <= u <= max(self)):
            raise ValueError
        return np.sum(np.abs(np.array(self) - u) < 1e-12)

    def mult(self, u: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        u = np.array(u)
        if u.ndim == 0:
            return self.mult_onevalue(u)
        npts = u.shape[0]
        result = np.zeros([npts] + list(u.shape[1:]), dtype="int16")
        for i in range(npts):
            result[i] = self.mult(u[i])
        return result

    def verify_insert_remove_knot(self, knot: float, times: Optional[int] = 1):
        if not isinstance(times, int):
            raise TypeError
        if times < 1:
            raise ValueError
        if not isinstance(knot, float):
            raise TypeError
        if not (min(self) <= knot <= max(self)):
            raise ValueError

    def __knot_insert(self, knot: float, times: int):
        if times == 1:
            span = self.span_onevalue(knot)
            copylist = list(self)
            copylist.insert(span + 1, knot)
            VerifyKnotVector.all(copylist)
            self.insert(span + 1, knot)
            degree, npts = self.compute_pn(list(self))
            VerifyKnotVector.PN(degree, npts)
            self.__degree = degree
            self.__npts = npts
            return
        for i in range(times):
            self.__knot_insert(knot, 1)

    def knot_insert(self, knot: float, times: Optional[int] = 1):
        self.verify_insert_remove_knot(knot, times)
        self.__knot_insert(knot, times)

    def __knot_remove(self, knot: float, times: int):
        if times == 1:
            span = self.span_onevalue(knot)
            self.remove(knot)
            degree, npts = self.compute_pn(list(self))
            VerifyKnotVector.PN(degree, npts)
            self.__degree = degree
            self.__npts = npts
            return
        for i in range(times):
            self.__knot_remove(knot, 1)

    def knot_remove(self, knot: float, times: Optional[int] = 1):
        self.verify_insert_remove_knot(knot, times)
        if knot not in self:
            raise ValueError(f"Cannot remove knot {knot} cause it's not in {self}")
        self.__knot_remove(knot, times)

    def __eq__(self, obj: object):
        if not isinstance(obj, (list, tuple, self.__class__)):
            raise TypeError(
                f"Cannot compare {type(obj)} with a {self.__class__} instance"
            )
        try:
            obj = self.__class__(obj)
        except Exception as e:
            raise ValueError(
                f"No sucess trying to convert {type(obj)} into {self.__class__}. Cause {str(e)}"
            )
        print("Comparing!")
        print("    self.npts = ", self.npts)
        print("  self.degree = ", self.degree)
        print(" ---")
        print("     obj.npts = ", obj.npts)
        print("   obj.degree = ", obj.degree)
        if self.npts != obj.npts:
            return False
        if self.degree != obj.degree:
            return False
        for i, v in enumerate(self):
            if v != obj[i]:
                return False
        return True

    def __ne__(self, __obj: object) -> bool:
        return not self.__eq__(__obj)


class GeneratorKnotVector:
    @staticmethod
    def bezier(degree: int) -> KnotVector:
        VerifyKnotVector.isIntegerNonNegative(degree)
        return GeneratorKnotVector.uniform(degree, degree + 1)

    @staticmethod
    def weight(degree: int, ws: Tuple[float]) -> KnotVector:
        VerifyKnotVector.isFloatArray1D(ws)
        VerifyKnotVector.isIntegerNonNegative(degree)
        U = np.cumsum(ws)
        U -= U[0]
        U /= U[-1]
        U *= VerifyKnotVector.maxU - VerifyKnotVector.minU
        U += VerifyKnotVector.minU
        U = degree * [0] + list(U) + degree * [1]
        return KnotVector(U)

    @staticmethod
    def uniform(degree: int, npts: int) -> KnotVector:
        VerifyKnotVector.PN(degree, npts)
        ws = np.ones(npts - degree + 1)
        return GeneratorKnotVector.weight(degree, ws)

    @staticmethod
    def random(degree: int, npts: int) -> KnotVector:
        VerifyKnotVector.PN(degree, npts)
        ws = np.random.rand(npts - degree + 1)
        return GeneratorKnotVector.weight(degree, ws)
