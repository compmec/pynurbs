from typing import Iterable, Optional, Tuple, Union

import numpy as np


class VerifyKnotVector(object):

    minU = 0
    maxU = 1

    @staticmethod
    def isFloatArray1D(U: Tuple[float]) -> None:
        try:
            Ud = np.array(U, dtype="float16")
        except TypeError as e:
            raise TypeError(
                f"Cannot convert U (type={type(U)}) into a numpy float array"
            )
        except ValueError as e:
            raise TypeError(f"All the elements inside U must be floats!")
        if Ud.ndim == 0:
            raise TypeError(
                f"Received U is type {type(U)}, but it's required a Tuple[float]"
            )
        if Ud.ndim != 1:
            raise ValueError(f"U is not a 1D array")

    @staticmethod
    def isOrdenedVector(U: Tuple[float]) -> None:
        n = len(U)
        for i in range(n - 1):
            if U[i] > U[i + 1]:
                raise ValueError("The given U must be ordened")

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
    def PN(p: int, n: int):
        VerifyKnotVector.isIntegerNonNegative(p)
        VerifyKnotVector.isIntegerNonNegative(n)
        if n <= p:
            raise ValueError("Must n > p. Received n=%d, p=%d" % (n, p))

    @staticmethod
    def all(U: Tuple[float]) -> None:
        VerifyKnotVector.isFloatArray1D(U)
        VerifyKnotVector.isOrdenedVector(U)
        VerifyKnotVector.Limits(U)
        VerifyKnotVector.SameQuantityBoundary(U)
        VerifyKnotVector.CountInternalValues(U)


class KnotVector(list):
    def __init__(self, U: Tuple[float]):
        VerifyKnotVector.all(U)
        super().__init__(U)
        self.compute_np()

    @property
    def p(self):
        return self.__p

    @property
    def n(self):
        return self.__n

    def compute_np(self):
        """
        We have that U = [0, ..., 0, ?, ..., ?, 1, ..., 1]
        And that U[p] = 0, but U[p+1] != 0
        The same way, U[n] = 1, but U[n-1] != 0

        Using that, we know that
            len(U) = m + 1 = n + p + 1
        That means that
            m = n + p
        """
        minU = min(self)
        p = 0
        while self[p + 1] == minU:
            p += 1
        n = len(self) - p - 1
        VerifyKnotVector.PN(p, n)
        self.__p = p
        self.__n = n

    def compute_spot_onevalue(self, u: float) -> int:
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

    def compute_spot(self, u: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        u = np.array(u)
        if u.ndim == 0:
            return self.compute_spot_onevalue(u)
        npts = u.shape[0]
        result = np.zeros([npts] + list(u.shape[1:]), dtype="int16")
        for i in range(npts):
            result[i] = self.compute_spot(u[i])
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

    def __insert_knot(self, knot: float, times: int):
        if times == 1:
            spot = self.compute_spot_onevalue(knot)
            copylist = list(self)
            copylist.insert(spot + 1, knot)
            VerifyKnotVector.all(copylist)
            self.insert(spot + 1, knot)
            self.compute_np()
            return
        for i in range(times):
            self.__insert_knot(knot, 1)

    def insert_knot(self, knot: float, times: Optional[int] = 1):
        self.verify_insert_remove_knot(knot, times)
        self.__insert_knot(knot, times)

    def __remove_knot(self, knot: float, times: int):
        if times == 1:
            spot = self.compute_spot_onevalue(knot)
            self.remove(knot)
            self.compute_np()
            return
        for i in range(times):
            self.__remove_knot(knot, 1)

    def remove_knot(self, knot: float, times: Optional[int] = 1):
        self.verify_insert_remove_knot(knot, times)
        if knot not in self:
            raise ValueError(f"Cannot remove knot {knot} cause it's not in {self}")
        self.__remove_knot(knot, times)

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
        if self.n != obj.n:
            return False
        if self.p != obj.p:
            return False
        for i, v in enumerate(self):
            if v != obj[i]:
                return False
        return True

    def __ne__(self, __obj: object) -> bool:
        return not self.__eq__(__obj)


class GeneratorKnotVector:
    @staticmethod
    def bezier(p: int) -> KnotVector:
        VerifyKnotVector.isIntegerNonNegative(p)
        return GeneratorKnotVector.uniform(p=p, n=p + 1)

    @staticmethod
    def weight(p: int, ws: Tuple[float]) -> KnotVector:
        VerifyKnotVector.isFloatArray1D(ws)
        VerifyKnotVector.isIntegerNonNegative(p)
        U = np.cumsum(ws)
        U -= U[0]
        U /= U[-1]
        U *= VerifyKnotVector.maxU - VerifyKnotVector.minU
        U += VerifyKnotVector.minU
        U = p * [0] + list(U) + p * [1]
        return KnotVector(U)

    @staticmethod
    def uniform(p: int, n: int) -> KnotVector:
        VerifyKnotVector.PN(p, n)
        ws = np.ones(n - p + 1)
        return GeneratorKnotVector.weight(p=p, ws=ws)

    @staticmethod
    def random(p: int, n: int) -> KnotVector:
        VerifyKnotVector.PN(p, n)
        ws = np.random.rand(n - p + 1)
        return GeneratorKnotVector.weight(p, ws)
