from typing import Iterable, Tuple, Union

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


class KnotVector(list):
    def __init__(self, U: Iterable[float]):
        VerifyKnotVector.all(U)
        super().__init__(U)
        self.compute_np()

    @property
    def p(self):
        return self.__p

    @property
    def n(self):
        return self.__n

    @p.setter
    def p(self, value: int):
        VerifyKnotVector.isIntegerNonNegative(value)
        self.__p = int(value)

    @n.setter
    def n(self, value: int):
        VerifyKnotVector.isIntegerNonNegative(value)
        VerifyKnotVector.PN(self.p, value)
        self.__n = int(value)

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
        self.p = p
        self.n = len(self) - p - 1

    def __find_spot_onevalue(self, u: float) -> int:
        U = np.array(self)
        minU = np.min(self)
        maxU = np.max(self)
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

    def __find_spot_vector(self, u: Iterable[float]) -> np.ndarray:
        """
        find the spots of an vector. It's not very efficient, but ok for the moment
        """
        spots = np.zeros(len(u), dtype="int64")
        for i, ui in enumerate(u):
            spots[i] = self.__find_spot_onevalue(ui)
        return spots

    def spot(self, u: Union[float, Iterable[float]]) -> Union[int, np.ndarray]:
        if isinstance(u, np.ndarray):
            return self.__find_spot_vector(u)
        else:
            return np.array(self.__find_spot_onevalue(u))

    def __eq__(self, __obj: object):
        if not isinstance(__obj, (list, tuple, self.__class__)):
            raise TypeError(
                f"Cannot compare {type(__obj)} with a {self.__class__} instance"
            )
        try:
            __obj = self.__class__(__obj)
        except Exception as e:
            raise ValueError(
                f"No sucess trying to convert {type(__obj)} into {self.__class__}. Cause {str(e)}"
            )
        if self.n != __obj.n:
            return False
        if self.p != __obj.p:
            return False
        for i, v in enumerate(self):
            if v != __obj[i]:
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
        ws = np.random.random(n - p + 1)
        return GeneratorKnotVector.weight(p, ws)
