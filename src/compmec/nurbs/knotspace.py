from typing import Iterable, Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Interface_KnotVector


class ValidationKnotVector(object):

    minU = 0
    maxU = 1

    @staticmethod
    def isFloatArray1D(knotvector: Tuple[float]) -> None:
        if not isinstance(knotvector, (tuple, list, np.ndarray)):
            error_msg = f"Cannot convert knotvector (type={type(knotvector)}) into a numpy float array"
            raise TypeError(error_msg)
        for i, knot in enumerate(knotvector):
            if not isinstance(knot, (int, float)):
                error_msg = f"knot vector must be 1D array of floats. Type(knotvector[{i}]) = {type(knot)} "
                raise TypeError(error_msg)

    @staticmethod
    def isOrdenedVector(knotvector: Tuple[float]) -> None:
        npts = len(knotvector)
        for i in range(npts - 1):
            if knotvector[i] > knotvector[i + 1]:
                error_msg = f"The given KnotVector must be ordened.\n"
                error_msg += f"    knotvector = {list(knotvector)}"
                raise ValueError(error_msg)

    @staticmethod
    def Limits(knotvector: Tuple[float]) -> None:
        if ValidationKnotVector.minU is not None:
            ValidationKnotVector.InferiorLimit(knotvector)
        if ValidationKnotVector.maxU is not None:
            ValidationKnotVector.SuperiorLimit(knotvector)

    @staticmethod
    def InferiorLimit(knotvector: Tuple[float]) -> None:
        minU = ValidationKnotVector.minU
        for u in knotvector:
            if u < minU:
                error_msg = f"All the values in knotvector must be >= {minU}"
                raise ValueError(error_msg)

    @staticmethod
    def SuperiorLimit(knotvector: Tuple[float]) -> None:
        maxU = ValidationKnotVector.maxU
        for u in knotvector:
            if u > maxU:
                error_msg = f"All the values in knotvector must be <= {maxU}"
                raise ValueError(error_msg)

    @staticmethod
    def SameQuantityBoundary(knotvector: Tuple[float]) -> None:
        knotvector = np.array(knotvector)
        minU = np.min(knotvector)
        maxU = np.max(knotvector)
        if np.sum(knotvector == minU) != np.sum(knotvector == maxU):
            error_msg = (
                "knot vector must contain the same quantity of {minU} and {maxU}."
            )
            raise ValueError(error_msg)

    @staticmethod
    def CountInternalMultiplicity(knotvector: Tuple[float]) -> None:
        setU = list(set(knotvector))
        setU.remove(knotvector[-1])
        setU.remove(knotvector[0])
        if setU == []:
            return
        knotvector = np.array(knotvector)
        degree = np.sum(knotvector == knotvector[0]) - 1
        multknots = [np.sum(knotvector == u) for u in setU]
        maxmultknots = np.max(multknots)
        if degree != 0:
            if maxmultknots > degree:
                u = knotvector[np.where(multknots == maxmultknots)[0]]
                error_msg = (
                    f"The knot {u} has multiplicty {maxmultknots} > degree = {degree}"
                )
                raise ValueError(error_msg)
        else:
            if maxmultknots > 1:
                error_msg = "For degree = 0, each knot must appear only once"
                raise ValueError(error_msg)

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
        ValidationKnotVector.isInteger(value)
        ValidationKnotVector.isNonNegative(value)

    @staticmethod
    def PN(degree: int, npts: int):
        ValidationKnotVector.isIntegerNonNegative(degree)
        ValidationKnotVector.isIntegerNonNegative(npts)
        if npts <= degree:
            raise ValueError(
                "Must npts > degree. Received npts=%d, degree=%d" % (npts, degree)
            )

    @staticmethod
    def all(knotvector: Tuple[float]) -> None:
        if isinstance(knotvector, Interface_KnotVector):
            return
        try:
            ValidationKnotVector.isFloatArray1D(knotvector)
            ValidationKnotVector.isOrdenedVector(knotvector)
            ValidationKnotVector.Limits(knotvector)
            ValidationKnotVector.SameQuantityBoundary(knotvector)
            ValidationKnotVector.CountInternalMultiplicity(knotvector)
        except Exception as e:
            error_msg = f"The received knot vector is not valid!\n"
            error_msg += f"   type = {type(knotvector)}\n"
            error_msg += f"  value = {str(knotvector)[:400]}\n"
            error_msg += f"  Cause = {str(e)}"
            raise e.__class__(error_msg)


class KnotVector(Interface_KnotVector, list):
    def __init__(self, knotvector: Tuple[float]):
        ValidationKnotVector.all(knotvector)
        degree, npts = self.compute_pn(knotvector)
        self.__degree = degree
        self.__npts = npts
        super().__init__(knotvector)

    @property
    def degree(self) -> int:
        return self.__degree

    @property
    def npts(self) -> int:
        return self.__npts

    @property
    def knots(self) -> Tuple[float]:
        knts = list(set(self))
        knts.sort()
        return tuple(knts)

    @staticmethod
    def compute_pn(knotvector: Tuple[float]):
        """
        We have that knotvector = [0, ..., 0, ?, ..., ?, 1, ..., 1]
        And that knotvector[degree] = 0, but knotvector[degree+1] != 0
        The same way, knotvector[npts] = 1, but knotvector[npts-1] != 0

        Using that, we know that
            len(knotvector) = m + 1 = npts + degree + 1
        That means that
            m = npts + degree
        """
        minU = min(knotvector)
        degree = 0
        while knotvector[degree + 1] == minU:
            degree += 1
        npts = len(knotvector) - degree - 1
        return degree, npts

    def span_onevalue(self, u: float) -> int:
        try:
            u = float(u)
        except Exception as e:
            raise TypeError
        knotvector = np.array(self)
        minU = np.min(self)
        maxU = np.max(self)
        if u < minU:
            raise ValueError(f"Received u = {u} < minU = {minU}")
        if maxU < u:
            raise ValueError(f"Received u = {u} > maxU = {maxU}")
        lower = int(np.max(np.where(knotvector == minU)))
        upper = int(np.min(np.where(knotvector == maxU)))
        if u == minU:
            return lower
        if u == maxU:
            return upper
        mid = (lower + upper) // 2
        while True:
            if u < knotvector[mid]:
                upper = mid
            elif knotvector[mid + 1] <= u:
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
            ValidationKnotVector.all(copylist)
            self.insert(span + 1, knot)
            degree, npts = self.compute_pn(list(self))
            ValidationKnotVector.PN(degree, npts)
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
            ValidationKnotVector.PN(degree, npts)
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
            return False
        try:
            obj = self.__class__(obj)
        except Exception as e:
            return False
        if self.npts != obj.npts:
            return False
        if self.degree != obj.degree:
            return False
        for i, v in enumerate(self):
            if v != obj[i]:
                return False
        return True

    def __ne__(self, __obj: object) -> bool:
        return not (self == __obj)


class GeneratorKnotVector:
    @staticmethod
    def bezier(degree: int) -> KnotVector:
        ValidationKnotVector.isIntegerNonNegative(degree)
        return GeneratorKnotVector.uniform(degree, degree + 1)

    @staticmethod
    def weight(degree: int, ws: Tuple[float]) -> KnotVector:
        ValidationKnotVector.isFloatArray1D(ws)
        ValidationKnotVector.isIntegerNonNegative(degree)
        knotvector = np.cumsum(ws)
        knotvector -= knotvector[0]
        knotvector /= knotvector[-1]
        knotvector *= ValidationKnotVector.maxU - ValidationKnotVector.minU
        knotvector += ValidationKnotVector.minU
        knotvector = degree * [0] + list(knotvector) + degree * [1]
        return KnotVector(knotvector)

    @staticmethod
    def uniform(degree: int, npts: int) -> KnotVector:
        ValidationKnotVector.PN(degree, npts)
        ws = np.ones(npts - degree + 1)
        return GeneratorKnotVector.weight(degree, ws)

    @staticmethod
    def random(degree: int, npts: int) -> KnotVector:
        ValidationKnotVector.PN(degree, npts)
        ws = np.random.rand(npts - degree + 1)
        return GeneratorKnotVector.weight(degree, ws)
