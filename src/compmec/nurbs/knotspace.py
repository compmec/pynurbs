from typing import Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Intface_KnotVector


class ValidationKnotVector(object):
    @staticmethod
    def array1D(knotvector: Tuple[float]):
        knotvector = np.array(knotvector, "float64")
        if knotvector.ndim != 1:
            error_msg = "Argument must be a 1D array.\n"
            error_msg += f"Received shape = {knotvector.shape}"
            raise ValueError(error_msg)

    @staticmethod
    def degree_npts(degree: int, npts: int):
        if not isinstance(degree, int):
            raise TypeError("Degree must be an integer")
        if not isinstance(npts, int):
            raise TypeError("Number of points must be integer")
        if degree <= 0:
            raise ValueError("Degree must be >= 1")
        if not (npts > degree):
            raise ValueError("Must have npts > degree")

    @staticmethod
    def all(knotvector: Tuple[float]) -> None:
        ValidationKnotVector.array1D(knotvector)
        knotvector = np.array(knotvector, dtype="float64")
        # Ordered
        sorted_array = np.copy(knotvector)
        sorted_array.sort()
        if np.any(knotvector != sorted_array):
            raise ValueError("Knotvector must be ordored")
        # Multiplicity
        set_knotvector = list(set(knotvector.tolist()))
        set_knotvector.sort()
        multknots = [np.sum(knotvector == u) for u in set_knotvector]
        if multknots[0] != multknots[-1]:
            raise ValueError("Extremities quantities are not equal")
        if len(multknots) == 2:  # No internal knot
            return
        if max(multknots[1:-1]) >= multknots[0]:
            raise ValueError("An internal knot has multiplicity too high")


class KnotVector(Intface_KnotVector, list):
    def __init__(self, knotvector: Tuple[float]):
        ValidationKnotVector.all(knotvector)
        knotvector = np.array(knotvector)
        degree, npts = self.compute_degree_npts(knotvector)
        self.__degree = degree
        self.__npts = npts
        super().__init__(knotvector)

    @property
    def degree(self) -> int:
        return int(self.__degree)

    @property
    def npts(self) -> int:
        return int(self.__npts)

    @property
    def knots(self) -> Tuple[float]:
        knts = list(set(self))
        knts.sort()
        return tuple(knts)

    @staticmethod
    def compute_degree_npts(knotvector: Tuple[float]):
        """
        We have that knotvector = [0, ..., 0, ?, ..., ?, 1, ..., 1]
        And that knotvector[degree] = 0, but knotvector[degree+1] != 0
        The same way, knotvector[npts] = 1, but knotvector[npts-1] != 0

        Using that, we know that
            len(knotvector) = m + 1 = npts + degree + 1
        That means that
            m = npts + degree
        """
        min_knots = min(knotvector)
        degree = np.sum(knotvector == min_knots) - 1
        npts = len(knotvector) - degree - 1
        return int(degree), int(npts)

    def __valid_knot(self, u: float):
        try:
            u = float(u)
        except Exception:
            raise TypeError
        minU = np.min(self)
        maxU = np.max(self)
        if u < minU:
            raise ValueError(f"Received u = {u} < minU = {minU}")
        if maxU < u:
            raise ValueError(f"Received u = {u} > maxU = {maxU}")

    def __valid_insert_knot(self, u: float, times: float):
        if times <= 0:
            raise ValueError
        mult = self.mult_onevalue(u)
        if times > self.degree - mult:
            raise ValueError

    def __valid_remove_knot(self, u: float, times: float):
        if times <= 0:
            raise ValueError
        mult = self.mult_onevalue(u)
        if times > mult:
            raise ValueError

    def span_onevalue(self, u: float) -> int:
        self.__valid_knot(u)
        vector = np.array(self)
        lower = int(np.max(np.where(vector == self[0])))
        upper = int(np.min(np.where(vector == self[-1])))
        if u == self[0]:
            return lower
        if u == self[-1]:
            return upper
        mid = (lower + upper) // 2
        while True:
            if u < vector[mid]:
                upper = mid
            elif vector[mid + 1] <= u:
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
        self.__valid_knot(u)
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

    def __knot_insert(self, knot: float, times: int):
        if times == 1:
            span = self.span_onevalue(knot)
            copylist = list(self)
            copylist.insert(span + 1, knot)
            ValidationKnotVector.all(copylist)
            self.insert(span + 1, knot)
            degree, npts = self.compute_degree_npts(list(self))
            ValidationKnotVector.degree_npts(degree, npts)
            self.__degree = degree
            self.__npts = npts
            return
        for i in range(times):
            self.__knot_insert(knot, 1)

    def __knot_remove(self, knot: float, times: int):
        if times == 1:
            span = self.span_onevalue(knot)
            self.pop(span)
            degree, npts = self.compute_degree_npts(list(self))
            ValidationKnotVector.degree_npts(degree, npts)
            self.__degree = degree
            self.__npts = npts
            return
        for i in range(times):
            self.__knot_remove(knot, 1)

    def knot_insert(self, knot: float, times: Optional[int] = 1):
        self.__valid_knot(knot)
        self.__valid_insert_knot(knot, times)
        self.__knot_insert(knot, times)

    def knot_remove(self, knot: float, times: Optional[int] = 1):
        self.__valid_knot(knot)
        self.__valid_remove_knot(knot, times)
        self.__knot_remove(knot, times)

    def __eq__(self, obj: object):
        if not isinstance(obj, self.__class__):
            # Will try to cenvert obj
            try:
                obj = self.__class__(obj)
            except ValueError:
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
        npts = degree + 1
        ValidationKnotVector.degree_npts(degree, npts)
        return KnotVector((degree + 1) * [0] + (degree + 1) * [1])

    @staticmethod
    def uniform(degree: int, npts: int) -> KnotVector:
        ValidationKnotVector.degree_npts(degree, npts)
        weights = np.ones(npts - degree)
        return GeneratorKnotVector.weight(degree, weights)

    @staticmethod
    def random(degree: int, npts: int) -> KnotVector:
        ValidationKnotVector.degree_npts(degree, npts)
        weights = np.random.rand(npts - degree)
        return GeneratorKnotVector.weight(degree, weights)

    @staticmethod
    def weight(degree: int, weights: Tuple[float]) -> KnotVector:
        """
        Creates a KnotVector with degree ```degree``` and spaced points
        depending on the ```weights``` vectors.
        The number of segments are equal to lenght of weights

        degree = 1 and weights = [1] -> [0, 0, 1, 1]
        degree = 1 and weights = [1, 1] -> [0, 0, 0.5, 1, 1]
        degree = 1 and weights = [1, 1, 2] -> [0, 0.25, 0.5, 1, 1]
        """
        ValidationKnotVector.array1D(weights)
        npts = len(weights) + degree
        ValidationKnotVector.degree_npts(degree, npts)
        cumsum = np.cumsum(weights)
        listknots = [knot / cumsum[-1] for knot in cumsum]
        knotvector = (degree + 1) * [0] + listknots + degree * [1]
        return KnotVector(knotvector)
