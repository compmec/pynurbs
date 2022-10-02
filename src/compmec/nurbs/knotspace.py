import numpy as np
from typing import Iterable, Union

class VerifyKnotVector(object):

    __minU = 0
    __maxU = 1

    @staticmethod
    def isIterable(U: Iterable) -> None:
        try:
            iter(U)
        except TypeError as e:
            raise TypeError(f"The given U ({type(U)}) is not iterable")
        
    @staticmethod
    def eachElementIsFloat(U: Iterable[float]) -> None:
        try:
            for u in U:
                float(u)
        except Exception as e:
            raise TypeError(f"Each element inside U must be a float, cannot transform {type(u)} on float")
        
    @staticmethod
    def isOrdenedVector(U: Iterable[float]) -> None:
        n = len(U)
        for i in range(n-1):
            if U[i] > U[i+1]:
                raise ValueError("The given U must be ordened")
            
    @staticmethod
    def Limits(U: Iterable[float]) -> None:
        if VerifyKnotVector.__minU is not None:
            VerifyKnotVector.InferiorLimit(U)
        if VerifyKnotVector.__maxU is not None:
            VerifyKnotVector.SuperiorLimit(U)

    @staticmethod
    def InferiorLimit(U: Iterable[float]) -> None:
        for u in U:
            if u < VerifyKnotVector.__minU:
                raise ValueError(f"All the values in U must be bigger than {VerifyKnotVector.__minU}")

    @staticmethod
    def SuperiorLimit(U: Iterable[float]) -> None:
        for u in U:
            if u > VerifyKnotVector.__maxU:
                raise ValueError(f"All the values in U must be less than {VerifyKnotVector.__maxU}")
        
    @staticmethod
    def SameQuantityBoundary(U: Iterable[float]) -> None:
        U = np.array(U)
        minU = np.min(U)
        maxU = np.max(U)
        if np.sum(U == minU) != np.sum(U == maxU):
            raise ValueError("U must contain the same quantity of 0 and 1. U = ", U)

    @staticmethod
    def all(U: Iterable[float]) -> None:
        VerifyKnotVector.isIterable(U)
        VerifyKnotVector.eachElementIsFloat(U)
        VerifyKnotVector.Limits(U)
        VerifyKnotVector.SameQuantityBoundary(U)

class GeneratorKnotVector:

    @staticmethod
    def bezier(p: int):
        return KnotVector((p+1)*[0] + (p+1)*[1])

    @staticmethod
    def uniform(p: int, n: int):
        if n <= p:
            raise ValueError("Must n >= p. Received n=%d, p=%d"%(n, p))
        U = np.linspace(0, 1, n - p + 1)
        U = p*[0] + list(U) + p*[1]
        return KnotVector(U)

    def random(p: int, n: int):
        if n <= p:
            raise ValueError("Must n >= p. Received n=%d, p=%d"%(n, p))
        hs = np.random.random(n - p + 1)
        U = np.cumsum(hs)
        U -= U[0]
        U /= U[-1]
        U = p*[0] + list(U) + p*[1]
        return KnotVector(U) 


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
        value = int(value)
        if value < 0:
            raise ValueError(f"Cannot set p to {value}")
        self.__p = value

    @n.setter
    def n(self, value: int):
        value = int(value)
        if value < 0:
            raise ValueError(f"Cannot set n to {value}")
        if value <= self.p:
            raise ValueError(f"The value of n ({n}) must be greater than p = {p}")
        self.__n = value

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
        m = len(self) - 1
        for i in range(m+1):
            if self[i] != minU:
                break
        self.p = i-1
        self.n = m - self.p

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
            raise TypeError(f"Cannot compare {type(__obj)} with a {self.__class__} instance")
        try:
            __obj = self.__class__(__obj)
        except Exception as e:
            raise ValueError(f"No sucess trying to convert {type(__obj)} into {self.__class__}. Cause {str(e)}")
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