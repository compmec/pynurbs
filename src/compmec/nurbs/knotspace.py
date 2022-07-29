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

def getU_uniform(n, p):
    if n < p:
        n = p
    U = np.linspace(0, 1, n - p + 1)
    U = np.concatenate((np.zeros(p), U))
    U = np.concatenate((U, np.ones(p)))
    return KnotVector(U)

def getU_random(n, p):
    if n < p:
        n = p
    hs = np.random.random(n - p)
    hs *= 1.2
    hs[hs> 1] = 0
    U = np.cumsum(hs)
    if U[0] == 0:
        U += 0.1*np.random.random(1)
    
    U /= U[-1]*(1+0.4*np.random.random(1))
    
    U = np.concatenate((np.zeros(p+1), U))
    U = np.concatenate((U, np.ones(p+1)))
    return KnotVector(U)

class KnotVector(list):

    def __new__(cls, U: Iterable[float]):
        VerifyKnotVector.all(U)
        return super().__new__(cls, U)

    def __init__(self, U: Iterable[float]):
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
        m = len(self) - 1
        for i in range(m):
            if self[i] != minU:
                break
        self.__p = i-1
        self.__n = m - self.p

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


    def insert_knot(self, knot: float, times: int = 1):
        k = self.spot(knot)
        for i in range(times):
            self.insert(k+1, knot)
        