import numpy as np
from typing import Iterable
from compmec.nurbs.spaceu import VectorU 

def N(i: int, j: int, k: int, u: float, U: VectorU) -> float:
    """
    Returns the value of N_{ij}(u) in the interval [u_{k}, u_{k+1}]
    Remember that N_{i, j}(u) = 0   if  ( u not in [U[i], U[i+j+1]] )
    """
    
    n = U.n
    
    if k < i:
        return 0
    if j == 0:
        if i == k:
            return 1
        if i + 1 == n and k == n:
            return 1
        return 0
    if i + j < k:
        return 0

    if i+j >= len(U):
        factor1 = 0
    elif U[i] == U[i+j]:
        factor1 = 0
    else:
        factor1 = (u-U[i])/(U[i+j]-U[i])

    if i+j+1 >= len(U):
        factor2 = 0
    elif U[i+j+1] == U[i+1]:
        factor2 = 0
    else:
        factor2 = (U[i+j+1]-u)/(U[i+j+1]-U[i+1])

    result = factor1 * N(i, j-1, k, u, U) + factor2 * N(i+1, j-1, k, u, U)
    return result

class BaseFunction(object):

    def __init__(self, U: Iterable[float]):
        """
        We have that the object can be called like
        N = SplineBaseFunction(U)

        And so, to get the value of N_{i, j}(u) can be express like
        N[i, j](u)
        """
        self.U = VectorU(U)
        n, p = compute_np(self.U)
        self.__p = int(p)
        self.__n = int(n)
        # self.__get_divisors()

    @property
    def p(self):
        return self.__p

    @property
    def n(self):
        return self.__n

    def __transform_index(self, tup):
        if isinstance(tup, tuple):
            if len(tup) > 2:
                raise IndexError("The dimension of N is maximum 2")
            i, j = tup
        else:
            i = tup
            j = self.p

        if not isinstance(j, int):
            raise TypeError("The second value must be an integer, not %s" % type(j))
        elif j < 0:
            raise ValueError("The second value (%d) must be >= 0" % j)
        elif self.p < j:
            raise ValueError("The second value (%d) must be <= p = %d" % (j, self.p))

        if isinstance(i, int):
            if i > self.n:
                raise ValueError("The frist value must be <= n = %d" % self.n)
        return i, j

    def eval(self, u: np.ndarray) -> np.ndarray:
        return self.evalfunction()(u)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.eval(u)

class SplineBaseFunction(BaseFunction):

    def __doc__(self):
        """
        This function is recursively determined like

        N_{i, 0}(u) = { 1   if  U[i] <= u < U[i+1]
                      { 0   else

                          u - U[i]
        N_{i, j}(u) = --------------- * N_{i, j-1}(u)
                       U[i+j] - U[i]
                            U[i+j+1] - u
                      + ------------------- * N_{i+1, j-1}(u)
                         U[i+j+1] - U[i+1]

        As consequence, we have that

        N_{i, j}(u) = 0   if  ( u not in [U[i], U[i+j+1]] )

        """


    

    

    def __getitem__(self, tup):
        i, j = self.__transform_index(tup)
        newobject = SplineEvaluationFunction(self.U, i, j)
        return newobject

    def evalfunction(self):
        return SplineEvaluationFunction(self.U, slice(None,None,None), self.p)

    


class SplineEvaluationFunction(SplineBaseFunction):

    # def __doc__(self):
    #     pass

    def __init__(self, U: np.ndarray, i, j:int):
        super().__init__(U)
        # self.U = np.copy(U)
        if isinstance(i, slice):
            start = 0 if i.start is None else i.start
            stop = self.n if i.stop is None else i.stop
            step = 1 if i.step is None else i.step
            self.i = range(start, stop, step)
        else:
            self.i = i
        self.j = j

    

    def __validate_evaluation_u(self, u: np.ndarray):
        minU = np.min(self.U)
        maxU = np.max(self.U)
        if not np.all((minU <= u) * (u <= maxU)):
            raise Exception("u must be inside the interval [", minU, ", ", maxU, "]")


    def __find_spot_onevalue(self, u: float) -> int:
        minU = np.min(self.U)
        maxU = np.max(self.U)
        lower = int(np.max(np.where(self.U == minU)))
        upper = int(np.min(np.where(self.U == maxU)))
        if u == minU:
            return lower
        if u == maxU:
            return upper
        mid = (lower + upper) // 2
        while True:
            if u < self.U[mid]:
                upper = mid
            elif self.U[mid + 1] <= u:
                lower = mid
            else:
                return mid
            mid = (lower + upper) // 2

    def __find_spot_vector(self, u: np.ndarray) -> np.ndarray:
        """
        find the spots of an vector. It's not very efficient, but ok for the moment
        """
        spots = np.zeros(u.shape, dtype="int64")
        for i, ui in enumerate(u):
            spots[i] = self.__find_spot_onevalue(ui)
        return spots



    def spot(self, u: np.ndarray) -> np.ndarray:
        if isinstance(u, np.ndarray):
            return self.__find_spot_vector(u)
        else:
            return np.array(self.__find_spot_onevalue(u))
            
    def compute_scalar(self, u: float) -> float:
        k = self.spot(u)
        return N(self.i, self.j, k, u, self.U)

    def compute_vectori(self, u: float) -> np.ndarray:
        """
        In this function, u is a scalar, while i is a vector
        """
        r = np.zeros(self.n)
        k = self.spot(u)
        for i in range(self.n):
            r[i] = N(i, self.j, k, u, self.U)
        return r

    def compute_vectoru(self, u: np.ndarray) -> np.ndarray:
        """
        In this function, u is a vector, while i is a integer
        """
        u = np.copy(u)
        r = np.zeros(u.shape)
        for w, uw in enumerate(u):
            k = self.spot(uw)
            r[w] = N(self.i, self.j, k, uw, self.U)
        return r

    def compute_matrix(self, u: np.ndarray) -> np.ndarray:
        """
        In this function, u is a vector, while i is a range
        """
        r = np.zeros((self.n, len(u)))
        for w, uw in enumerate(u):
            k = self.spot(uw)
            for ind in self.i:
                r[ind, w] = N(ind, self.j, k, uw, self.U)
        return r

    def compute_all(self, u: np.ndarray) -> np.ndarray:
        r = np.zeros((self.n, len(u)))
        for w, uw in enumerate(u):
            r[:, w] += self.compute_vetori(uw)
        return r

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """

        """
        self.__validate_evaluation_u(u)
        if np.array(u).ndim > 1:
            raise ValueError("For the moment we can only evaluate scalars or 1D array")
        
        if isinstance(self.i, int):
            if np.array(u).ndim == 1:
                return self.compute_vectoru(u)
            else:
                return self.compute_scalar(u)
        else:
            if np.array(u).ndim == 1:
                if isinstance(self.i, range):
                    return self.compute_matrix(u)
                else:
                    return self.compute_all(u)
            else:
                return self.compute_vectori(u)
        raise ValueError("Cannot compute :(")