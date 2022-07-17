import numpy as np
from typing import Iterable, Any, Optional, Union, Tuple, Type
from compmec.nurbs.spaceu import VectorU 

def N(i: int, j: int, k: int, u: float, U: VectorU) -> float:
    """
    Returns the value of N_{ij}(u) in the interval [u_{k}, u_{k+1}]
    Remember that N_{i, j}(u) = 0   if  ( u not in [U[i], U[i+j+1]] )
    """
    
    n, p = U.n, U.p
    
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

    if i+j > n+p:
        factor1 = 0
    elif U[i] == U[i+j]:
        factor1 = 0
    else:
        factor1 = (u-U[i])/(U[i+j]-U[i])

    if i+j+1 > n+p:
        factor2 = 0
    elif U[i+j+1] == U[i+1]:
        factor2 = 0
    else:
        factor2 = (U[i+j+1]-u)/(U[i+j+1]-U[i+1])

    result = factor1 * N(i, j-1, k, u, U) + factor2 * N(i+1, j-1, k, u, U)
    return result



class EvaluationFunction(object):

    def __init__(self, tup : Union[None, int, slice, Tuple]= None):
        if tup is None:
            self.i = None
            self.j = None
        elif isinstance(tup, tuple):
            if len(tup) > 2:
                raise IndexError("The dimension of N is maximum 2")
            self.i = tup[0]
            self.j = tup[1]
        else:
            self.i = tup
            self.j = None

    @property
    def i(self):
        return self._i
    
    @property
    def j(self):
        return self._j

    @i.setter
    def i(self, value: Union[None, int, slice]):
        if value is None:
            self._i = slice(0, self.n, 1)
            return
        if isinstance(value, slice):
            start = 0 if value.start is None else value.start
            stop = self.n if value.stop is None else value.stop
            step = 1 if value.step is None else value.step
            self._i = slice(start, stop, step)
            return
        try:
            value = int(value)
            if value > self.n:
                raise ValueError("The frist value must be <= n = %d" % self.n)
            self._i = value
            return
        except Exception as e:
            raise TypeError(f"Type {type(value)} is incorrect to set i")
        
    @j.setter
    def j(self, value: Union[None, int]):
        if value is None:
            self._j = self.p
            return
        try:
            value = int(value)
            if value < 0:
                raise ValueError("The second value (%d) must be >= 0" % value)
            elif self.p < value:
                raise ValueError("The second value (%d) must be <= p = %d" % (value, self.p))
            self._j = value
            return
        except Exception as e:
            raise TypeError(f"Type {type(value)} is incorrect to set j")



class BaseFunction(object):

    def __init__(self, U: Iterable[float]):
        """
        We have that the object can be called like
        N = SplineBaseFunction(U)

        And so, to get the value of N_{i, j}(u) can be express like
        N[i, j](u)
        """
        if isinstance(U, VectorU):
            self._U = U
        else:
            self._U = VectorU(U)

    @property
    def p(self) -> int:
        return self._U.p

    @property
    def n(self) -> int:
        return self._U.n

    @property
    def U(self) -> Tuple[float]:
        return tuple(self._U)

    @property
    def evalfunction(self) -> type[EvaluationFunction]:
        raise NotImplementedError("This function must be overwritten")

    def evaluationobject(self, tup: Union[None, int, slice]= None) -> EvaluationFunction:
        return self.evalfunction(self.U, tup)

    def eval(self, u: np.ndarray, tup = None) -> np.ndarray:
        function = self.evalfunction(self.U, tup)
        return function(u)

    def __getitem__(self, tup: slice) -> EvaluationFunction:
        return self.evaluationobject(tup)

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
    
    def __init__(self, U: Iterable[float]):
        super().__init__(U)

    @property
    def evalfunction(self) -> type[EvaluationFunction]:
        return SplineEvaluationFunction

class RationalBaseFunction(BaseFunction):
    def __init__(self, U: Iterable[float]):
        super().__init__(U)

    @property
    def evalfunction(self) -> type[EvaluationFunction]:
        return RationalEvaluationFunction


class SplineEvaluationFunction(SplineBaseFunction, EvaluationFunction):

    def __init__(self, U: Iterable[float], tup: Any):
        SplineBaseFunction.__init__(self, U)
        EvaluationFunction.__init__(self, tup)
        
    def __validate_evaluation_u(self, u: np.ndarray):
        U = self._U
        minU = np.min(U)
        maxU = np.max(U)
        if not np.all((minU <= u) * (u <= maxU)):
            raise Exception("u must be inside the interval [", minU, ", ", maxU, "]")

    
            
    def compute_scalar(self, u: float) -> float:
        k = self._U.spot(u)
        return N(self.i, self.j, k, u, self._U)

    def compute_vectori(self, u: float) -> np.ndarray:
        """
        In this function, u is a scalar, while i is a vector
        """
        r = np.zeros(self.n)
        k = self._U.spot(u)
        for i in range(self.n):
            r[i] = N(i, self.j, k, u, self._U)
        return r

    def compute_vectoru(self, u: np.ndarray) -> np.ndarray:
        """
        In this function, u is a vector, while i is a integer
        """
        u = np.copy(u)
        r = np.zeros(u.shape)
        for w, uw in enumerate(u):
            k = self.spot(uw)
            r[w] = N(self.i, self.j, k, uw, self._U)
        return r

    def compute_matrix(self, u: np.ndarray) -> np.ndarray:
        """
        In this function, u is a vector, while i is a range
        """
        r = np.zeros((self.n, len(u)))
        for w, uw in enumerate(u):
            k = self.spot(uw)
            for ind in self.i:
                r[ind, w] = N(ind, self.j, k, uw, self._U)
        return r

    def compute_all(self, u: np.ndarray) -> np.ndarray:
        r = np.zeros((self.n, len(u)))
        for w, uw in enumerate(u):
            r[:, w] += self.compute_vectori(uw)
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

class RationalEvaluationFunction(RationalBaseFunction, EvaluationFunction):
    def __init__(self, U: Iterable[float], tup: Any):
        super().__init__(U=U)
        super().__init__(tup=tup)