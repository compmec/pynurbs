import abc
from typing import Any, Optional, Tuple, Union

import numpy as np

from compmec.nurbs.__classes__ import Intface_BaseFunction, Intface_Evaluator
from compmec.nurbs.algorithms import N, R
from compmec.nurbs.knotspace import KnotVector


class BaseFunction(Intface_BaseFunction):
    def __init__(self, knotvector: KnotVector):
        self.__knotvector = KnotVector(knotvector)
        self.__weights = None

    def __eq__(self, other: Intface_BaseFunction) -> bool:
        if not isinstance(other, Intface_BaseFunction):
            return False
        if self.degree != other.degree:
            return False
        if self.npts != other.npts:
            return False
        if self.knotvector != other.knotvector:
            return False
        return np.all(self.weights == other.weights)

    def __ne__(self, other: Intface_BaseFunction) -> bool:
        return not self.__eq__(other)

    @property
    def knotvector(self) -> KnotVector:
        return self.__knotvector

    @property
    def degree(self) -> int:
        return self.knotvector.degree

    @property
    def npts(self) -> int:
        return self.knotvector.npts

    @property
    def knots(self) -> Tuple[float]:
        return self.knotvector.knots

    @property
    def weights(self) -> Union[Tuple[float], None]:
        return self.__weights

    @weights.setter
    def weights(self, value: Tuple[float]):
        if value is None:
            self.__weights = None
            return
        value = np.array(value, dtype="float64")
        if not np.all(value > 0):
            error_msg = "All weights must be positive!"
            raise ValueError(error_msg)
        if value.shape != (self.npts,):
            error_msg = f"Weights shape invalid! {value.shape} != ({self.npts})"
            raise ValueError(error_msg)
        self.__weights = value

    def knot_insert(self, knot: float, times: Optional[int] = 1):
        self.knotvector.knot_insert(knot, times)

    def knot_remove(self, knot: float, times: Optional[int] = 1):
        self.knotvector.knot_remove(knot, times)

    def degree_increase(self, times: Optional[int] = 1):
        pass

    def degree_decrease(self, times: Optional[int] = 1):
        pass

    @abc.abstractmethod
    def evalf(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def __call__(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.evalf(nodes)


class FunctionEvaluator(Intface_Evaluator):
    def __init__(self, func: BaseFunction, i: Union[int, slice], j: int):
        self.__knotvector = func.knotvector
        self.__weights = func.weights
        self.__first_index = i
        self.__second_index = j

    def compute_one_value(self, i: int, node: float, span: int) -> float:
        j = self.__second_index
        U = tuple(self.__knotvector)
        if self.__weights is None:
            return N(i, j, span, node, U)
        return R(i, j, span, node, U, self.__weights)

    def compute_vector(self, node: float, span: int) -> np.ndarray:
        """
        Given a 'u' float, it returns the vector with all BasicFunctions:
        compute_vector(u, span) = [F_{0j}(u), F_{1j}(u), ..., F_{npts-1,j}(u)]
        """
        npts = self.__knotvector.npts
        result = np.zeros(npts, dtype="float64")
        # for i in range(span, span+self.second_index):
        for i in range(npts):
            result[i] = self.compute_one_value(i, node, span)
        return result

    def compute_matrix(
        self, nodes: Tuple[float], spans: Union[int, np.ndarray]
    ) -> Tuple[Tuple[float]]:
        """
        Receives an 1D array of nodes, and returns a 2D array.
        nodes.shape = (len(nodes), )
        result.shape = (npts, len(nodes))
        """
        nodes = np.array(nodes, dtype="float64")
        newshape = [self.__knotvector.npts] + list(nodes.shape)
        result = np.zeros(newshape, dtype="float64")
        for i, (nodei, spani) in enumerate(zip(nodes, spans)):
            result[:, i] = self.compute_vector(nodei, spani)
        return result

    def __evalf(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        If i is integer, u is float -> float
        If i is integer, u is np.ndarray, ndim = k -> np.ndarray, ndim = k
        If i is slice, u is float -> 1D np.ndarray
        if i is slice, u is np.ndarray, ndim = k -> np.ndarray, ndim = k+1
        """
        nodes = np.array(nodes, dtype="float64")
        flat_nodes = nodes.flatten()
        flat_spans = self.__knotvector.span(flat_nodes)
        flat_spans = np.array(flat_spans, dtype="int16")
        newshape = [self.__knotvector.npts] + list(nodes.shape)
        result = self.compute_matrix(flat_nodes, flat_spans)
        return result.reshape(newshape)

    def evalf(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        If i is integer, u is float -> float
        If i is integer, u is np.ndarray, ndim = k -> np.ndarray, ndim = k
        If i is slice, u is float -> 1D np.ndarray
        if i is slice, u is np.ndarray, ndim = k -> np.ndarray, ndim = k+1
        """
        return self.__evalf(nodes)[self.__first_index]

    def __call__(self, nodes: np.ndarray) -> np.ndarray:
        return self.evalf(nodes)


class IndexableFunction(BaseFunction):
    def __init__(self, knotvector: KnotVector):
        super().__init__(knotvector)

    def __valid_first_index(self, index: Union[int, slice]):
        if not isinstance(index, (int, slice)):
            raise TypeError
        if isinstance(index, int):
            npts = self.npts
            if not (-npts <= index < npts):
                raise IndexError

    def __valid_second_index(self, index: int):
        if not isinstance(index, int):
            raise TypeError
        if not (0 <= index <= self.degree):
            error_msg = f"Second index (={index}) "
            error_msg += f"must be in [0, {self.degree}]"
            raise IndexError(error_msg)

    def __getitem__(self, tup: Any):
        if isinstance(tup, tuple):
            if len(tup) > 2:
                raise IndexError
            i, j = tup
        else:
            i, j = tup, self.degree
        self.__valid_first_index(i)
        self.__valid_second_index(j)
        return FunctionEvaluator(self, i, j)

    def evalf(self, nodes: np.ndarray) -> np.ndarray:
        evaluator = self[:, self.degree]
        return evaluator(nodes)


class DerivableFunction(IndexableFunction):
    def __init__(self, knotvector: KnotVector):
        super().__init__(knotvector)
        self.__number_derivated = 0
        self.__matrix = np.eye(self.npts, dtype="float64")

    @property
    def degree(self) -> int:
        return self.knotvector.degree - self.__number_derivated

    @property
    def matrix(self) -> np.ndarray:
        return np.copy(self.__matrix)

    def derivate(self, times: Optional[int] = 1):
        if times != 1:
            for i in range(times):
                self.derivate()
            return
        avals = np.zeros(self.npts)
        for i in range(self.npts):
            diff = (
                self.knotvector[i + self.degree] - self.knotvector[i]
            )  # Maybe it's wrong
            if diff != 0:
                avals[i] = self.degree / diff
        newA = np.diag(avals)
        for i in range(self.npts - 1):
            newA[i, i + 1] = -avals[i + 1]
        self.__matrix = self.__matrix @ newA
        self.__number_derivated += 1

    def evalf(self, nodes: np.ndarray) -> np.ndarray:
        evaluator = self[:, self.degree]
        return self.matrix @ evaluator(nodes)


class Function(DerivableFunction):
    def __doc__(self):
        """
        Spline and rational base function
        """

    def __repr__(self) -> str:
        if self.npts == self.degree + 1:
            return f"Bezier function of degree {self.degree}"
        elif self.weights is None:
            msg = "Spline"
        else:
            msg = "Rational"
        msg += f" function of degree {self.degree} "
        msg += f"and {self.npts} points"
        return msg
