from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from compmec.nurbs import heavy
from compmec.nurbs.__classes__ import Intface_BaseFunction, Intface_Evaluator
from compmec.nurbs.knotspace import KnotVector


class BaseFunction(Intface_BaseFunction):
    def __init__(self, knotvector: KnotVector):
        self.knotvector = knotvector
        self.weights = None

    def __eq__(self, other: Intface_BaseFunction) -> bool:
        if not isinstance(other, Intface_BaseFunction):
            return False
        if self.knotvector != other.knotvector:
            return False
        weightleft = self.weights
        weightrigh = other.weights
        weightleft = np.ones(self.npts) if self.weights is None else self.weights
        weightrigh = np.ones(self.npts) if weightrigh is None else weightrigh
        return np.all(weightleft == weightrigh)

    def __ne__(self, other: Intface_BaseFunction) -> bool:
        return not self.__eq__(other)

    def __call__(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.eval(nodes)

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

    @degree.setter
    def degree(self, value: int):
        value = int(value)
        self.knotvector.degree = value

    @knotvector.setter
    def knotvector(self, value: KnotVector):
        self.__knotvector = KnotVector(value)

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

    def deepcopy(self) -> BaseFunction:
        newfunc = self.__class__(self.knotvector)
        newfunc.weights = self.weights
        return newfunc


class FunctionEvaluator(Intface_Evaluator):
    def __init__(self, func: BaseFunction, i: Union[int, slice], j: int):
        self.__knotvector = func.knotvector
        self.__weights = func.weights
        self.__first_index = i
        self.__second_index = j
        self.__matrix = heavy.speval_matrix(tuple(self.__knotvector), j)
        self.__knots = heavy.KnotVector.find_knots(tuple(self.__knotvector))
        self.__spans = np.zeros(len(self.__knots), dtype="int16")
        for k, knot in enumerate(self.__knots):
            self.__spans[k] = heavy.KnotVector.find_span(knot, tuple(self.__knotvector))
        self.__spans = tuple(self.__spans)

    def compute_one_value(self, i: int, node: float, span: int) -> float:
        j = self.__second_index
        U = tuple(self.__knotvector)
        if self.__weights is None:
            return heavy.N(i, j, span, node, U)
        return heavy.R(i, j, span, node, U, self.__weights)

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

    def __eval(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Private and unprotected method of eval
        """
        nodes = np.array(nodes, dtype="float64")
        flat_nodes = nodes.flatten()
        flat_spans = self.__knotvector.span(flat_nodes)
        flat_spans = np.array(flat_spans, dtype="int16")
        newshape = [self.__knotvector.npts] + list(nodes.shape)
        result = self.compute_matrix(flat_nodes, flat_spans)
        return result.reshape(newshape)

    def eval(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        If i is integer, u is float -> float
        If i is integer, u is np.ndarray, ndim = k -> np.ndarray, ndim = k
        If i is slice, u is float -> 1D np.ndarray
        if i is slice, u is np.ndarray, ndim = k -> np.ndarray, ndim = k+1
        """
        return self.__eval(nodes)[self.__first_index]

    def __call__(self, nodes: np.ndarray) -> np.ndarray:
        return self.eval(nodes)


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

    def __getitem__(self, index) -> FunctionEvaluator:
        if isinstance(index, tuple):
            if len(index) > 2:
                raise IndexError
            i, j = index
        else:
            i, j = index, self.degree
        self.__valid_first_index(i)
        self.__valid_second_index(j)
        return FunctionEvaluator(self, i, j)

    def eval(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        evaluator = self[:, self.degree]
        return evaluator(nodes)


class Function(IndexableFunction):
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
