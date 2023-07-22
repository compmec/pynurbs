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
        value = np.array(value, dtype="object")
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
        self.__matrix = heavy.BasisFunction.speval_matrix(tuple(self.__knotvector), j)
        self.__knots = heavy.KnotVector.find_knots(tuple(self.__knotvector))
        self.__spans = np.zeros(len(self.__knots), dtype="int16")
        for k, knot in enumerate(self.__knots):
            self.__spans[k] = heavy.KnotVector.find_span(knot, tuple(self.__knotvector))
        self.__spans = tuple(self.__spans)

    def __compute_vector_spline(self, node: float, span: int) -> np.ndarray:
        """
        Given a 'u' float, it returns the vector with all Spline Basis Functions:
        compute_vector(u, span) = [N_{0j}(u), N_{1j}(u), ..., N_{npts-1,j}(u)]
        """
        npts = self.__knotvector.npts
        result = [0 * node] * npts
        z = self.__spans.index(span)
        denom = self.__knots[z + 1] - self.__knots[z]
        shifnode = (node - self.__knots[z]) / denom
        for y in range(self.__second_index + 1):
            i = y + span - self.__second_index
            for k in range(self.__second_index, -1, -1):
                result[i] *= shifnode
                result[i] += self.__matrix[z][y][k]
        return result

    def __compute_vector(self, node: float, span: int) -> np.ndarray:
        """
        Given a 'u' float, it returns the vector with all BasicFunctions:
        compute_vector(u, span) = [F_{0j}(u), F_{1j}(u), ..., F_{npts-1,j}(u)]
        """
        result = self.__compute_vector_spline(node, span)
        if self.__weights is None:
            return result
        return self.__weights * result / np.inner(self.__weights, result)

    def __compute_matrix(
        self, nodes: Tuple[float], spans: Tuple[int]
    ) -> Tuple[Tuple[float]]:
        """
        Receives an 1D array of nodes, and returns a 2D array.
        nodes.shape = (len(nodes), )
        result.shape = (npts, len(nodes))
        """
        nodes = tuple(nodes)
        npts = self.__knotvector.npts
        matrix = np.empty((npts, len(nodes)), dtype="object")
        for j, (nodej, spanj) in enumerate(zip(nodes, spans)):
            values = self.__compute_vector(nodej, spanj)
            for i in range(npts):
                matrix[i][j] = values[i]
        matrix = matrix.tolist()
        for i, line in enumerate(matrix):
            matrix[i] = tuple(line)
        return tuple(matrix)

    def __eval(self, nodes: Tuple[float]) -> Tuple[Tuple[float]]:
        """
        Private and unprotected method of eval
        """
        nodes = tuple(nodes)
        spans = self.__knotvector.span(nodes)
        matrix = self.__compute_matrix(nodes, spans)
        return matrix

    def eval(
        self, nodes: Union[float, Tuple[float]]
    ) -> Union[float, Tuple[float], Tuple[Tuple[float]]]:
        """
        If i is integer, u is float -> float
        If i is integer, u is Tuple[float], ndim = k -> np.ndarray, ndim = k
        If i is slice, u is float -> Tuple[float]
        if i is slice, u is Tuple[float], ndim = k -> Tuple[Tuple[float]], ndim = k+1
        """
        singlenode = True
        try:
            iter(nodes)
            singlenode = False
        except TypeError:
            nodes = (nodes,)
        matrix = self.__eval(nodes)
        if singlenode:
            matrix = tuple([ri[0] for ri in matrix])
        result = matrix[self.__first_index]
        return result

    def __call__(
        self, nodes: Union[float, Tuple[float]]
    ) -> Union[float, Tuple[float], Tuple[Tuple[float]]]:
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
