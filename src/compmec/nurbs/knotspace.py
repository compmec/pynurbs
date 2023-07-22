from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from compmec.nurbs import heavy
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
        if degree < 0:
            raise ValueError("Degree must be >= 0")
        if not (npts > degree):
            raise ValueError("Must have npts > degree")

    @staticmethod
    def all(knotvector: Tuple[float]) -> None:
        ValidationKnotVector.array1D(knotvector)
        knotvector = np.array(knotvector, dtype="float64")
        knotvector = tuple(knotvector)
        if not heavy.KnotVector.is_valid_vector(knotvector):
            error_msg = f"KnotVector is invalid: {knotvector}"
            raise ValueError(error_msg)


class KnotVector(Intface_KnotVector):
    def __init__(self, knotvector: Tuple[float]):
        if not isinstance(knotvector, self.__class__):
            ValidationKnotVector.all(knotvector)
        self.__update_vector(knotvector)

    def __update_vector(self, newvector: Tuple[float]):
        """
        Unprotected private function to set newvector
        """
        self.__degree = None
        self.__npts = None
        self.__internal_vector = tuple(newvector)
        return self

    def __str__(self) -> str:
        return str(self.__internal_vector)

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self) -> float:
        for knot in self.__internal_vector:
            yield knot

    def __getitem__(self, index):
        return self.__internal_vector[index]

    def __len__(self) -> int:
        return len(self.__internal_vector)

    def __iadd__(self, other: Union[float, Tuple[float]]):
        try:
            return self.shift(other)
        except TypeError:
            return self.__insert_knots(tuple(other))

    def __isub__(self, other: Union[float, Tuple[float]]):
        try:
            return self.shift(-other)
        except TypeError:
            return self.__remove_knots(tuple(other))

    def __imul__(self, other: float):
        return self.scale(other)

    def __itruediv__(self, other: float):
        return self.scale(1 / other)

    def __ior__(self, other: KnotVector) -> KnotVector:
        newvector = heavy.KnotVector.unite_vectors(tuple(self), tuple(other))
        if not heavy.KnotVector.is_valid_vector(newvector):
            raise ValueError("Cannot use __or__ in this case")
        return self.__update_vector(newvector)

    def __iand__(self, other: KnotVector) -> KnotVector:
        newvector = heavy.KnotVector.intersect_vectors(tuple(self), tuple(other))
        if not heavy.KnotVector.is_valid_vector(newvector):
            raise ValueError("Cannot use __and__ in this case")
        return self.__update_vector(newvector)

    def __add__(self, other: Union[float, Tuple[float]]):
        return self.deepcopy().__iadd__(other)

    def __sub__(self, other: Union[float, Tuple[float]]):
        return self.deepcopy().__isub__(other)

    def __mul__(self, other: float):
        return self.deepcopy().__imul__(other)

    def __rmul__(self, other: float):
        return self.deepcopy().__imul__(other)

    def __truediv__(self, other: float):
        return self.deepcopy().__itruediv__(other)

    def __or__(self, other: float):
        return self.deepcopy().__ior__(other)

    def __and__(self, other: float):
        return self.deepcopy().__iand__(other)

    def __eq__(self, other: object):
        if not isinstance(other, self.__class__):
            # Will try to cenvert other
            try:
                other = self.__class__(other)
            except ValueError:
                return False
        if self.npts != other.npts:
            return False
        if self.degree != other.degree:
            return False
        for i, v in enumerate(self):
            if v != other[i]:
                return False
        return True

    def __ne__(self, other: object) -> bool:
        return not self == other

    @property
    def degree(self) -> int:
        if self.__degree is None:
            self.__degree = heavy.KnotVector.find_degree(self)
        return int(self.__degree)

    @property
    def npts(self) -> int:
        if self.__npts is None:
            self.__npts = heavy.KnotVector.find_npts(self)
        return int(self.__npts)

    @property
    def knots(self) -> Tuple[float]:
        return heavy.KnotVector.find_knots(tuple(self))

    @property
    def limits(self) -> Tuple[float]:
        return (self[0], self[-1])

    @degree.setter
    def degree(self, value: int):
        value = int(value)
        knots = self.knots
        diff = value - self.degree
        if diff < 0:  # Decrease degree
            self -= (-diff) * knots
        if 0 < diff:  # Increase degree
            self += diff * knots

    def __insert_knots(self, knots: Tuple[float]):
        newvector = heavy.KnotVector.insert_knots(tuple(self), knots)
        if not heavy.KnotVector.is_valid_vector(newvector):
            error_msg = f"Cannot insert knots {knots} in knotvector {self}"
            raise ValueError(error_msg)
        return self.__update_vector(newvector)

    def __remove_knots(self, knots: Tuple[float]):
        try:
            newvector = heavy.KnotVector.remove_knots(tuple(self), knots)
            if not heavy.KnotVector.is_valid_vector(newvector):
                raise ValueError
        except ValueError:
            error_msg = f"Cannot remove knots {knots} in knotvector {self}"
            raise ValueError(error_msg)
        return self.__update_vector(newvector)

    def deepcopy(self) -> KnotVector:
        """
        Returns a exact object, but with different ID
        """
        return self.__class__(self.__internal_vector)

    def shift(self, value: float):
        """
        Moves every knot by an amount `value`
        """
        value = float(value)
        newvector = tuple([knoti + value for knoti in self])
        return self.__update_vector(newvector)

    def scale(self, value: float) -> KnotVector:
        """
        Multiply every knot by amount `value`
        """
        value = float(value)
        newvector = tuple([knoti * value for knoti in self])
        return self.__update_vector(newvector)

    def normalize(self) -> KnotVector:
        """
        Shift and scale the vector to match the interval [0, 1]
        """
        self.shift(-self[0])
        self.scale(1 / self[-1])
        return self

    def span(self, nodes: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Finds the index position of
        """
        nodes = np.array(nodes, dtype="float64")
        if np.any(nodes < self[0]) or np.any(self[-1] < nodes):
            raise ValueError("Nodes outside interval knotvector")
        flatnodes = nodes.flatten()
        flatspans = np.empty(flatnodes.shape, dtype="int16")
        for i, node in enumerate(flatnodes):
            flatspans[i] = heavy.KnotVector.find_span(node, tuple(self))
        return flatspans.reshape(nodes.shape)

    def mult(self, nodes: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Counts how many times a node is inside the knotvector
        """
        nodes = np.array(nodes, dtype="float64")
        if np.any(nodes < self[0]) or np.any(self[-1] < nodes):
            raise ValueError("Nodes outside interval knotvector")
        flatnodes = nodes.flatten()
        flatspans = np.empty(flatnodes.shape, dtype="int16")
        for i, node in enumerate(flatnodes):
            flatspans[i] = heavy.KnotVector.find_mult(node, tuple(self))
        return flatspans.reshape(nodes.shape)


class GeneratorKnotVector:
    @staticmethod
    def bezier(degree: int) -> KnotVector:
        ValidationKnotVector.degree_npts(degree, degree + 1)
        return KnotVector((degree + 1) * [0] + (degree + 1) * [1])

    @staticmethod
    def uniform(degree: int, npts: int) -> KnotVector:
        ValidationKnotVector.degree_npts(degree, npts)
        weights = np.ones(npts - degree)
        knotvector = GeneratorKnotVector.weight(degree, weights)
        knotvector.normalize()
        return knotvector

    @staticmethod
    def random(degree: int, npts: int) -> KnotVector:
        ValidationKnotVector.degree_npts(degree, npts)
        weights = np.random.rand(npts - degree)
        knotvector = GeneratorKnotVector.weight(degree, weights)
        knotvector.shift(np.random.uniform(-1, 1))
        return knotvector

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
        listknots = list(np.cumsum(weights))
        knotvector = (degree + 1) * [0] + listknots + degree * [listknots[-1]]
        return KnotVector(knotvector)
