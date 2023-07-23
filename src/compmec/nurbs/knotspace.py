from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np

from compmec.nurbs import heavy
from compmec.nurbs.__classes__ import Intface_KnotVector


class KnotVector(Intface_KnotVector):
    def __init__(self, knotvector: Tuple[float]):
        if isinstance(knotvector, self.__class__):
            self.__safe_init(knotvector)
        elif not heavy.KnotVector.is_valid_vector(knotvector):
            msg = f"Invalid KnotVector of type {type(knotvector)} = {knotvector}"
            raise ValueError(msg)
        self.__safe_init(knotvector)

    def __safe_init(self, newvector: Tuple[float]):
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
        try:
            newvector = heavy.KnotVector.unite_vectors(tuple(self), tuple(other))
        except AssertionError:
            error_msg = f"Cannot use __or__ between {self} and {other}"
            raise ValueError(error_msg)
        return self.__safe_init(newvector)

    def __iand__(self, other: KnotVector) -> KnotVector:
        try:
            newvector = heavy.KnotVector.intersect_vectors(tuple(self), tuple(other))
        except AssertionError:
            error_msg = f"Cannot use __and__ between {self} and {other}"
            raise ValueError(error_msg)
        return self.__safe_init(newvector)

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
            self.__degree = heavy.KnotVector.find_degree(tuple(self))
        return int(self.__degree)

    @property
    def npts(self) -> int:
        if self.__npts is None:
            self.__npts = heavy.KnotVector.find_npts(tuple(self))
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

    def __insert_knots(self, nodes: Tuple[float]):
        nodes = tuple(nodes)
        for node in nodes:
            self.__validate_node(node)
        try:
            newvector = heavy.KnotVector.insert_knots(tuple(self), nodes)
        except AssertionError:
            error_msg = f"Cannot insert nodes {nodes} in knotvector {self}"
            raise ValueError(error_msg)
        return self.__safe_init(newvector)

    def __remove_knots(self, nodes: Tuple[float]):
        nodes = tuple(nodes)
        for node in nodes:
            self.__validate_node(node)
        try:
            newvector = heavy.KnotVector.remove_knots(tuple(self), nodes)
        except AssertionError:
            error_msg = f"Cannot remove nodes {nodes} in knotvector {self}"
            raise ValueError(error_msg)
        return self.__safe_init(newvector)

    def deepcopy(self) -> KnotVector:
        """
        Returns a exact object, but with different ID
        """
        knotvector = [deepcopy(knot) for knot in self]
        return self.__class__(knotvector)

    def shift(self, value: float):
        """
        Moves every knot by an amount `value`
        """
        float(value)  # Verify if it's a number
        newvector = tuple([knoti + value for knoti in self])
        return self.__safe_init(newvector)

    def scale(self, value: float) -> KnotVector:
        """
        Multiply every knot by amount `value`
        """
        float(value)  # Verify if it's a number
        newvector = tuple([knoti * value for knoti in self])
        return self.__safe_init(newvector)

    def normalize(self) -> KnotVector:
        """
        Shift and scale the vector to match the interval [0, 1]
        """
        self.shift(-self[0])
        self.scale(1 / self[-1])
        return self

    def __validate_node(self, node: float):
        """
        Private method of to verify if a node is
        - Is a number
        - inside the interval
        """
        if isinstance(node, (str, dict, tuple, list)):
            raise TypeError
        float(node)  # Verify if it's a number
        if node < self[0] or self[-1] < node:
            msg = f"{node} outside interval [{self[0]}, {self[-1]}]"
            raise ValueError(msg)

    def __span(self, nodes: Tuple[float]) -> Tuple[int]:
        """
        Private method of self.span, doesn't raise errors
        """
        spans = []
        vector = tuple(self)
        for node in nodes:
            newspan = heavy.KnotVector.find_span(node, vector)
            spans.append(newspan)
        return tuple(spans)

    def span(self, nodes: Union[float, Tuple[float]]) -> Union[int, Tuple[int]]:
        """
        Finds the index position of
        """
        onevalue = True
        try:
            iter(nodes)
            onevalue = False
        except TypeError:
            nodes = (nodes,)
        for node in nodes:
            self.__validate_node(node)  # may raise error
        spans = self.__span(nodes)
        return spans[0] if onevalue else spans

    def __mult(self, nodes: Tuple[float]) -> Tuple[int]:
        mults = []
        vector = tuple(self)
        for node in nodes:
            newmult = heavy.KnotVector.find_mult(node, vector)
            mults.append(newmult)
        return tuple(mults)

    def mult(self, nodes: Union[float, Tuple[float]]) -> Union[int, Tuple[int]]:
        """
        Counts how many times a node is inside the knotvector
        """
        onevalue = True
        try:
            iter(nodes)
            onevalue = False
        except TypeError:
            nodes = (nodes,)
        for node in nodes:
            self.__validate_node(node)  # may raise error
        mults = self.__mult(nodes)
        return mults[0] if onevalue else mults


class GeneratorKnotVector:
    @staticmethod
    def bezier(degree: int, cls: Optional[type] = None) -> KnotVector:
        """
        Returns a knotvector for a bezier curve.
        The parameter ```cls``` is good if you want your knotvector
        to be of a certain type
        """
        assert isinstance(degree, int)
        assert degree >= 0
        cls = cls if cls else int
        knotvector = (degree + 1) * [cls(0)] + (degree + 1) * [cls(1)]
        return KnotVector(knotvector)

    @staticmethod
    def uniform(degree: int, npts: int, cls: Optional[type] = None) -> KnotVector:
        """
        Creates a equally distributed knotvector between [0, 1]
        """
        assert isinstance(degree, int)
        assert isinstance(npts, int)
        assert degree >= 0
        assert npts > degree
        cls = cls if cls else int
        nintknots = npts - degree - 1
        knotvector = degree * [cls(0)]
        knotvector += [cls(i) for i in range(nintknots + 2)]
        knotvector += degree * [cls(nintknots + 1)]
        knotvector = KnotVector(knotvector)
        knotvector.normalize()
        return knotvector

    @staticmethod
    def random(degree: int, npts: int) -> KnotVector:
        assert isinstance(degree, int)
        assert isinstance(npts, int)
        assert degree >= 0
        assert npts > degree
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
        assert isinstance(degree, int)
        assert degree >= 0
        weights = tuple(weights)
        if len(weights) == 0:
            return GeneratorKnotVector.bezier(degree)
        listknots = list(np.cumsum(weights))
        knotvector = (degree + 1) * [0] + listknots + degree * [listknots[-1]]
        knotvector = KnotVector(knotvector)
        knotvector.normalize()
        return knotvector
