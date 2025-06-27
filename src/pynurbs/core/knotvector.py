from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


class ImmutableKnotVector(tuple):
    @staticmethod
    def __get_unique(vector: Tuple[float]):
        unique = []
        for node in vector:
            for knot in unique:
                if abs(node - knot) < 1e-6:
                    break
            else:
                unique.append(node)
        unique.sort()
        return tuple(unique)

    @staticmethod
    def __is_valid(vector: Tuple[float], degree: Union[int, None]):
        try:
            for knot in vector:
                float(knot)
        except TypeError:
            return False
        lenght = len(vector)
        if lenght < 2:
            return False
        for i in range(lenght - 1):
            if not vector[i] <= vector[i + 1]:
                return False
        if degree is None:
            degree = 0
            while vector[degree] == vector[degree + 1]:
                degree += 1
        npts = lenght - degree - 1
        if not degree < npts:
            return False
        knots = ImmutableKnotVector.__get_unique(vector[degree : npts + 1])
        for knot in knots:
            mult = vector.count(knot)
            if mult > degree + 1:
                return False
        if vector.count(vector[degree]) != vector.count(vector[npts]):
            return False
        return True

    def __new__(cls, knotvector: Tuple[float], degree: Optional[int] = None):
        if isinstance(knotvector, ImmutableKnotVector):
            return knotvector
        try:
            knotvector = tuple(knotvector)
        except TypeError:
            raise ValueError
        if not cls.__is_valid(knotvector, degree):
            msg = f"Invalid knot vector (deg {degree}): {knotvector}"
            raise ValueError(msg)
        if degree is None:
            degree = 0
            while knotvector[degree] == knotvector[degree + 1]:
                degree += 1
        instance = super(ImmutableKnotVector, cls).__new__(cls, tuple(knotvector))
        instance._ImmutableKnotVector__degree = degree
        instance._ImmutableKnotVector__npts = len(knotvector) - degree - 1
        return instance

    def __or__(self, other: ImmutableKnotVector) -> ImmutableKnotVector:
        other = ImmutableKnotVector(other)
        if self.limits != other.limits:
            raise ValueError
        all_knots = list(self.knots) + list(other.knots)
        all_knots = ImmutableKnotVector.__get_unique(all_knots)
        all_mults = [0] * len(all_knots)
        for vector in [self, other]:
            for knot in vector:
                index = all_knots.index(knot)
                mult = vector.mult(knot)
                if mult > all_mults[index]:
                    all_mults[index] = mult
        final_vector = []
        for knot, mult in zip(all_knots, all_mults):
            final_vector += [knot] * mult
        final_vector = tuple(sorted(final_vector))
        return ImmutableKnotVector(final_vector)

    def __and__(self, other: ImmutableKnotVector) -> ImmutableKnotVector:
        other = ImmutableKnotVector(other)
        if self.limits != other.limits:
            raise ValueError
        all_knots = tuple(sorted(set(self.knots) & set(other.knots)))
        all_mults = [float("inf")] * len(all_knots)
        for vector in [self, other]:
            for knot in vector:
                if knot not in all_knots:
                    continue
                index = all_knots.index(knot)
                mult = vector.mult(knot)
                if mult < all_mults[index]:
                    all_mults[index] = mult
        final_vector = []
        for knot, mult in zip(all_knots, all_mults):
            final_vector += [knot] * mult
        return ImmutableKnotVector(sorted(final_vector))

    def __add__(self, other):
        raise ValueError

    def __sub__(self, other):
        raise ValueError

    @property
    def degree(self) -> int:
        return self.__degree

    @property
    def npts(self) -> int:
        return self.__npts

    @property
    def knots(self) -> Tuple[float]:
        vector = self[self.degree : self.npts + 1]
        return ImmutableKnotVector.__get_unique(vector)

    @property
    def limits(self) -> Tuple[float]:
        return (self[self.degree], self[self.npts])

    def __span_single(self, node: float) -> int:
        if node == self[self.npts]:  # Special case
            return self.npts - 1
        low, high = self.degree, self.npts + 1  # Do binary search
        mid = (low + high) // 2
        while True:
            if node < self[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
            if self[mid] <= node < self[mid + 1]:
                return mid

    def __mult_single(self, node: Tuple[float]) -> Tuple[int]:
        return sum(abs(node - knot) < 1e-9 for knot in self)

    def __valid_single(self, node: float) -> bool:
        try:
            float(node)  # Verify if it's a number
        except TypeError:
            return False
        umin, umax = self.limits
        if node < umin or umax < node:
            return False
        return True

    def span(self, nodes: Union[float, Tuple[float]]) -> Union[int, Tuple[int]]:
        if not self.valid(nodes):
            raise ValueError
        try:
            return tuple(map(self.span, nodes))
        except TypeError:
            return self.__span_single(nodes)

    def mult(self, nodes: Union[float, Tuple[float]]) -> Union[int, Tuple[int]]:
        if not self.valid(nodes):
            raise ValueError
        try:
            return tuple(map(self.mult, nodes))
        except TypeError:
            return self.__mult_single(nodes)

    def valid(self, nodes: Tuple[float]) -> bool:
        if isinstance(nodes, str):
            return False
        try:
            for node in nodes:
                if not self.valid(node):
                    return False
            return True
        except TypeError:
            return self.__valid_single(nodes)

    def split(self, nodes: Tuple[float]) -> Tuple[ImmutableKnotVector]:
        """
        It splits the knotvector at nodes.
        You may put initial and final values, but they are ignored.
        Example:
            >> U = [0, 0, 0.5, 1, 1]
            >> split(U, [0.5])
            [[0, 0, 0.5, 0.5],
             [0.5, 0.5, 1, 1]]
            >> split(U, [0.25])
            [[0, 0, 0.25, 0.25],
             [0.25, 0.25, 0.5, 1, 1]]
            >> split(U, [0, 0.25, 0.75])
            [[0, 0, 0.25, 0.25],
             [0.25, 0.25, 0.5, 0.75, 0.75],
             [0.75, 0.75, 1, 1]]
        """
        if not self.valid(nodes):
            raise ValueError
        nodes = set(nodes)
        if len(nodes) == 0:
            return (self,)
        nodes = tuple(sorted(nodes | set(self.limits)))
        vector = np.array(tuple(self))

        retorno = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            middle = list(vector[(a < vector) * (vector < b)])
            newknotvect = (self.degree + 1) * [a] + middle + (self.degree + 1) * [b]
            newknotvect = ImmutableKnotVector(newknotvect)
            retorno.append(newknotvect)
        return tuple(retorno)

    def increase(self, times: int) -> ImmutableKnotVector:
        """Degree increase"""
        vector = sorted(list(self) + times * list(self.knots))
        return self.__class__(vector, self.degree + times)

    def decrease(self, times: int) -> ImmutableKnotVector:
        """Degree decrease"""
        vector = list(self)
        knots = self.knots[1:-1]
        for _ in range(times):
            vector.pop(0)
            vector.pop(-1)
            for node in knots:
                vector.remove(node)
            vector = sorted(vector)
        return self.__class__(vector, self.degree - times)

    def remove(self, nodes: Tuple[float]) -> ImmutableKnotVector:
        """Remove knots"""
        vector = list(self)
        for node in nodes:
            vector.remove(node)
        vector = sorted(vector)
        return self.__class__(vector, self.degree)

    def insert(self, nodes: Tuple[float]) -> ImmutableKnotVector:
        """Insert knots"""
        vector = sorted(list(self) + list(nodes))
        return self.__class__(vector, self.degree)
