from __future__ import annotations

from collections import Counter
from numbers import Real
from typing import Iterable, Tuple, Union

from .custom_math import isnumber


def find_degree(vector: Tuple[Real, ...]) -> int:
    return max(Counter(vector).values()) - 1


def is_sorted(vector: Tuple[Real, ...]) -> bool:
    return all(val <= vector[i + 1] for i, val in enumerate(vector[:-1]))


class ImmutableKnotVector:

    def __init__(self, vector: Iterable[Real], degree: Union[None, int] = None):
        try:
            vector = tuple(vector)
        except Exception:
            raise ValueError(f"Wrong argument: '{vector}'")
        if not all(map(isnumber, vector)):
            raise ValueError(f"Cannot create KnotVector with {vector}")
        if not is_sorted(vector):
            raise ValueError(f"Cannot create KnotVector with {vector}")
        if degree is None:
            degree = find_degree(vector)
        elif int(degree) < find_degree(vector):
            raise ValueError(f"Cannot create KnotVector with {vector}")
        npts = len(vector) - degree - 1
        if degree >= npts:
            raise ValueError(f"Cannot have {degree} <= {npts}")
        knots = tuple(sorted(set(vector[degree : npts + 1])))
        if len(knots) < 2:
            raise ValueError(f"Cannot create KnotVector with {vector}")
        if vector[degree] == vector[degree + 1]:
            raise ValueError(f"Cannot create KnotVector with {vector}")
        if degree != 0 and vector[npts - 1] == vector[npts]:
            raise ValueError(f"Cannot create KnotVector with {vector}")
        self.__degree = degree
        self.__npts = npts
        self.__knots = knots
        self.__vector = vector

    @property
    def degree(self) -> int:
        return self.__degree

    @property
    def npts(self) -> int:
        return self.__npts

    @property
    def knots(self) -> Tuple[Real, ...]:
        return self.__knots

    @property
    def limits(self) -> Tuple[Real, Real]:
        return (self[self.degree], self[self.npts])

    def __str__(self) -> str:
        return "(" + ", ".join(map(str, self)) + ")"

    def __repr__(self) -> str:
        return "(" + ", ".join(map(repr, self)) + ")"

    def __getitem__(self, index):
        return self.__vector[index]

    def __len__(self) -> int:
        return len(self.__vector)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImmutableKnotVector):
            return self.degree == other.degree and tuple(self) == tuple(other)
        try:
            return tuple(self) == tuple(other)
        except Exception:
            return NotImplemented

    def span(self, node: Real) -> int:
        if not isnumber(node):
            raise ValueError(f"Node '{node}' must be Real instance")
        if node < self[self.degree] or self[self.npts] < node:
            raise ValueError(f"Node {node} outside [{self.knots[0], self.knots[-1]}]")
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

    def mult(self, node: Real) -> int:
        if not isnumber(node):
            raise ValueError(f"Node '{node}' must be Real instance")
        if node < self[self.degree] or self[self.npts] < node:
            raise ValueError(f"Node {node} outside [{self.knots[0], self.knots[-1]}]")
        return sum(abs(node - knot) < 1e-9 for knot in self)
