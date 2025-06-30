"""
Defines the Piecewise Polynomial class
"""

from __future__ import annotations

from numbers import Real
from typing import Iterable, Tuple, Union

from .polynomial import Polynomial
from .tools import vectorize


def find_span(node: Real, knots: Tuple[Real, ...]):
    """
    Finds the span of the given node

    Example
    -------
    >>> knots = [0, 1, 3, 4]
    >>> find_span(-1, knots)
    -1
    >>> find_span(0, knots)
    0
    >>> find_span(0.5, knots)
    0
    >>> find_span(1, knots)
    1
    >>> find_span(2, knots)
    1
    >>> find_span(3, knots)
    2
    >>> find_span(4, knots)
    2
    >>> find_span(5, knots)
    3
    """
    if node < knots[0]:
        return -1
    if knots[-1] < node:
        return len(knots) - 1
    for i, knot in enumerate(knots[:-1]):
        if knot <= node:
            return i
    return len(knots) - 2


class PiecewisePolynomial:
    """
    Defines a Polynomial piecewise function
    """

    def __init__(self, functions: Iterable[Polynomial], knots: Iterable[Real]) -> None:
        self.__functions = tuple(functions)
        self.__knots = tuple(knots)

    @property
    def knots(self) -> Tuple[Real, ...]:
        return self.__knots

    @property
    def functions(self) -> Tuple[Polynomial, ...]:
        return self.__functions

    @vectorize(1, 0)
    def __call__(self, node: Real) -> Real:
        span = find_span(node, self.knots)
        function = self.functions[span]
        return function(node)

    def __neg__(self) -> Polynomial:
        return self.__class__((-func for func in self.functions), self.knots)

    def __add__(
        self, other: Union[Real, Polynomial, PiecewisePolynomial]
    ) -> PiecewisePolynomial:
        if not isinstance(other, PiecewisePolynomial):
            return self.__class__((func + other for func in self.functions), self.knots)
        allknots = sorted(set(self.knots) | set(other.knots))
        functions = [None] * (len(allknots) - 1)
        for i, (knota, knotb) in enumerate(zip(allknots, allknots[1:])):
            midknot = (knota + knotb) / 2
            spana = find_span(midknot, self.knots)
            spanb = find_span(midknot, self.knots)
            functions[i] = self.functions[spana] + other.functions[spanb]
        return self.__class__(functions, allknots)

    def __mul__(
        self, other: Union[Real, Polynomial, PiecewisePolynomial]
    ) -> PiecewisePolynomial:
        if not isinstance(other, PiecewisePolynomial):
            return self.__class__((func + other for func in self.functions), self.knots)
        allknots = sorted(set(self.knots) | set(other.knots))
        functions = [None] * (len(allknots) - 1)
        for i, (knota, knotb) in enumerate(zip(allknots, allknots[1:])):
            midknot = (knota + knotb) / 2
            spana = find_span(midknot, self.knots)
            spanb = find_span(midknot, self.knots)
            functions[i] = self.functions[spana] * other.functions[spanb]
        return self.__class__(functions, allknots)

    def __matmul__(
        self, other: Union[Real, Polynomial, PiecewisePolynomial]
    ) -> PiecewisePolynomial:
        if not isinstance(other, PiecewisePolynomial):
            return self.__class__((func + other for func in self.functions), self.knots)
        allknots = sorted(set(self.knots) | set(other.knots))
        functions = [None] * (len(allknots) - 1)
        for i, (knota, knotb) in enumerate(zip(allknots, allknots[1:])):
            midknot = (knota + knotb) / 2
            spana = find_span(midknot, self.knots)
            spanb = find_span(midknot, self.knots)
            functions[i] = self.functions[spana] @ other.functions[spanb]
        return self.__class__(functions, allknots)

    def __sub__(
        self, other: Union[Real, Polynomial, PiecewisePolynomial]
    ) -> PiecewisePolynomial:
        return self.__add__(-other)

    def __rsub__(self, other: Real) -> PiecewisePolynomial:
        return (-self).__add__(other)

    def __radd__(self, other: Real) -> PiecewisePolynomial:
        return self.__add__(other)

    def __rmul__(self, other: Real) -> PiecewisePolynomial:
        return self.__mul__(other)

    def __rmatmul__(self, other: Real) -> PiecewisePolynomial:
        return self.__matmul__(other)
