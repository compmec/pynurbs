"""
Defines the Piecewise Polynomial class
"""

from numbers import Real
from typing import Iterable, Tuple

from .polynomial import Polynomial


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

    def eval(self, node: Real, times: int = 0) -> Real:
        nsegs = len(self.functions)
        mask = self.knots[nsegs - 1] <= node
        mask *= node < self.knots[nsegs]
        result = mask * self.functions[-1].eval(node, times)
        for i in range(nsegs - 1):
            mask = self.knots[i] <= node
            mask *= node < self.knots[i + 1]
            result += mask * self.functions[i].eval(node, times)
        return result
