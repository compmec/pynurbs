"""
This file contains a class Polynomial that allows evaluating and
making operations with polynomials, like adding, multiplying, etc
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, List, Union


class Polynomial:
    """
    Defines a polynomial with coefficients

    p(x) = a0 + a1 * x + a2 * x^2 + ... + ap * x^p

    By receiving the coefficients

    coefs = [a0, a1, a2, ..., ap]

    This class allows evaluating, adding, multiplying, etc

    Example
    -------
    >>> poly = Polynomial([3, 2])
    >>> poly(0)
    3
    >>> poly(1)
    5
    """

    def __init__(self, coefs: Iterable[Real]):
        coefs = tuple(coefs)
        if len(coefs) == 0:
            coefs = (0,)
        self.__coefs = tuple(coefs)

    @property
    def degree(self) -> int:
        """
        Gives the degree of the polynomial
        """
        return len(self.__coefs) - 1

    def __iter__(self):
        yield from self.__coefs

    def __getitem__(self, index):
        return self.__coefs[index]

    def __neg__(self) -> Polynomial:
        coefs = tuple(-coef for coef in self)
        return self.__class__(coefs)

    def __add__(self, other: Union[Real, Polynomial]) -> Polynomial:
        if isinstance(other, Polynomial):
            coefs = [0] * (1 + max(self.degree, other.degree))
            for i, coef in enumerate(self):
                coefs[i] += coef
            for i, coef in enumerate(other):
                coefs[i] += coef
        else:
            coefs = list(self)
            coefs[0] += other
        return self.__class__(coefs)

    def __mul__(self, other: Union[Real, Polynomial]) -> Polynomial:
        if isinstance(other, Polynomial):
            coefs = [0] * (self.degree + other.degree + 1)
            for i, coefi in enumerate(self):
                for j, coefj in enumerate(other):
                    coefs[i + j] += coefi * coefj
        else:
            coefs = tuple(other * coef for coef in self)
        return self.__class__(coefs)

    def __truediv__(self, other: Real) -> Polynomial:
        coefs = (coef / other for coef in self)
        return self.__class__(coefs)

    def __pow__(self, other: int) -> Polynomial:
        result = self
        for _ in range(int(other) - 1):
            result = result * self
        return result

    def __sub__(self, other: Union[Real, Polynomial]) -> Polynomial:
        return self.__add__(-other)

    def __rsub__(self, other: Real) -> Polynomial:
        return (-self).__add__(other)

    def __radd__(self, other: Real) -> Polynomial:
        return self.__add__(other)

    def __rmul__(self, other: Real) -> Polynomial:
        return self.__mul__(other)

    def __call__(self, node: Real) -> Real:
        return self.eval(node, 0)

    def __str__(self):
        msgs: List[str] = []
        flag = False
        for i, coef in enumerate(self):
            if coef == 0:
                continue
            if coef < 0:
                msg = "- "
            elif flag:
                msg = "+ "
            else:
                msg = ""
            flag = True
            coef = abs(coef)
            if coef != 1 or i == 0:
                msg += str(coef)
            if i > 0:
                if coef != 1:
                    msg += " * "
                msg += "x"
            if i > 1:
                msg += f"^{i}"
            msgs.append(msg)
        return " ".join(msgs)

    def __repr__(self) -> str:
        return str(self)

    def eval(self, node: Real, derivate: int = 0) -> Real:
        """
        Evaluates the polynomial at given node

        Example
        -------
        >>> poly = Polynomial([1, 2])
        >>> poly.eval(0)
        1
        >>> poly.eval(1)
        3
        >>> poly.eval(1, 1)
        2
        >>> poly.eval(1, 2)
        0
        """
        if not derivate:
            coefs = self.__coefs
        else:
            coefs = tuple(self.derivate(derivate))
        if len(coefs) == 1:
            return coefs[0]
        result: Real = 0 * coefs[0]
        for coef in coefs[::-1]:
            result = node * result + coef
        return result

    def derivate(self, times: int = 1) -> Polynomial:
        """
        Derivate the polynomial curve, giving a new one

        Example
        -------
        >>> poly = Polynomial([1, 2, 5])
        >>> print(poly)
        1 + 2 * x + 5 * x^2
        >>> dpoly = poly.derivate()
        >>> print(dpoly)
        2 + 10 * x
        """
        coefs = (
            math.factorial(n + times) * coef / math.factorial(n)
            for n, coef in enumerate(self[times:])
        )
        return self.__class__(coefs)


def scale(polynomial: Polynomial, amount: Real) -> Polynomial:
    """
    Transforms the polynomial p(x) into p(A*x) by
    scaling the argument of the polynomial by 'A'.

    p(x) = a0 + a1 * x + ... + ap * x^p
    p(A * x) = a0 + a1 * (A*x) + ... + ap * (A * x)^p
             = b0 + b1 * x + ... + bp * x^p

    Example
    -------
    >>> old_poly = Polynomial([0, 0, 0, 1])
    >>> print(old_poly)
    x^3
    >>> new_poly = scale(poly, 1)  # transform to (x-1)^3
    >>> print(new_poly)
    - 1 + 3 * x - 3 * x^2 + x^3
    """
    coefs = tuple(coef * amount**i for i, coef in enumerate(polynomial))
    return Polynomial(coefs)


def shift(polynomial: Polynomial, amount: Real) -> Polynomial:
    """
    Transforms the polynomial p(x) into p(x-d) by
    translating the polynomial by 'd' to the right.

    p(x) = a0 + a1 * x + ... + ap * x^p
    p(x-d) = a0 + a1 * (x-d) + ... + ap * (x-d)^p
            = b0 + b1 * x + ... + bp * x^p

    Example
    -------
    >>> old_poly = Polynomial([0, 0, 0, 1])
    >>> print(old_poly)
    x^3
    >>> new_poly = shift(poly, 1)  # transform to (x-1)^3
    >>> print(new_poly)
    - 1 + 3 * x - 3 * x^2 + x^3
    """
    newcoefs = list(polynomial)
    for i, coef in enumerate(polynomial):
        for j in range(i):
            binom = math.comb(i, j)
            value = binom * (amount ** (i - j))
            if (i + j) % 2:
                value *= -1
            newcoefs[j] += coef * value
    return Polynomial(newcoefs)
