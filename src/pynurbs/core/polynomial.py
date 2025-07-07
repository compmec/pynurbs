"""
This file contains a class Polynomial that allows evaluating and
making operations with polynomials, like adding, multiplying, etc
"""

from __future__ import annotations

from numbers import Real
from typing import Iterable, List, Tuple, Union

from .custom_math import Math, isscalar, supports_linear_operation


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
        coefs = tuple(coefs) if not isscalar(coefs) else (coefs,)
        if len(coefs) == 0:
            raise ValueError("Cannot receive an empty tuple")
        if isscalar(coefs[0]):
            degree = max((i for i, v in enumerate(coefs) if v), default=0)
        else:
            degree = len(coefs) - 1
        coefs = coefs[: degree + 1]
        if not all(map(supports_linear_operation, coefs)):
            raise ValueError
        self.__coefs = tuple(coefs[: degree + 1])

    @property
    def degree(self) -> int:
        """
        Gives the degree of the polynomial
        """
        return len(self.__coefs) - 1

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Polynomial):
            return tuple(self) == tuple(value)
        return self.degree == 0 and value == self[0]

    def __iter__(self):
        yield from self.__coefs

    def __getitem__(self, index):
        return self.__coefs[index]

    def __neg__(self) -> Polynomial:
        return self.__class__(-coef for coef in self)

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

    def __matmul__(self, other: Union[Real, Polynomial]) -> Polynomial:
        if not isinstance(other, Polynomial):
            newcoefs = tuple(coef @ other for coef in self)
            print(newcoefs)
            return self.__class__(newcoefs)
        coefs = [0] * (self.degree + other.degree + 1)
        for i, coefi in enumerate(self):
            for j, coefj in enumerate(other):
                coefs[i + j] += coefi @ coefj
        return self.__class__(coefs)

    def __truediv__(self, other: Real) -> Polynomial:
        coefs = (coef / other for coef in self)
        return self.__class__(coefs)

    def __pow__(self, other: int) -> Polynomial:
        if other == 0:
            return self.__class__([1 + 0 * sum(self)])
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

    def __rmatmul__(self, other: Real) -> Polynomial:
        return self.__matmul__(other)

    def __call__(self, node: Real) -> Real:
        if self.degree == 0:
            return self[0]
        result: Real = 0 * self[0]
        for coef in self[::-1]:
            result = node * result + coef
        return result

    def __str__(self):
        if self.degree == 0:
            return str(self[0])
        msgs: List[str] = []
        if not isscalar(self[0]):
            msgs.append(f"({self[0]})")
            if self.degree > 0:
                msgs.append(f"({self[1]}) * x")
            for i, coef in enumerate(self[2:]):
                msgs.append(f"({coef}) * x^{i+2}")
            return " + ".join(msgs)
        flag = False
        for i, coef in enumerate(self):
            if coef == 0:
                continue
            msg = "- " if coef < 0 else "+ " if flag else ""
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
            value = Math.binom(i, j) * (amount ** (i - j))
            if (i + j) % 2:
                value *= -1
            newcoefs[j] += coef * value
    return Polynomial(newcoefs)


def derivate(polynomial: Polynomial, times: int = 1) -> Polynomial:
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
    if polynomial.degree < times:
        return Polynomial([0 * polynomial[0]])
    coefs = (
        Math.factorial(n + times) // Math.factorial(n) * coef
        for n, coef in enumerate(polynomial[times:])
    )
    return Polynomial(coefs)


def integrate(polynomial: Polynomial, domain: Tuple[Real, Real]) -> Real:
    """
    Computes the definite integral of a polynomial

    Example
    -------
    >>> poly = Polynomial([1, 2, 5])
    >>> print(poly)
    1 + 2 * x + 5 * x^2
    >>> integrate(poly, (-2, 1))
    15
    """
    return sum(
        coef * (domain[1] ** (n + 1) - domain[0] ** (n + 1)) / (n + 1)
        for n, coef in enumerate(polynomial)
    )
