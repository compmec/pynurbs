"""
Finds the roots of polynomials
"""

from fractions import Fraction
from numbers import Real
from typing import Tuple

import numpy as np

from .polynomial import Polynomial


def division(poly: Polynomial, doly: Polynomial) -> Tuple[Polynomial, Polynomial]:
    """
    Given the polynomials poly and doly, finds qoly and roly such:

    poly = doly * qoly + roly

    with:
    * degree(qoly) = degree(poly) - degree(doly)
    * degree(roly) < degree(doly)
    """
    if not isinstance(poly, Polynomial) or not isinstance(doly, Polynomial):
        raise TypeError
    if doly.degree > poly.degree:
        return Polynomial([0]), poly
    if doly.degree == 0:
        return Polynomial([coef / doly[0] for coef in poly]), Polynomial([0])
    qoly = Polynomial([0])
    roly = Polynomial(poly)
    index = poly.degree
    while index >= doly.degree:
        const = roly[index] / doly[doly.degree]
        qoly += Polynomial([0] * (index - doly.degree) + [const])
        roly = poly - doly * qoly
        index -= 1
    return qoly, Polynomial(roly[: doly.degree])


def roots(poly: Polynomial) -> Tuple[Real, ...]:
    """
    Finds the real roots of the given polynomial

    Example
    -------
    >>> x = Polynomial([0, 1])
    >>> roots(x**2 + 3*x + 2)
    (-2, -1)
    >>> roots(x**3 - 6*x**2 + 11*x - 6)
    (1, 2, 3)
    """
    values = sorted(np.roots(tuple(poly)[::-1]))
    for i, value in enumerate(values):
        if abs(round(1440 * value, 0) - 1440 * value) < 1e-6:
            values[i] = Fraction(round(1440 * value), 1440)
    return tuple(values)
