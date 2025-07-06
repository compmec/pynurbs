"""
Finds the roots of polynomials
"""

from fractions import Fraction
from typing import Tuple, Union

import numpy as np
import rbool

from ..core.custom_math import isscalar
from ..core.piecepoly import PiecewisePolynomial
from ..core.polynomial import Polynomial


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


def roots_polynomial(poly: Polynomial) -> rbool.SubSetR1:
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
    if not isinstance(poly, Polynomial):
        raise TypeError
    if not all(map(isscalar, poly)):
        raise ValueError
    if poly.degree == 0:
        return rbool.WholeR1() if poly[0] == 0 else rbool.EmptyR1()
    result = rbool.EmptyR1()
    for value in np.roots(tuple(poly)[::-1]):
        if not isinstance(value, complex):
            if abs(round(1440 * value, 0) - 1440 * value) < 1e-6:
                value = Fraction(round(1440 * value), 1440)
            result |= value
    return result


def roots_piecewise(piece: PiecewisePolynomial) -> rbool.SubSetR1:
    """
    Finds the real roots of the piecewise polynomial function
    """
    result = rbool.EmptyR1()
    for i, poly in enumerate(piece.functions):
        knota, knotb = piece.knots[i], piece.knots[i + 1]
        closed_right = i + 1 == len(piece.functions)
        interval = rbool.IntervalR1(knota, knotb, True, closed_right)
        result |= interval & roots_polynomial(poly)
    return result


def roots(function: Union[Polynomial, PiecewisePolynomial]) -> rbool.SubSetR1:
    if isinstance(function, Polynomial):
        return roots_polynomial(function)
    elif isinstance(function, PiecewisePolynomial):
        return roots_piecewise(function)
    raise ValueError
