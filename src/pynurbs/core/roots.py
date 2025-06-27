"""
Finds the roots of polynomials
"""

from typing import Tuple

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
