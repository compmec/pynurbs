from numbers import Real
from typing import Tuple, Union

import numpy as np

from .custom_math import totuple
from .knotvector import ImmutableKnotVector
from .piecepoly import PiecewisePolynomial
from .polynomial import Polynomial, scale, shift


def spectral_matrix(
    knotvector: ImmutableKnotVector, reqdegree: int
) -> Tuple[Tuple[Tuple[Real, ...], ...], ...]:
    """
    Given a knotvector, it has properties like
        - number of points: npts
        - polynomial degree: degree
        - knots: A list of non-repeted knots
        - spans: The span of each knot
    This function returns a matrix of size
        (m) x (j+1) x (j+1)
    which
        - m is the number of segments: len(knots)-1
        - j is the requested degree
    """
    if not isinstance(knotvector, ImmutableKnotVector):
        raise TypeError
    if not isinstance(reqdegree, int):
        raise TypeError("reqdegree must be integer")
    if reqdegree < 0 or knotvector.degree < reqdegree:
        raise ValueError(f"reqdegree must be in [0, {knotvector.degree}]")
    knots = knotvector.knots
    spans = tuple(map(knotvector.span, knots))
    j = reqdegree

    ninter = len(knots) - 1
    matrix = [[[0 * knots[0]] * (j + 1)] * (j + 1)] * ninter
    matrix = np.array(matrix, dtype="object")
    if j == 0:
        matrix.fill(1)
        return matrix
    matrix_less1 = spectral_matrix(knotvector, j - 1)
    matrix_less1 = np.array(matrix_less1).tolist()
    for y in range(j):
        for z, sz in enumerate(spans[:-1]):
            i = y + sz - j + 1
            denom = knotvector[i + j] - knotvector[i]
            for k in range(j):
                matrix_less1[z][y][k] /= denom

            a0 = knots[z] - knotvector[i]
            a1 = knots[z + 1] - knots[z]
            b0 = knotvector[i + j] - knots[z]
            b1 = knots[z] - knots[z + 1]
            for k in range(j):
                matrix[z][y][k] += b0 * matrix_less1[z][y][k]
                matrix[z][y][k + 1] += b1 * matrix_less1[z][y][k]
                matrix[z][y + 1][k] += a0 * matrix_less1[z][y][k]
                matrix[z][y + 1][k + 1] += a1 * matrix_less1[z][y][k]
    return totuple(matrix)


class ImmutableSplineBasis:

    def __init__(
        self, knotvector: ImmutableKnotVector, degree: Union[int, None] = None
    ):
        if not isinstance(knotvector, ImmutableKnotVector):
            raise TypeError
        degree = degree if degree is not None else knotvector.degree
        self.__matrix = tuple(
            tuple(tuple(Polynomial(coefs) for coefs in all_coefs))
            for all_coefs in spectral_matrix(knotvector, degree)
        )
        self.__degree = degree
        self.__npts = knotvector.npts
        self.__knotvector = knotvector

    @property
    def degree(self) -> int:
        return self.__degree

    @property
    def npts(self) -> int:
        return self.__npts

    @property
    def knots(self) -> Tuple[Real, ...]:
        return self.__knotvector.knots

    def __getitem__(self, index: int) -> PiecewisePolynomial:
        index = int(index)
        basis = list(polys[index] for polys in self.__matrix)
        for i, base in enumerate(basis):
            knota, knotb = self.knots[i], self.knots[i + 1]
            basis[i] = shift(scale(base, knotb - knota), knota)
        return PiecewisePolynomial(basis, self.knots)

    def __call__(self, node: Real) -> Tuple[Real, ...]:
        npts = self.__knotvector.npts
        knots = self.__knotvector.knots
        spans = tuple(map(self.__knotvector.span, knots))
        degree = self.__degree
        result = [0] * npts

        span = self.__knotvector.span(node)
        ind = spans.index(span)
        shifnode = node - knots[ind]
        shifnode /= knots[ind + 1] - knots[ind]
        for y in range(self.__degree + 1):
            i = y + span - degree
            polynomial = self.__matrix[ind][y]
            result[i] = polynomial(shifnode)
        return tuple(result)
