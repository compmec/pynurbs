"""
Given two hypotetic curves C0 and C1, which are associated
with knotvectors U and V, and control points P and Q.
    C0(u) = sum_{i=0}^{n-1} N_{i}(u) * P_{i}
    C1(u) = sum_{i=0}^{m-1} M_{i}(u) * Q_{i}
Then, this class has functions to return [T] and [E] such
    [Q] = [T] * [P]
    error = [P]^T * [E] * [P]
Then, C1 keeps near to C1 by using galerkin projections.

They minimizes the integral
    int_{a}^{b} abs(C0(u) - C1(u))^2 du
The way it does it by using the norm of inner product:
    abs(X) = sqrt(< X, X >)
Then finally it finds the matrix [A] and [B]
    [C] * [Q] = [B] * [P]
    [A]_{ij} = int_{0}^{1} < Ni(u), Nj(u) > du
    [B]_{ij} = int_{0}^{1} < Mi(u), Nj(u) > du
    [C]_{ij} = int_{0}^{1} < Mi(u), Mj(u) > du
    --> [T] = [C]^{-1} * [B]^T
    --> [E] = [A] - [B] * [T]
"""

from __future__ import annotations

from fractions import Fraction
from typing import Tuple, Union

import numpy as np

from ..core.custom_math import (
    IntegratorArray,
    Linalg,
    NodeSample,
    number_type,
    totuple,
)
from ..core.knotvector import ImmutableKnotVector
from ..core.spline_basis import ImmutableSplineBasis


def eval_spline_nodes(
    knotvector: ImmutableKnotVector, nodes: Tuple[float], degree: int
) -> Tuple[Tuple[float]]:
    """
    Returns a matrix M of which M_{ij} = N_{i,degree}(node_j)
    M.shape = (npts, len(nodes))
    """
    knotvector = ImmutableKnotVector(knotvector)
    basis = ImmutableSplineBasis(knotvector, degree)
    return np.transpose(tuple(map(basis, nodes)))


def eval_rational_nodes(
    knotvector: ImmutableKnotVector,
    weights: Tuple[float],
    nodes: Tuple[float],
    degree: int,
) -> Tuple[Tuple[float]]:
    """
    Returns a matrix M of which M_{ij} = N_{i,p}(node_j)
    M.shape = (len(weights), len(nodes))
    """
    matrix = eval_spline_nodes(knotvector, nodes, degree)
    denominators = 1 / np.dot(weights, matrix)
    return np.einsum("j,ij,i->ij", denominators, matrix, weights)


def fit_function(
    knotvector: ImmutableKnotVector,
    nodes: Tuple[float],
    weights: Union[Tuple[float], None],
) -> Tuple[Tuple[float]]:
    """
    Let C(u) be a curve C(u) of base functions F of given knot vector
        C(u) = sum_i F_i(u) * P_i
    it's wanted to fit a C(u) into the curve f(u)

    To do it, we do least squares by minimizing
        J(P) = sum_j abs(C(nodej)-f(nodej))

    This function returns a matrix M such
        [P] = [M] * [f(nodes)]
    """
    knotvector = ImmutableKnotVector(knotvector)
    basis = ImmutableSplineBasis(knotvector)
    matrix = np.transpose(tuple(map(basis, nodes)))
    if weights is not None:
        denominators = 1 / np.dot(weights, matrix)
        matrix = np.einsum("j,ij->ij", denominators, matrix)
    return Linalg.lstsq(np.transpose(matrix))


def spline2spline(
    oldknotvector: ImmutableKnotVector,
    newknotvector: ImmutableKnotVector,
    fit_nodes: Tuple[float] = None,
) -> Tuple["Matrix2D"]:
    """
    Given two bspline curves A(u) and B(u), this
    function returns a matrix [M] such
        [Q] = [M] * [P]
        A(u) = sum_i N_i(u) * P_i
        B(u) = sum_i N_i(u) * Q_i
    """
    oldknotvector = ImmutableKnotVector(oldknotvector)
    newknotvector = ImmutableKnotVector(newknotvector)
    oldnpts = oldknotvector.npts
    newnpts = newknotvector.npts
    oldweights = [Fraction(1) for i in range(oldnpts)]
    newweights = [Fraction(1) for i in range(newnpts)]
    result = func2func(
        oldknotvector, oldweights, newknotvector, newweights, fit_nodes
    )
    return totuple(result)


def func2func(
    oldknotvector: ImmutableKnotVector,
    oldweights: Tuple[float],
    newknotvector: ImmutableKnotVector,
    newweights: Tuple[float],
    fit_nodes: Tuple[float] = None,
) -> Tuple[np.ndarray]:
    """
    Given two rational bspline curves A(u) and B(u), this
    function returns a matrix [M] such
        [Q] = [M] * [P]
        A(u) = sum_i R_i(u) * P_i
        B(u) = sum_i R_i(u) * Q_i
    """
    oldknotvector = ImmutableKnotVector(oldknotvector)
    newknotvector = ImmutableKnotVector(newknotvector)
    for val in oldweights:
        float(val)
    for val in newweights:
        float(val)

    olddegree = oldknotvector.degree
    oldnpts = oldknotvector.npts
    oldknots = oldknotvector.knots

    newdegree = newknotvector.degree
    newnpts = newknotvector.npts
    newknots = newknotvector.knots

    oldknotvector = tuple(
        Fraction(node) if isinstance(node, int) else node
        for node in oldknotvector
    )
    newknotvector = tuple(
        Fraction(node) if isinstance(node, int) else node
        for node in newknotvector
    )
    oldknotvector = ImmutableKnotVector(oldknotvector, olddegree)
    newknotvector = ImmutableKnotVector(newknotvector, newdegree)

    if fit_nodes and len(fit_nodes) > newnpts:
        raise NotImplementedError
    allknots = list(set(oldknots + newknots))
    allknots.sort()

    numbtype = number_type(allknots)
    numbtype = Fraction if (numbtype is int) else numbtype
    nptsinteg = olddegree + newdegree + 3  # Number integration points
    if numbtype is Fraction:
        nodes0to1 = NodeSample.closed_linspace(nptsinteg)
        integrator = IntegratorArray.closed_newton_cotes(nptsinteg)
    else:
        nodes0to1 = NodeSample.chebyshev(nptsinteg)
        integrator = IntegratorArray.chebyshev(nptsinteg)
    nodes0to1 = np.array(nodes0to1)
    integrator = np.array(integrator, dtype=numbtype)

    FF = np.zeros((oldnpts, oldnpts), dtype=numbtype)  # F*F
    GF = np.zeros((newnpts, oldnpts), dtype=numbtype)  # F*G
    GG = np.zeros((newnpts, newnpts), dtype=numbtype)  # G*G
    for start, end in zip(allknots[:-1], allknots[1:]):
        nodes = start + (end - start) * nodes0to1
        # Integral of the functions in the interval [a, b]
        Fvalues = eval_rational_nodes(
            oldknotvector, oldweights, tuple(nodes), olddegree
        )
        Gvalues = eval_rational_nodes(
            newknotvector, newweights, tuple(nodes), newdegree
        )
        Fvalues = np.array(Fvalues, dtype=numbtype)
        Gvalues = np.array(Gvalues, dtype=numbtype)
        for k, integ in enumerate(integrator):
            FF += integ * np.tensordot(Fvalues[:, k], Fvalues[:, k], axes=0)
            GF += integ * np.tensordot(Gvalues[:, k], Fvalues[:, k], axes=0)
            GG += integ * np.tensordot(Gvalues[:, k], Gvalues[:, k], axes=0)

    GGinv = Linalg.invert(GG)
    if fit_nodes is None:
        T = np.dot(GGinv, GF)
        E = FF - np.dot(GF.T, T)
        return totuple(T), totuple(E)
    fit_nodes = tuple(
        Fraction(node) if isinstance(node, int) else node for node in fit_nodes
    )
    F = eval_rational_nodes(
        oldknotvector, oldweights, tuple(fit_nodes), olddegree
    )
    G = eval_rational_nodes(
        newknotvector, newweights, tuple(fit_nodes), newdegree
    )
    F = np.array(F, dtype="object").T
    GT = np.array(G, dtype="object")
    G = np.transpose(GT)
    LL = np.dot(G, np.dot(GGinv, GT))
    LLinv = Linalg.invert(LL)
    LG = np.dot(LLinv, np.dot(G, GGinv))
    QG = GGinv - np.dot(GGinv, np.dot(GT, LG))
    QF = np.dot(GGinv, np.dot(GT, LLinv))
    T = np.dot(QG, GF) + np.dot(QF, F)
    E = (FF - 2 * np.dot(T.T, GF) + np.dot(T.T, np.dot(GG, T))) / 2
    return totuple(T), totuple(E)
