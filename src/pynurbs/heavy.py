"""
This module contains very low level functions that can be easily change to another language such as C/C++ (further may be).
They are 'heavy' functions that are called many times and don't require any special package 
Most of these functions works only with integers, floats and tuples.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Tuple, Union

import numpy as np

from .cmath import IntegratorArray, Linalg, NodeSample, number_type, totuple
from .core.basisfunction import ImmutableBasisFunction
from .core.knotvector import ImmutableKnotVector


def find_roots(
    knotvector: ImmutableKnotVector, ctrlvalues: Tuple[float]
) -> Tuple[float]:
    """
    Finds the roots of given a spline function
    Each subinterval [u_{k}, u_{k+1}] can be interpoled
    by a polynomial of degree p.
    Taking out the case of constant equal

    We do it by sampling
    """
    knotvector = ImmutableKnotVector(knotvector)
    assert isinstance(ctrlvalues, tuple)
    tolerance = 1e-8
    for value in ctrlvalues:
        float(value)
    ctrlvalues = np.array(ctrlvalues, dtype="float64")
    knots = knotvector.knots
    degree = knotvector.degree
    nsample = 100
    nodes0to1 = NodeSample.open_linspace(nsample)
    manynodes = []
    for start, end in zip(knots[:-1], knots[1:]):
        nodes = [start + (end - start) * node for node in nodes0to1]
        manynodes += nodes
    manynodes = tuple(sorted(manynodes + list(knots)))
    matrixeval = eval_spline_nodes(knotvector, manynodes, degree)
    manyvalues = np.dot(np.transpose(matrixeval), ctrlvalues)
    manyvalues = tuple(manyvalues)
    while 0 in manyvalues:
        index = manyvalues.index(0)
        manyvalues.pop(index)
        manynodes.pop(index)
    # return tuple(sorted(manynodes))

    # Bissection algorithm
    lefts = []  # a
    righs = []  # b
    fleft = []  # f(a)
    frigh = []  # f(b)
    maxdist = 0
    for i, (aval, bval) in enumerate(zip(manyvalues[:-1], manyvalues[1:])):
        if aval * bval < 0:
            maxdist = max(maxdist, manynodes[i + 1] - manynodes[i])
            lefts.append(manynodes[i])
            righs.append(manynodes[i + 1])
            fleft.append(aval)
            frigh.append(bval)
    nintervs = len(lefts)
    if nintervs == 0:
        return tuple()
    lefts = np.array(lefts, dtype="float64")
    righs = np.array(righs, dtype="float64")
    fleft = np.array(fleft, dtype="float64")
    frigh = np.array(frigh, dtype="float64")
    niters = 1 + int(np.ceil(np.log2(maxdist / tolerance)))
    for i in range(niters):
        mednodes = (lefts + righs) / 2
        matrixeval = eval_spline_nodes(knotvector, tuple(mednodes), degree)
        medvals = np.dot(np.transpose(matrixeval), ctrlvalues)
        for i, medval in enumerate(medvals):
            if medval == 0:
                lefts[i] = mednodes[i]
                righs[i] = mednodes[i]
                fleft[i] = 0
                frigh[i] = 0
            elif fleft[i] * medval < 0:
                righs[i] = mednodes[i]
                frigh[i] = medval
            else:
                lefts[i] = mednodes[i]
                fleft[i] = medval
    roots = (lefts + righs) / 2
    filtered_roots = []
    for root in roots:
        for filtroot in filtered_roots:
            if abs(root - filtroot) < tolerance:
                break
        else:
            filtered_roots.append(root)
    return tuple(sorted(filtered_roots))


def eval_spline_nodes(
    knotvector: ImmutableKnotVector, nodes: Tuple[float], degree: int
) -> Tuple[Tuple[float]]:
    """
    Returns a matrix M of which M_{ij} = N_{i,degree}(node_j)
    M.shape = (npts, len(nodes))
    """
    knotvector = ImmutableKnotVector(knotvector)
    basis = ImmutableBasisFunction(knotvector, degree)
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


class LeastSquare:
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

    @staticmethod
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
        basis = ImmutableBasisFunction(knotvector)
        matrix = np.transpose(tuple(map(basis, nodes)))
        if weights is not None:
            denominators = 1 / np.dot(weights, matrix)
            matrix = np.einsum("j,ij->ij", denominators, matrix)
        return Linalg.lstsq(np.transpose(matrix))

    @staticmethod
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
        result = LeastSquare.func2func(
            oldknotvector, oldweights, newknotvector, newweights, fit_nodes
        )
        return totuple(result)

    @staticmethod
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
            Fraction(node) if isinstance(node, int) else node for node in oldknotvector
        )
        newknotvector = tuple(
            Fraction(node) if isinstance(node, int) else node for node in newknotvector
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
        F = eval_rational_nodes(oldknotvector, oldweights, tuple(fit_nodes), olddegree)
        G = eval_rational_nodes(newknotvector, newweights, tuple(fit_nodes), newdegree)
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


class Operations:
    """
    Contains algorithms to
    * knot insertion,
    * knot removal,
    * degree increase
    * degree decrease
    """

    def split_curve(knotvector: ImmutableKnotVector, nodes: Tuple[float]):
        """
        Breaks curves in the nodes

        Given a curve A(u) defined in a interval [a, b] and
        associated with control points P, this function breaks
        A(u) into m curves A_{0}, ..., A_{m}, which m is the
        number of nodes.

        # INPUT
            - vector: The knotvector
            - nodes: The places to split the curves

        # OUTPUT
            - matrices: (m+1) transformation matrix

        # Cautions:
            - If the extremities are in nodes, they are ignored
            - Repeted nodes are ignored, [0.5, 0.5] is the same as [0.5]
        """
        knotvector = ImmutableKnotVector(knotvector)
        if not knotvector.valid(nodes):
            msg = f"Invalid nodes {nodes} in knotvector {knotvector}"
            raise ValueError(msg)
        degree = knotvector.degree
        nodes = set(nodes)  # Remove repeted nodes
        nodes -= set([knotvector[0], knotvector[-1]])  # Take out extremities
        nodes = tuple(nodes)
        manynodes = []
        for node in nodes:
            mult = knotvector.mult(node)
            manynodes += [node] * (degree + 1 - mult)
        bigvector = knotvector.insert(manynodes)
        bigmatrix = Operations.knot_insert(knotvector, manynodes)
        newvectors = bigvector.split(nodes)
        matrices = []
        for newvector in newvectors:
            umin = newvector.limits[0]
            span = bigvector.span(umin)
            lowerind = span - degree
            upperind = lowerind + len(newvector) - degree - 1
            newmatrix = bigmatrix[lowerind:upperind]
            matrices.append(newmatrix)
        return matrices

    def one_knot_insert_once(
        knotvector: ImmutableKnotVector, node: float
    ) -> "Matrix2D":
        """
        Given the knotvector and a node to be inserted, this function
        returns a matrix of transformation T of control points

        Let
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_j N_j(u) * Q_j

        This function returns T such
            [Q] = [T] @ [P]
        """
        knotvector = ImmutableKnotVector(knotvector)
        if not knotvector.valid(node):
            msg = f"Invalid nodes {node} in knotvector {knotvector}"
            raise ValueError(msg)

        oldnpts = knotvector.npts
        degree = knotvector.degree
        oldspan = knotvector.span(node)
        oldmult = knotvector.mult(node)
        one = node / node
        matrix = np.zeros((oldnpts + 1, oldnpts), dtype="object")
        for i in range(oldspan - degree + 1):
            matrix[i, i] = one
        for i in range(oldspan - oldmult, oldnpts):
            matrix[i + 1, i] = one
        for i in range(oldspan - degree + 1, oldspan + 1):
            alpha = node - knotvector[i]
            alpha /= knotvector[i + degree] - knotvector[i]
            matrix[i, i] = alpha
            matrix[i, i - 1] = 1 - alpha
        return totuple(matrix)

    def one_knot_insert(
        knotvector: ImmutableKnotVector, node: float, times: int
    ) -> "Matrix2D":
        """
        Given the knotvector and a node to be inserted, this function
        returns a matrix of transformation T of control points

        Let
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_j N_j(u) * Q_j

        This function returns T such
            [Q] = [T] @ [P]
        """
        knotvector = ImmutableKnotVector(knotvector)
        if not knotvector.valid(node):
            msg = f"Invalid node {node} in knotvector {knotvector}"
            raise ValueError(msg)
        if not isinstance(times, int):
            msg = f"Times must be an int, not {times}"
            raise TypeError(msg)
        if times <= 0:
            msg = f"Times must be positive! Received {times}"
            raise ValueError(msg)
        oldnpts = knotvector.npts
        matrix = np.eye(oldnpts, dtype="object")
        for _ in range(times):
            incmatrix = Operations.one_knot_insert_once(knotvector, node)
            matrix = incmatrix @ matrix
            knotvector = knotvector.insert([node])
        return totuple(matrix)

    def knot_insert(knotvector: ImmutableKnotVector, nodes: Tuple[float]) -> "Matrix2D":
        """
        Given the knotvector and a node to be inserted, this function
        returns a matrix of transformation T of control points

        Let
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_j N_j(u) * Q_j

        This function returns T such
            [Q] = [T] @ [P]

        # Caution:
            - Nodes in extremities are not considered
        """

        knotvector = ImmutableKnotVector(knotvector)
        if not knotvector.valid(nodes):
            msg = f"Invalid nodes {nodes} in knotvector {knotvector}"
            raise ValueError(msg)
        nodes = tuple(nodes)
        setnodes = tuple(sorted(set(nodes) - set([knotvector[0], knotvector[-1]])))
        oldnpts = knotvector.npts
        matrix = np.eye(oldnpts, dtype="object")
        if len(nodes) == 0:
            return totuple(matrix)
        for node in setnodes:
            times = nodes.count(node)
            incmatrix = Operations.one_knot_insert(knotvector, node, times)
            matrix = incmatrix @ matrix
            knotvector = knotvector.insert(times * [node])
        return totuple(matrix)

    def knot_remove(knotvector: ImmutableKnotVector, nodes: Tuple[float]) -> "Matrix2D":
        """ """
        knotvector = ImmutableKnotVector(knotvector)
        if not knotvector.valid(nodes):
            msg = f"Invalid nodes {nodes} in knotvector {knotvector}"
            raise ValueError(msg)
        newknotvector = knotvector.remove(nodes)
        matrix, _ = LeastSquare.spline2spline(knotvector, newknotvector)
        return totuple(matrix)

    def degree_increase_bezier_once(knotvector: ImmutableKnotVector) -> "Matrix2D":
        knotvector = ImmutableKnotVector(knotvector)
        one = knotvector[-1] - knotvector[0]
        one /= one
        degree = knotvector.degree
        matrix = np.zeros((degree + 2, degree + 1), dtype="object")
        matrix[0, 0] = one
        for i in range(1, degree + 1):
            alpha = (one * i) / (degree + 1)
            matrix[i, i - 1] = alpha
            matrix[i, i] = one - alpha
        matrix[degree + 1, degree] = one
        return totuple(matrix)

    def degree_increase_bezier(
        knotvector: ImmutableKnotVector, times: int
    ) -> "Matrix2D":
        """
        Given a bezier curve A(u) of degree p, we want a new bezier curve B(u)
        of degree (p+t) such B(u) = A(u) for every u
        Then, this function returns the matrix of transformation T
            [Q] = [T] @ [P]
            A(u) = sum_{i=0}^{p} B_{i,p}(u) * P_i
            B(u) = sum_{i=0}^{p+t} B_{i,p+t}(u) * Q_i
        """
        knotvector = ImmutableKnotVector(knotvector)
        if not isinstance(times, int):
            msg = f"Times must be an int, not {times}"
            raise TypeError(msg)
        if times <= 0:
            msg = f"Times must be positive! Received {times}"
            raise ValueError(msg)
        degree = knotvector.degree
        matrix = np.eye(degree + 1, dtype="object")
        for i in range(times):
            elevateonce = Operations.degree_increase_bezier_once(knotvector)
            matrix = elevateonce @ matrix
            knotvector = knotvector.increase(1)
        return totuple(matrix)

    def degree_increase(knotvector: ImmutableKnotVector, times: int) -> "Matrix2D":
        """
        Given a curve A(u) associated with control points P, we want
        to do a degree elevation
        """
        knotvector = ImmutableKnotVector(knotvector)
        if not isinstance(times, int):
            msg = f"Times must be an int, not {times}"
            raise TypeError(msg)
        if times == 0:
            return totuple(np.eye(knotvector.npts, dtype="object"))
        elif times < 0:
            msg = f"Times must be >= 0! Received {times}"
            raise ValueError(msg)
        degree = knotvector.degree
        npts = knotvector.npts
        if degree + 1 == npts:
            return Operations.degree_increase_bezier(knotvector, times)
        nodes = knotvector.knots
        newvectors = knotvector.split(nodes)
        matrices = Operations.split_curve(knotvector, nodes)

        bigmatrix = []
        for splitedvector, splitedmatrix in zip(newvectors, matrices):
            splitedmatrix = np.array(splitedmatrix)
            elevatedmatrix = Operations.degree_increase_bezier(splitedvector, times)
            newmatrix = elevatedmatrix @ splitedmatrix
            for linemat in newmatrix:
                bigmatrix.append(linemat)
        bigmatrix = np.array(bigmatrix)

        insertednodes = []
        for node in nodes:
            mult = knotvector.mult(node)
            insertednodes += (degree + 1 - mult) * [node]
        bigvector = knotvector.insert(insertednodes)
        incbigvector = bigvector.increase(times)
        removematrix = Operations.knot_remove(incbigvector, insertednodes)

        bigmatrix = np.array(bigmatrix)
        removematrix = np.array(removematrix)
        finalmatrix = removematrix @ bigmatrix
        return totuple(finalmatrix)

    def matrix_transformation(
        knotvectora: ImmutableKnotVector, knotvectorb: ImmutableKnotVector
    ):
        """
        Given two curve A(u) and B(u), associated with controlpoints P and Q
        this function returns the transformation matrix T such
            [P] = [T] @ [Q]
        It's only possible when the knotvectorb is a transformation of knotvectora
        by using knot_insertion and degree_increase

        # Caution
            - We suppose the limits of vectors are the same
            - We suppose degreeB >= degreeA
        """
        knotvectora = ImmutableKnotVector(knotvectora)
        knotvectorb = ImmutableKnotVector(knotvectorb)
        assert knotvectora.limits == knotvectorb.limits

        degreea = knotvectora.degree
        degreeb = knotvectorb.degree
        knotsa = knotvectora.knots
        assert degreea <= degreeb
        matrix_deginc = Operations.degree_increase(knotvectora, degreeb - degreea)
        knotvectora = knotvectora.increase(degreeb - degreea)

        nodes2ins = []
        for knot in knotvectorb.knots:
            times = knotvectorb.mult(knot) - knotvectora.mult(knot)
            nodes2ins += times * [knot]
        matrix_knotins = Operations.knot_insert(knotvectora, nodes2ins)

        finalresult = np.array(matrix_knotins) @ matrix_deginc
        return totuple(finalresult)


class MathOperations:
    @staticmethod
    def mult_nonrat_bezier(
        knotvectora: Tuple[float], knotvectorb: Tuple[float]
    ) -> Tuple[Tuple[float]]:
        """
        Given two bezier curves A(u) and B(u) of degrees p and q,
        we want to find a bezier curve C(u) of degree (p+q) such
            C(u) = A(u) * B(u) forall u
        This function returns [M] of shape (p+1, p+q+1, q+1) such
            [C] = [A] * [M] * [B]
            C_j = sum_{i,k}^{p,q} M_{ijk} A_i B_k
        """
        knotvectora = ImmutableKnotVector(knotvectora)
        knotvectorb = ImmutableKnotVector(knotvectorb)
        assert knotvectora.limits == knotvectorb.limits
        return MathOperations.mul_spline_curve(knotvectora, knotvectorb)

    @staticmethod
    def knotvector_mul(
        knotvectora: Tuple[float], knotvectorb: Tuple[float]
    ) -> Tuple[float]:
        knotvectora = ImmutableKnotVector(knotvectora)
        knotvectorb = ImmutableKnotVector(knotvectorb)
        assert knotvectora.limits == knotvectorb.limits

        degreea = knotvectora.degree
        degreeb = knotvectorb.degree
        allknots = tuple(sorted(set(knotvectora) | set(knotvectorb)))
        classes = [0] * len(allknots)
        for i, knot in enumerate(allknots):
            multa = knotvectora.mult(knot)
            multb = knotvectorb.mult(knot)
            classes[i] = min(degreea - multa, degreeb - multb)
        degreec = degreea + degreeb
        knotvectorc = [knotvectora[0]] * (degreec + 1)
        for knot, classe in zip(allknots[1:-1], classes):
            knotvectorc += [knot] * (degreec - classe)
        knotvectorc += [knotvectora[-1]] * (degreec + 1)
        return ImmutableKnotVector(knotvectorc)

    @staticmethod
    def add_spline_curve(
        knotvectora: Tuple[float], knotvectorb: Tuple[float]
    ) -> Tuple["Matrix2D"]:
        """
        Given two spline curves, A(u) and B(u), such
            A(u) = sum_{i=0}^{n} N_i(u) * P_i
            B(u) = sum_{j=0}^{m} N_j(u) * Q_j
        It's wantted the curve C(u) such
            C(u) = A(u) + B(u) forall u
        It means, computing the new knotvector and newcontrol points
            C(u) = sum_{k=0}^{k} N_{k} * R_k
        But instead, we return matrix [Ma] and [Mb] such
            [R] = [Ma] * [P] + [Mb] * [Q]

        # INPUT
            - knotvectora: The knotvector of curve A
            - knotvectorb: The knotvector of curve B

        # OUTPUT
            - matrixa: Matrix of transformation of points A
            - matrixb: Matrix of transformation of points B

        # Caution:
            - We suppose the knotvectora and knotvectorb limits are equal
            - We suppose the curves has same degree
        """
        knotvectora = ImmutableKnotVector(knotvectora)
        knotvectorb = ImmutableKnotVector(knotvectorb)
        assert knotvectora.limits == knotvectorb.limits

        knotvectorc = knotvectora | knotvectorb
        matrixa = Operations.matrix_transformation(knotvectora, knotvectorc)
        matrixb = Operations.matrix_transformation(knotvectorb, knotvectorc)
        return totuple(matrixa), totuple(matrixb)

    @staticmethod
    def mul_spline_curve(
        knotvectora: Tuple[float], knotvectorb: Tuple[float]
    ) -> Tuple["Matrix3D"]:
        """
        Given two spline curves, called A(u) and B(u), it computes and returns
        a new curve C(u) such C(u) = A(u) * B(u) for every u
        Restrictions: The limits of B(u) must be the same as the limits of A(u)
        The parameter `simplify` shows if the function try to reduce at maximum
        the degree and the knots inside.

        The matrix is such
            [C] = [A] @ [M] @ [B]
            C_j = sum_{i, k} A_i * M_{ijk} * B_k

        """
        knotvectora = ImmutableKnotVector(knotvectora)
        knotvectorb = ImmutableKnotVector(knotvectorb)
        assert knotvectora.limits == knotvectorb.limits

        knotvectorc = MathOperations.knotvector_mul(knotvectora, knotvectorb)
        degreea = knotvectora.degree
        degreeb = knotvectorb.degree
        degreec = knotvectorc.degree
        nptsa = knotvectora.npts
        nptsb = knotvectorb.npts
        nptsc = knotvectorc.npts
        allknots = knotvectorc.knots

        nptseval = 2 * (degreec + 1)
        nptstotal = nptseval * (len(allknots) - 1)
        allevalnodes = np.empty(nptstotal, dtype="object")
        nodes0to1 = NodeSample.closed_linspace(nptseval)
        for i in range(len(allknots) - 1):
            start, end = allknots[i : i + 2]
            nodes = tuple(start + (end - start) * node for node in nodes0to1)
            allevalnodes[i * nptseval : (i + 1) * nptseval] = nodes
        allevalnodes = tuple(allevalnodes)

        avals = eval_spline_nodes(knotvectora, allevalnodes, degreea)
        bvals = eval_spline_nodes(knotvectorb, allevalnodes, degreeb)
        cvals = eval_spline_nodes(knotvectorc, allevalnodes, degreec)
        avals = np.array(avals)
        bvals = np.array(bvals)
        cvals = np.array(cvals)

        lstsqmat = Linalg.lstsq(np.transpose(cvals))

        matrix3d = np.empty((nptsa, nptsc, nptsb), dtype="object")
        for i, linei in enumerate(avals):
            for j, linej in enumerate(lstsqmat):
                for k, linek in enumerate(bvals):
                    matrix3d[i, j, k] = np.sum(linei * linej * linek)
        return totuple(matrix3d)


class Calculus:
    @staticmethod
    def difference_vector(knotvector: ImmutableKnotVector) -> Tuple[float]:
        knotvector = ImmutableKnotVector(knotvector)
        degree = knotvector.degree
        assert degree > 0
        npts = knotvector.npts
        avals = np.zeros(npts, dtype="float64")
        for i in range(npts):
            diff = knotvector[i + degree] - knotvector[i]
            if diff != 0:
                avals[i] = degree / diff
        return totuple(avals)

    @staticmethod
    def difference_matrix(knotvector: ImmutableKnotVector) -> np.ndarray:
        knotvector = ImmutableKnotVector(knotvector)

        avals = Calculus.difference_vector(knotvector)
        npts = len(avals)
        matrix = np.diag(avals)
        for i in range(npts - 1):
            matrix[i, i + 1] = -avals[i + 1]
        return totuple(matrix)

    @staticmethod
    def derivate_nonrational_bezier(
        knotvector: ImmutableKnotVector, reduce: bool = True
    ) -> Tuple[Tuple[float]]:
        """
        Given a nonrational bezier C(u) of degree p, this function returns matrix [M] such
            [Q] = [M] * [P]
            C(u) = sum_{i=0}^p B_{i,p}(u) * P_i
            C'(u) = sum_{i=0}^q B_{i,q}(u) * Q_i
        The matrix size if (q+1, p+1)

        Normally q = p-1, since it decreases the degree.
        If reduce is False, it does a degree elevation and keeps the same degree
        """
        knotvector = ImmutableKnotVector(knotvector)
        degree = knotvector.degree
        assert degree > 0
        matrix = np.zeros((degree, degree + 1), dtype="object")
        for i in range(degree):
            matrix[i, i] = -degree
            matrix[i, i + 1] = degree
        matrix /= knotvector[-1] - knotvector[0]
        if reduce:
            return totuple(matrix)
        elevate = Operations.degree_increase_bezier_once(knotvector[1:-1])
        return totuple(np.dot(elevate, matrix))

    @staticmethod
    def derivate_nonrational_spline(
        knotvector: ImmutableKnotVector,
    ) -> Tuple[Tuple[float]]:
        """
        Given a spline C(u) of degree p, this function returns matrix [M] such
            [Q] = [M] * [P]
            C(u) = sum_{i=0}^{n} N_{i,p}(u) * P_i
            C'(u) = sum_{i=0}^{m} N_{i,q-1}(u) * Q_i
        The matrix size if (m, n)

        Normally q = p-1, since it decreases the degree.
        If reduce is False, it does a degree elevation and keeps the same degree
        """
        knotvector = ImmutableKnotVector(knotvector)
        matrix = Calculus.difference_matrix(knotvector)
        matrix = np.transpose(matrix)[1:]
        return totuple(matrix)

    @staticmethod
    def derivate_rational_bezier(
        knotvector: ImmutableKnotVector,
    ) -> Tuple[Tuple[float]]:
        """
        Does'nt work yet

        Given a rational bezier C(u) of degree p, control points P_i and weights w_i,
        this function returns matrix [M] and [K] such

            [D] = [P/w] * [M] * [w]
            [z] = [w] * [K] * [w]
            [M].shape = (p+1, 2p+1, p+1)
            [z].shape = (p+1, 2p+1, p+1)

            C(u) = A(u)/w(u)
            A(u) = sum_i B_{i,p}(u) * (w_i * P_i)
                 = sum_i B_{i,p}(u) A_i
            w(u) = sum_i B_{i,p}(u) * w_i

            C'(u) = (A'(u) * w(u) - A(u) * w'(u))/(w(u)^2)
            C'(u) = (sum_{i=0}^{2p} B_{i,2p} * D_i)/(sum_{i=0}^{2p} B_{i,2p} * z_i)
        """
        knotvector = ImmutableKnotVector(knotvector)
        matrixmult = MathOperations.mult_nonrat_bezier(knotvector, knotvector)
        # matrixderi = Calculus.derivate_nonrational_bezier(knotvector, False)
        matrixderi = Calculus.derivate_nonrational_bezier(knotvector)
        elevate = Operations.degree_increase_bezier_once(knotvector[1:-1])
        matrixderi = np.dot(elevate, matrixderi)
        matrixleft = np.tensordot(np.transpose(matrixderi), matrixmult, axes=1)
        matrixrigh = matrixmult @ matrixderi
        return totuple(matrixleft - matrixrigh), totuple(matrixmult)
