from functools import cache
from typing import Tuple

import numpy as np


def binom(n: int, i: int):
    """
    Returns binomial (n, i)
    if n
    """
    prod = 1
    if i <= 0 or i >= n:
        return 1
    for j in range(i):
        prod *= (n - j) / (i - j)
    return int(prod)


def eval_spline_nodes(knotvector: Tuple[float], nodes: Tuple[float]):
    """
    Returns a matrix M of which M_{ij} = N_{i,p}(node_j)
    """
    degree = KnotVector.find_degree(knotvector)
    npts = KnotVector.find_npts(knotvector)
    knots = KnotVector.find_knots(knotvector)
    spans = [KnotVector.find_span(knot, knotvector) for knot in knots]
    result = np.zeros((npts, len(nodes)), dtype="float64")
    matrix3d = BasisFunction.speval_matrix(knotvector, degree)
    for j, node in enumerate(nodes):
        span = KnotVector.find_span(node, knotvector)
        ind = spans.index(span)
        shifnode = node - knots[ind]
        shifnode /= knots[ind + 1] - knots[ind]
        for y in range(degree + 1):
            i = y + span - degree
            coefs = matrix3d[ind, y]
            value = BasisFunction.horner_method(coefs, shifnode)
            result[i, j] = value
    return result


def eval_rational_nodes(
    knotvector: Tuple[float], weights: Tuple[float], nodes: Tuple[float]
):
    """
    Returns a matrix M of which M_{ij} = N_{i,p}(node_j)
    """
    rationalvals = eval_spline_nodes(knotvector, nodes)
    for j, node in enumerate(nodes):
        denom = np.inner(rationalvals[:, j], weights)
        for i, weight in enumerate(weights):
            rationalvals[i, j] *= weight / denom
    return rationalvals


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
    def chebyshev_nodes(npts: int, a: float = 0, b: float = 1) -> Tuple[float]:
        """
        Returns a list of floats which are spaced in [a, b]
        for integration purpose.
        """
        k = np.arange(0, npts)
        nodes = np.cos(np.pi * (2 * k + 1) / (2 * npts))
        nodes *= (b - a) / 2
        nodes += (a + b) / 2
        nodes.sort()
        return nodes

    @staticmethod
    def interpolator_matrix(npts: int) -> Tuple[float]:
        """
        This function helps computing the values of F_{i} such
            f(x) approx g(x) = sum_{i=0}^{z} F_{i} * B_{i, z}(x)
        Which B_{i, z}(x) is a bezier function
            B_{i, z}(x) = binom(z, i) * (1-x)^{i} * x^{z-i}
        Then, a linear system is obtained by setting f(x_j) = g(x_j)
        at (z+1) points, which are the chebyshev nodes:
            [B0(x0)   B1(x0)   ...   Bz(x0)][ F0 ]   [ f(x0) ]
            [B0(x1)   B1(x1)   ...   Bz(x1)][ F1 ]   [ f(x1) ]
            [B0(x2)   B1(x2)   ...   Bz(x2)][ F2 ] = [ f(x2) ]
            [  |        |       |      |   ][ |  ]   [   |   ]
            [B0(xz)   B1(xz)   ...   Bz(xz)][ Fz ]   [ f(xz) ]
        The return is the inverse of the matrix
        """
        z = npts - 1
        chebyshev0to1 = LeastSquare.chebyshev_nodes(npts)
        matrixbezier = np.zeros((npts, npts), dtype="float64")
        for i, xi in enumerate(chebyshev0to1):
            for j in range(z + 1):
                matrixbezier[i, j] = binom(z, j) * (1 - xi) ** (z - j) * (xi**j)
        return np.linalg.inv(matrixbezier)

    @staticmethod
    def integrator_array(npts: int) -> Tuple[float]:
        """
        This function helps computing the integral of f(x) at the
        interval [0, 1] by using a polynomial of degree ```npts```.
        It returns an 1D array [A] such
            int_{0}^{1} f(x) dx = [A] * [f]
        Which
            [f] = [f(x0)  f(x1)  ...  f(xz)]
        The points x0, x1, ... are the chebyshev nodes

        We suppose that
            int_{0}^{1} f(x) dx approx int_{0}^{1} g(x) dx
        Which g(x) is made by bezier curves
            g(x) = sum_{i=0}^{z} F_{i} * B_{i, z}(x)
            B_{i, z}(x) = binom(z, i) * (1-x)^{i} * x^{z-i}
        We see that
            int_{0}^{1} B_{i, z}(x) dx = 1/(z+1)
        Therefore
            int_{0}^{1} f(x) dx approx 1/(z+1) * sum F_{i}
        The values of F_{i} are gotten from ```interpolator_matrix```
        """
        matrix = LeastSquare.interpolator_matrix(npts)
        return np.einsum("ij->j", matrix) / npts

    @staticmethod
    def spline2spline(
        oldvector: Tuple[float], newvector: Tuple[float]
    ) -> Tuple["Matrix2D"]:
        oldnpts = KnotVector.find_npts(oldvector)
        newnpts = KnotVector.find_npts(newvector)
        return LeastSquare.func2func(oldvector, [1] * oldnpts, newvector, [1] * newnpts)

    @staticmethod
    def func2func(
        oldvector: Tuple[float],
        oldweights: Tuple[float],
        newvector: Tuple[float],
        newweights: Tuple[float],
    ) -> Tuple[np.ndarray]:
        """ """
        oldvector = tuple(oldvector)
        olddegree = KnotVector.find_degree(oldvector)
        oldnpts = KnotVector.find_npts(oldvector)
        oldknots = KnotVector.find_knots(oldvector)

        newvector = tuple(newvector)
        newdegree = KnotVector.find_degree(newvector)
        newnpts = KnotVector.find_npts(newvector)
        newknots = KnotVector.find_knots(newvector)

        allknots = list(set(oldknots + newknots))
        allknots.sort()
        nptsinteg = olddegree + newdegree + 3  # Number integration points
        integrator = LeastSquare.integrator_array(nptsinteg)

        A = np.zeros((oldnpts, oldnpts), dtype="float64")  # F*F
        B = np.zeros((oldnpts, newnpts), dtype="float64")  # F*G
        C = np.zeros((newnpts, newnpts), dtype="float64")  # G*G
        chebynodes0to1 = LeastSquare.chebyshev_nodes(nptsinteg)
        for start, end in zip(allknots[:-1], allknots[1:]):
            chebynodes = start + (end - start) * chebynodes0to1
            # Integral of the functions in the interval [a, b]
            Fvalues = eval_rational_nodes(oldvector, oldweights, chebynodes)
            Gvalues = eval_rational_nodes(newvector, newweights, chebynodes)
            A += np.einsum("k,ik,jk->ij", integrator, Fvalues, Fvalues)
            B += np.einsum("k,ik,jk->ij", integrator, Fvalues, Gvalues)
            C += np.einsum("k,ik,jk->ij", integrator, Gvalues, Gvalues)

        T = np.linalg.solve(C, B.T)
        E = A - B @ T
        return T, E


class KnotVector:
    @staticmethod
    def is_valid_vector(knotvector: Tuple[float]) -> bool:
        lenght = len(knotvector)
        for i in range(lenght - 1):
            if knotvector[i] > knotvector[i + 1]:
                return False
        knots = []
        mults = []
        for knot in knotvector:
            if knot not in knots:
                knots.append(knot)
                mults.append(0)
            ind = knots.index(knot)
            mults[ind] += 1
        if mults[0] != mults[-1]:
            return False
        for mult in mults:
            if mult > mults[0]:
                return False
        return True

    @staticmethod
    def find_degree(knotvector: Tuple[float]) -> int:
        degree = 0
        while knotvector[degree] == knotvector[degree + 1]:
            degree += 1
        return degree

    @staticmethod
    def find_npts(knotvector: Tuple[float]) -> int:
        degree = KnotVector.find_degree(knotvector)
        return len(knotvector) - degree - 1

    @staticmethod
    def find_span(node: float, knotvector: Tuple[float]) -> int:
        n = KnotVector.find_npts(knotvector) - 1
        degree = KnotVector.find_degree(knotvector)
        if node == knotvector[n + 1]:  # Special case
            return n
        low, high = degree, n + 1  # Do binary search
        mid = (low + high) // 2
        while True:
            if node < knotvector[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
            if knotvector[mid] <= node < knotvector[mid + 1]:
                return mid

    @staticmethod
    def find_mult(node: float, knotvector: Tuple[float]) -> int:
        """
        #### Algorithm A2.1
            Returns how many times a knot is in knotvector
        #### Input:
            ``knotvector``: Tuple[float] -- knot vector
        #### Output:
            ``m``: int -- Multiplicity of the knot
        """
        mult = 0
        for knot in knotvector:
            if abs(knot - node) < 1e-9:
                mult += 1
        return int(mult)

    @staticmethod
    def find_knots(knotvector: Tuple[float]) -> Tuple[float]:
        """
        ####
            Returns a tuple with non repeted knots
        #### Input:
            ``npts``: int -- number of DOFs
            ``degree``: int -- degree
            ``u``: float -- knot value
            ``U``: Tuple[float] -- knot vector
        #### Output:
            ``s``: int -- Multiplicity of the knot
        """
        knots = []
        for knot in knotvector:
            if knot not in knots:
                knots.append(knot)
        return knots

    @staticmethod
    def insert_knots(knotvector: Tuple[float], addknots: Tuple[float]) -> Tuple[float]:
        """
        Returns a new knotvector which contains the previous and new knots.
        This function don't do the validation
        """
        newknotvector = list(knotvector) + list(addknots)
        newknotvector.sort()
        return tuple(newknotvector)

    @staticmethod
    def remove_knots(knotvector: Tuple[float], subknots: Tuple[float]) -> Tuple[float]:
        newknotvector = list(knotvector)
        for knot in subknots:
            newknotvector.remove(knot)
        return tuple(newknotvector)

    @staticmethod
    def unite_vectors(vector0: Tuple[float], vector1: Tuple[float]) -> Tuple[float]:
        all_knots = list(set(vector0) | set(vector1))
        all_mults = [0] * len(all_knots)
        for vector in [vector0, vector1]:
            for knot in vector:
                index = all_knots.index(knot)
                mult = KnotVector.find_mult(knot, vector)
                if mult > all_mults[index]:
                    all_mults[index] = mult
        final_vector = []
        for knot, mult in zip(all_knots, all_mults):
            final_vector += [knot] * mult
        final_vector.sort()
        return tuple(final_vector)

    @staticmethod
    def intersect_vectors(vector0: Tuple[float], vector1: Tuple[float]) -> Tuple[float]:
        all_knots = list(set(vector0) & set(vector1))
        all_mults = [9999] * len(all_knots)
        for vector in [vector0, vector1]:
            for knot in vector:
                if knot not in all_knots:
                    continue
                index = all_knots.index(knot)
                mult = KnotVector.find_mult(knot, vector)
                if mult < all_mults[index]:
                    all_mults[index] = mult
        final_vector = []
        for knot, mult in zip(all_knots, all_mults):
            final_vector += [knot] * mult
        final_vector.sort()
        return tuple(final_vector)

    @staticmethod
    def split(knotvector: Tuple[float], nodes: Tuple[float]) -> Tuple[Tuple[float]]:
        """
        It splits the knotvector at nodes.
        You may put initial and final values.
        Example:
            >> U = [0, 0, 0.5, 1, 1]
            >> split(U, [0.5])
            [[0, 0, 0.5, 0.5],
             [0.5, 0.5, 1, 1]]
            >> split(U, [0.25])
            [[0, 0, 0.25, 0.25],
             [0.25, 0.25, 0.5, 1, 1]]
            >> split(U, [0.25, 0.75])
            [[0, 0, 0.25, 0.25],
             [0.25, 0.25, 0.5, 0.75, 0.75],
             [0.75, 0.75, 1, 1]]
        """
        retorno = []
        knotvector = np.array(knotvector)
        minu, maxu = min(knotvector), max(knotvector)
        nodes = list(nodes) + [minu, maxu]
        nodes = list(set(nodes))
        nodes.sort()
        degree = np.sum(knotvector == minu) - 1
        for a, b in zip(nodes[:-1], nodes[1:]):
            middle = list(knotvector[(a < knotvector) * (knotvector < b)])
            newknotvect = (degree + 1) * [a] + middle + (degree + 1) * [b]
            retorno.append(newknotvect)
        return tuple(retorno)


class BasisFunction:
    @staticmethod
    def horner_method(coefs: Tuple[float], value: float) -> float:
        soma = 0
        for ck in coefs[::-1]:
            soma *= value
            soma += ck
        return soma

    @staticmethod
    def speval_matrix(knotvector: Tuple[float], reqdegree: int):
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
        knots = KnotVector.find_knots(knotvector)
        spans = np.zeros(len(knots), dtype="int16")
        for i, knot in enumerate(knots):
            spans[i] = KnotVector.find_span(knot, knotvector)
        spans = tuple(spans)
        j = reqdegree

        ninter = len(knots) - 1
        matrix = [[[0 * knots[0]] * (j + 1)] * (j + 1)] * ninter
        matrix = np.array(matrix, dtype="object")
        if j == 0:
            matrix.fill(1)
            return matrix
        matrix_less1 = BasisFunction.speval_matrix(knotvector, j - 1)
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
        return matrix


class Operations:
    """
    Contains algorithms to
    * knot insertion,
    * knot removal,
    * degree increase
    * degree decrease
    """

    def split_curve(vector: Tuple[float], nodes: Tuple[float]):
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
        degree = KnotVector.find_degree(vector)
        nodes = set(nodes)  # Remove repeted nodes
        nodes -= set([vector[0], vector[-1]])  # Take out extremities
        nodes = tuple(nodes)
        manynodes = []
        for node in nodes:
            mult = KnotVector.find_mult(node, vector)
            manynodes += [node] * (degree + 1 - mult)
        bigvector = KnotVector.insert_knots(vector, manynodes)
        bigmatrix = Operations.knot_insert(vector, manynodes)
        newvectors = KnotVector.split(vector, nodes)
        matrices = []
        for newvector in newvectors:
            span = KnotVector.find_span(newvector[0], bigvector)
            lowerind = span - degree
            upperind = lowerind + len(newvector) - degree - 1
            newmatrix = bigmatrix[lowerind:upperind]
            matrices.append(newmatrix)
        return matrices

    def knot_insert_once(vector: Tuple[float], node: float) -> "Matrix2D":
        """
        Given the knotvector and a node to be inserted, this function
        returns a matrix of transformation T of control points

        Let
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_j N_j(u) * Q_j

        This function returns T such
            [Q] = [T] @ [P]
        """
        times = 1
        oldnpts = KnotVector.find_npts(vector)
        degree = KnotVector.find_degree(vector)
        oldspan = KnotVector.find_span(node, vector)
        oldmult = KnotVector.find_mult(node, vector)
        one = node / node
        newnpts = oldnpts + times
        matrix = np.zeros((newnpts, oldnpts), dtype="object")
        for i in range(oldspan - degree + 1):
            matrix[i, i] = one
        for i in range(oldspan - oldmult, oldnpts):
            matrix[i + times, i] = one
        for i in range(oldspan - degree + 1, oldspan + 1):
            alpha = node - vector[i]
            alpha /= vector[i + degree] - vector[i]
            matrix[i, i] = alpha
            matrix[i, i - 1] = 1 - alpha

        matrix = matrix.tolist()
        for i, line in enumerate(matrix):
            matrix[i] = tuple(line)
        matrix = tuple(matrix)
        return matrix

    def knot_insert(vector: Tuple[float], nodes: Tuple[float]) -> "Matrix2D":
        """
        Given the knotvector and a node to be inserted, this function
        returns a matrix of transformation T of control points

        Let
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_j N_j(u) * Q_j

        This function returns T such
            [Q] = [T] @ [P]
        """
        nodes = tuple(nodes)
        setnodes = tuple(sorted(set(nodes)))
        oldnpts = KnotVector.find_npts(vector)
        newnpts = oldnpts + len(nodes)
        matrix = np.eye(oldnpts, dtype="object")
        for node in setnodes:
            times = nodes.count(node)
            for i in range(times):
                incT = Operations.knot_insert_once(vector, node)
                matrix = incT @ matrix
                vector = KnotVector.insert_knots(vector, [node])

        matrix = matrix.tolist()
        for i, line in enumerate(matrix):
            matrix[i] = tuple(matrix[i])
        matrix = tuple(matrix)
        return matrix

    def knot_remove(vector: Tuple[float], nodes: Tuple[float]) -> "Matrix2D":
        """ """
        nodes = tuple(nodes)
        newvector = KnotVector.remove_knots(vector, nodes)
        matrix, _ = LeastSquare.spline2spline(vector, newvector)

        matrix = matrix.tolist()
        for i, line in enumerate(matrix):
            matrix[i] = tuple(line)
        matrix = tuple(matrix)
        return matrix

    def degree_increase_bezier(vector: Tuple[float], times: int) -> "Matrix2D":
        """
        Given a bezier curve A(u) of degree p, we want a new bezier curve B(u)
        of degree (p+t) such B(u) = A(u) for every u
        Then, this function returns the matrix of transformation T
            [Q] = [T] @ [P]
            A(u) = sum_{i=0}^{p} B_{i,p}(u) * P_i
            B(u) = sum_{i=0}^{p+t} B_{i,p+t}(u) * Q_i
        """
        degree = KnotVector.find_degree(vector)
        if times > 1:
            matrix = np.eye(degree + 1, dtype="object")
            for i in range(times):
                elevateonce = Operations.degree_increase_bezier(vector, 1)
                matrix = elevateonce @ matrix
                vector = KnotVector.insert_knots(vector, [vector[0], vector[-1]])
            return matrix
        matrix = np.zeros((degree + 2, degree + 1), dtype="object")
        matrix[0, 0] = 1
        for i in range(1, degree + 1):
            alpha = i / (degree + 1)
            matrix[i, i - 1] = alpha
            matrix[i, i] = 1 - alpha
        matrix[degree + 1, degree] = 1

        matrix = matrix.tolist()
        for i, line in enumerate(matrix):
            matrix[i] = tuple(line)
        matrix = tuple(matrix)
        return matrix

    def degree_increase(vector: Tuple[float], times: int) -> "Matrix2D":
        """
        Given a curve A(u) associated with control points P, we want
        to do a degree elevation
        """
        degree = KnotVector.find_degree(vector)
        nodes = KnotVector.find_knots(vector)
        newvectors = KnotVector.split(vector, nodes)
        matrices = Operations.split_curve(vector, nodes)

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
            mult = KnotVector.find_mult(node, vector)
            insertednodes += (degree + 1 - mult) * [node]
        bigvector = KnotVector.insert_knots(vector, insertednodes)
        incbigvector = KnotVector.insert_knots(bigvector, times * nodes)
        removematrix = Operations.knot_remove(incbigvector, insertednodes)

        bigmatrix = np.array(bigmatrix)
        removematrix = np.array(removematrix)
        finalmatrix = removematrix @ bigmatrix

        finalmatrix = finalmatrix.tolist()
        for i, line in enumerate(finalmatrix):
            finalmatrix[i] = tuple(line)
        finalmatrix = tuple(finalmatrix)
        return finalmatrix

    def matrix_transformation(vectora: Tuple[float], vectorb: Tuple[float]):
        """
        Given two curve A(u) and B(u), associated with controlpoints P and Q
        this function returns the transformation matrix T such
            [P] = [T] @ [Q]
        It's only possible when the vectorB is a transformation of vectorA
        by using knot_insertion and degree_increase

        # Caution
            - We suppose the limits of vectors are the same
            - We suppose degreeB >= degreeA
        """
        pass


class MathOperations:
    @staticmethod
    def sum_spline_curve(
        vectora: Tuple[float], vectorb: Tuple[float]
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
            - vectora: The knotvector of curve A
            - vectorb: The knotvector of curve B

        # OUTPUT
            - matrixa: Matrix of transformation of points A
            - matrixb: Matrix of transformation of points B

        # Caution:
            - We suppose the vectora and vectorb limits are equal
            - We suppose the curves has same degree
        """
        vectorc = KnotVector.unite_vectors(vectora, vectorb)
