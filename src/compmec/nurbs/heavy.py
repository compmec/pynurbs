from functools import cache
from typing import Tuple

import numpy as np


def N(i: int, j: int, k: int, u: float, U: Tuple[float]) -> float:
    """
    Returns the value of N_{ij}(u) in the interval [u_{k}, u_{k+1}]
    Remember that N_{i, j}(u) = 0   if  ( u not in [U[i], U[i+j+1]] )
    """
    knots = KnotVector.find_knots(U)
    spans = [KnotVector.find_span(knot, U) for knot in knots]
    z = spans.index(k)
    coefs = get_coefs(U, i, j, z)
    if np.all(coefs == 0):
        return 0
    q = (u - knots[z]) / (knots[z + 1] - knots[z])
    # Horner's method
    return BasisFunction.horner_method(coefs, q)


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


def get_coefs(knotvector: Tuple[float], i: int, j: int, z: int):
    knots = KnotVector.find_knots(knotvector)
    spans = [KnotVector.find_span(knot, knotvector) for knot in knots]
    y = i + j - spans[z]
    if y < 0 or j < y:
        return np.zeros((j + 1), dtype="float64")
    return BasisFunction.speval_matrix(knotvector, j)[z, y]


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
        oldknotvect: Tuple[float], newknotvect: Tuple[float]
    ) -> Tuple[np.ndarray]:
        """ """
        oldknotvect = tuple(oldknotvect)
        newknotvect = tuple(newknotvect)
        olddegree = KnotVector.find_degree(oldknotvect)
        newdegree = KnotVector.find_degree(newknotvect)
        oldnpts = KnotVector.find_npts(oldknotvect)
        newnpts = KnotVector.find_npts(newknotvect)
        oldknots = KnotVector.find_knots(oldknotvect)
        newknots = KnotVector.find_knots(newknotvect)
        allknots = list(set(oldknots + newknots))
        allknots.sort()
        nptsinteg = olddegree + newdegree + 3  # Number integration points
        integrator = LeastSquare.integrator_array(nptsinteg)

        # Here we will store the values of N and M
        # for a certain interval [a, b], which will change
        Nvalues = np.empty((oldnpts, nptsinteg), dtype="float64")
        Mvalues = np.empty((newnpts, nptsinteg), dtype="float64")

        A = np.zeros((oldnpts, oldnpts), dtype="float64")  # N*N
        B = np.zeros((oldnpts, newnpts), dtype="float64")  # N*M
        C = np.zeros((newnpts, newnpts), dtype="float64")  # M*M
        for a, b in zip(allknots[:-1], allknots[1:]):
            chebynodes = LeastSquare.chebyshev_nodes(nptsinteg, a, b)
            # Integral of the functions in the interval [a, b]
            k0 = KnotVector.find_span(a, oldknotvect)
            k1 = KnotVector.find_span(a, newknotvect)
            Nvalues.fill(0)
            Mvalues.fill(0)
            for y in range(olddegree + 1):
                i = y + k0 - olddegree
                for j, uj in enumerate(chebynodes):
                    Nvalues[i, j] = N(i, olddegree, k0, uj, oldknotvect)
            for y in range(newdegree + 1):
                i = y + k1 - newdegree
                for j, uj in enumerate(chebynodes):
                    Mvalues[i, j] = N(i, newdegree, k1, uj, newknotvect)
            A += np.einsum("k,ik,jk->ij", integrator, Nvalues, Nvalues)
            B += np.einsum("k,ik,jk->ij", integrator, Nvalues, Mvalues)
            C += np.einsum("k,ik,jk->ij", integrator, Mvalues, Mvalues)

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
        matrix = np.zeros((ninter, j + 1, j + 1), dtype="float64")
        if j == 0:
            matrix.fill(1)
            return matrix
        matrix_less1 = BasisFunction.speval_matrix(knotvector, j - 1)
        for y in range(j):
            for z, sz in enumerate(spans[:-1]):
                i = y + sz - j + 1
                denom = knotvector[i + j] - knotvector[i]
                if not denom:
                    continue
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
