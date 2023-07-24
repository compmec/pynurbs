import math
from fractions import Fraction
from typing import Tuple, Union

import numpy as np


def number_type(number: Union[int, float, Fraction]):
    """
    Returns the type of a number, if it's a integer, a float, fraction
    It accepts tuple, lists and so on such:
        [int, int, int] -> int
        [int, Fraction, int] -> Fraction
        [int, int, float] -> float
        [Fraction, float, int] -> float
    """
    try:
        iter(number)
        tipos = []
        for numb in number:
            tipo = number_type(numb)
            if tipo is float:
                return float
            tipos.append(tipo)
        for tipo in tipos:
            if tipo is Fraction:
                return Fraction
        return int
    except TypeError:
        if isinstance(number, (int, np.integer)):
            return int
        if isinstance(number, (float, np.floating)):
            return float
        if isinstance(number, Fraction):
            return Fraction
        return float


def invert_integer_matrix(
    matrix: Tuple[Tuple[int]],
) -> Tuple[Tuple[int], Tuple[Tuple[int]]]:
    """
    Given a matrix with integer entries, this function computes the
    inverse of this matrix, returning a matrix of integers and their diagonal
        inverse @ matrix = diag(diagonal)
    """
    side = len(matrix)
    inverse = np.eye(side, dtype="int64")
    matrix = np.column_stack((matrix, inverse))
    for k in range(side):
        # Search biggest pivo
        absline = np.abs(matrix[k:, k])
        biggest = np.max(absline)
        index = k + np.where(absline == biggest)[0][0]
        # Switch lines
        copyline = np.copy(matrix[k, :])
        matrix[k, :] = matrix[index, :]
        matrix[index, :] = copyline[:]
        # Gaussian elimination
        for i in range(k + 1, side):
            matrix[i] = matrix[i] * matrix[k, k] - matrix[k] * matrix[i, k]
            gdcline = math.gcd(*matrix[i])
            if gdcline != 1:
                matrix[i] = matrix[i] / gdcline
    for k in range(side - 1, 0, -1):
        for i in range(k - 1, -1, -1):
            matrix[i] = matrix[i] * matrix[k, k] - matrix[k] * matrix[i, k]
            gdcline = math.gcd(*matrix[i])
            if gdcline != 1:
                matrix[i] = matrix[i] / gdcline
    diagonal = np.diag(matrix[:, :side])
    inverse = matrix[:, side:]
    return totuple(diagonal), totuple(inverse)


def matrix_fraction2integer(
    matrix: Tuple[Tuple[Fraction]],
) -> Tuple[Tuple[int], Tuple[Tuple[int]]]:
    matrix = np.array(matrix, dtype="object")
    side = len(matrix)
    diagonal = [1] * side
    for i, line in enumerate(matrix):
        denoms = [frac.denominator for frac in line]
        diagonal[i] = math.lcm(*denoms)
        matrix[i] = tuple([int(val * diagonal[i]) for val in matrix[i]])
    intdiag = [int(diagi) for diagi in diagonal]
    return tuple(intdiag), tuple(matrix)


def invert_fraction_matrix(matrix: Tuple[Tuple[Fraction]]) -> Tuple[Tuple[Fraction]]:
    """
    C
    """
    numbtype = number_type(matrix)
    assert numbtype is not float
    assert len(matrix) == len(matrix[0])  # Square matrix
    if numbtype is Fraction:
        updiag, matrix = matrix_fraction2integer(matrix)
    else:
        updiag = [1] * len(matrix)
    updiag = np.array(updiag, dtype="int64")
    integermatrix = np.array(matrix, dtype="int64")
    lowdiag, inverseint = invert_integer_matrix(integermatrix)
    inverse = np.array(inverseint, dtype="object")
    for i, line in enumerate(inverseint):
        for j, elem in enumerate(line):
            inverse[i, j] = Fraction(elem * updiag[j], lowdiag[i])
    return inverse


def totuple(array):
    """
    Convert recursively an array to tuples
    """
    try:
        iter(array[0])
        newtuple = []
        for subarray in array:
            newtuple.append(totuple(subarray))
        return tuple(newtuple)
    except TypeError:  # Cannot iterate
        return tuple(array)


def binom(n: int, i: int):
    """
    Returns binomial (n, i)
    """
    assert isinstance(n, int)
    assert isinstance(i, int)
    prod = 1
    if i <= 0 or i >= n:
        return 1
    for j in range(i):
        prod *= (n - j) / (i - j)
    return int(prod)


def eval_spline_nodes(
    knotvector: Tuple[float], nodes: Tuple[float], degree: int
) -> Tuple[Tuple[float]]:
    """
    Returns a matrix M of which M_{ij} = N_{i,degree}(node_j)
    M.shape = (npts, len(nodes))
    """
    assert isinstance(knotvector, (tuple, list))
    assert isinstance(nodes, (tuple, list))
    maxdegree = KnotVector.find_degree(knotvector)
    assert isinstance(degree, int)
    assert 0 <= degree
    assert degree <= maxdegree

    npts = KnotVector.find_npts(knotvector)
    knots = KnotVector.find_knots(knotvector)
    spans = [KnotVector.find_span(knot, knotvector) for knot in knots]
    result = np.zeros((npts, len(nodes)), dtype="object")
    matrix3d = BasisFunction.speval_matrix(knotvector, degree)
    matrix3d = np.array(matrix3d)
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
    return totuple(result)


def eval_rational_nodes(
    knotvector: Tuple[float], weights: Tuple[float], nodes: Tuple[float], degree: int
) -> Tuple[Tuple[float]]:
    """
    Returns a matrix M of which M_{ij} = N_{i,p}(node_j)
    M.shape = (len(weights), len(nodes))
    """
    assert isinstance(knotvector, (tuple, list))
    assert isinstance(weights, (tuple, list))
    assert isinstance(nodes, (tuple, list))
    maxdegree = KnotVector.find_degree(knotvector)
    degree = maxdegree if (degree is None) else degree
    assert isinstance(degree, int)
    assert 0 <= degree
    assert degree <= maxdegree

    rationalvals = eval_spline_nodes(knotvector, nodes, degree)
    rationalvals = np.array(rationalvals)
    for j, node in enumerate(nodes):
        denom = np.inner(rationalvals[:, j], weights)
        for i, weight in enumerate(weights):
            rationalvals[i, j] *= weight / denom
    return totuple(rationalvals)


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
        assert isinstance(npts, int)
        assert npts > 0
        float(a)
        float(b)
        assert a < b

        k = np.arange(0, npts)
        nodes = np.cos(np.pi * (2 * k + 1) / (2 * npts))
        nodes *= (b - a) / 2
        nodes += (a + b) / 2
        nodes.sort()
        return totuple(nodes)

    @staticmethod
    def uniform_nodes(npts: int, a: float = 0, b: float = 1) -> Tuple[float]:
        """
        Returns a list of floats which are equally spaced in [a, b]
        This function doesn't return the extremities
        """
        assert isinstance(npts, int)
        assert npts > 0
        float(a)
        float(b)
        assert a < b

        one = (b - a) / (b - a)
        nodes = [(one + 2 * i * one) / (2 * npts) for i in range(npts)]
        nodes = [a + (b - a) * node for node in nodes]
        return totuple(nodes)

    @staticmethod
    def interp_bezier_matrix(npts: int) -> Tuple[float]:
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
        assert isinstance(npts, int)
        assert npts > 0
        z = npts - 1
        chebyshev0to1 = LeastSquare.chebyshev_nodes(npts)
        matrixbezier = np.zeros((npts, npts), dtype="float64")
        for i, xi in enumerate(chebyshev0to1):
            for j in range(z + 1):
                matrixbezier[i, j] = binom(z, j) * (1 - xi) ** (z - j) * (xi**j)
        invmatrix = np.linalg.inv(matrixbezier)
        return totuple(invmatrix)

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
        The values of F_{i} are gotten from ```interp_bezier_matrix```
        """
        assert isinstance(npts, int)
        matrix = LeastSquare.interp_bezier_matrix(npts)
        return totuple(np.einsum("ij->j", matrix) / npts)

    @staticmethod
    def fit_function(
        knotvector: Tuple[float],
        nodes: Tuple[float],
        weights: Union[Tuple[float], None],
    ) -> Tuple[Tuple[float]]:
        """
        Let C(u) be a curve C(u) of base functions F of given vector
            C(u) = sum_i F_i(u) * P_i
        it's wanted to fit a C(u) into the curve f(u)

        To do it, we do least squares by minimizing
            J(P) = sum_j abs(C(nodej)-f(nodej))

        This function returns a matrix M such
            [P] = [M] * [f(nodes)]
        """
        assert KnotVector.is_valid_vector(knotvector)
        npts = KnotVector.find_npts(knotvector)
        degree = KnotVector.find_degree(knotvector)
        assert len(nodes) <= npts
        if weights is None:
            funcvals = eval_spline_nodes(knotvector, nodes, degree)
        else:
            funcvals = eval_rational_nodes(knotvector, nodes, weights, degree)
        matrix = np.zeros((npts, npts), dtype="object")
        for i in range(npts):
            for j in range(npts):
                matrix[i, j] += np.sum(funcvals[i], funcvals[j])
        numbtype = number_type(matrix)
        if numbtype is Fraction or numbtype is int:
            inverse = invert_fraction_matrix(matrix)
            return np.array(inverse) @ funcvals
        matrix = np.array(matrix)
        return np.linalg.solve(matrix, funcvals)

    @staticmethod
    def spline2spline(
        oldvector: Tuple[float], newvector: Tuple[float]
    ) -> Tuple["Matrix2D"]:
        assert KnotVector.is_valid_vector(oldvector)
        assert KnotVector.is_valid_vector(newvector)
        oldnpts = KnotVector.find_npts(oldvector)
        newnpts = KnotVector.find_npts(newvector)
        oldweights = [1] * oldnpts
        newweights = [1] * newnpts
        result = LeastSquare.func2func(oldvector, oldweights, newvector, newweights)
        return totuple(result)

    @staticmethod
    def func2func(
        oldvector: Tuple[float],
        oldweights: Tuple[float],
        newvector: Tuple[float],
        newweights: Tuple[float],
    ) -> Tuple[np.ndarray]:
        """ """
        assert isinstance(oldvector, (tuple, list))
        assert isinstance(oldweights, (tuple, list))
        assert isinstance(newvector, (tuple, list))
        assert isinstance(newweights, (tuple, list))
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
        integrator = np.array(integrator)

        A = np.zeros((oldnpts, oldnpts), dtype="float64")  # F*F
        B = np.zeros((oldnpts, newnpts), dtype="float64")  # F*G
        C = np.zeros((newnpts, newnpts), dtype="float64")  # G*G
        for start, end in zip(allknots[:-1], allknots[1:]):
            chebynodes = LeastSquare.chebyshev_nodes(nptsinteg, start, end)
            # Integral of the functions in the interval [a, b]
            Fvalues = eval_rational_nodes(oldvector, oldweights, chebynodes, olddegree)
            Gvalues = eval_rational_nodes(newvector, newweights, chebynodes, newdegree)
            A += np.einsum("k,ik,jk->ij", integrator, Fvalues, Fvalues)
            B += np.einsum("k,ik,jk->ij", integrator, Fvalues, Gvalues)
            C += np.einsum("k,ik,jk->ij", integrator, Gvalues, Gvalues)

        T = np.linalg.solve(C, B.T)
        E = A - B @ T
        return totuple(T), totuple(E)


class KnotVector:
    @staticmethod
    def is_valid_vector(knotvector: Tuple[float]) -> bool:
        if isinstance(knotvector, (dict, str)):
            return False
        try:
            knotvector = tuple(knotvector)
            for knot in knotvector:
                float(knot)
            assert knotvector[0] != knotvector[-1]
            for i in range(len(knotvector) - 1):
                assert knotvector[i] <= knotvector[i + 1]
            degree = 0
            while knotvector[degree] == knotvector[degree + 1]:
                degree += 1
            for knot in sorted(knotvector):
                count = knotvector.count(knot)
                assert count <= degree + 1
            assert count == degree + 1
            return True
        except TypeError:
            return False
        except IndexError:
            return False
        except AssertionError:
            return False

    @staticmethod
    def find_degree(knotvector: Tuple[float]) -> int:
        assert KnotVector.is_valid_vector(knotvector)
        degree = 0
        while knotvector[degree] == knotvector[degree + 1]:
            degree += 1
        return degree

    @staticmethod
    def find_npts(knotvector: Tuple[float]) -> int:
        assert KnotVector.is_valid_vector(knotvector)
        degree = KnotVector.find_degree(knotvector)
        return len(knotvector) - degree - 1

    @staticmethod
    def find_span(node: float, knotvector: Tuple[float]) -> int:
        assert KnotVector.is_valid_vector(knotvector)
        assert knotvector[0] <= node
        assert node <= knotvector[-1]
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
        assert KnotVector.is_valid_vector(knotvector)
        assert knotvector[0] <= node
        assert node <= knotvector[-1]
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
        assert KnotVector.is_valid_vector(knotvector)
        knots = []
        for knot in knotvector:
            if knot not in knots:
                knots.append(knot)
        return tuple(knots)

    @staticmethod
    def insert_knots(knotvector: Tuple[float], nodes: Tuple[float]) -> Tuple[float]:
        """
        Returns a new knotvector which contains the previous and new knots.
        This function don't do the validation
        """
        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert knotvector[0] <= node
            assert node <= knotvector[-1]
        newknotvector = tuple(sorted(list(knotvector) + list(nodes)))
        assert KnotVector.is_valid_vector(newknotvector)
        return newknotvector

    @staticmethod
    def remove_knots(knotvector: Tuple[float], nodes: Tuple[float]) -> Tuple[float]:
        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert knotvector[0] <= node
            assert node <= knotvector[-1]
        newknotvector = list(knotvector)
        for node in nodes:
            newknotvector.remove(node)
        newknotvector = tuple(newknotvector)
        assert KnotVector.is_valid_vector(newknotvector)
        return newknotvector

    @staticmethod
    def unite_vectors(vectora: Tuple[float], vectorb: Tuple[float]) -> Tuple[float]:
        """
        Given two vectors with equal limits, returns a valid knotvector such
        it's the union of the given vectors
        """
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)
        assert vectora[0] == vectorb[0]
        assert vectora[-1] == vectorb[-1]
        all_knots = list(set(vectora) | set(vectorb))
        all_mults = [0] * len(all_knots)
        for vector in [vectora, vectorb]:
            for knot in vector:
                index = all_knots.index(knot)
                mult = KnotVector.find_mult(knot, vector)
                if mult > all_mults[index]:
                    all_mults[index] = mult
        final_vector = []
        for knot, mult in zip(all_knots, all_mults):
            final_vector += [knot] * mult
        final_vector = tuple(sorted(final_vector))
        assert KnotVector.is_valid_vector(final_vector)
        return final_vector

    @staticmethod
    def intersect_vectors(vectora: Tuple[float], vectorb: Tuple[float]) -> Tuple[float]:
        """
        Given two vectors with equal limits, returns a knotvector such
        it's the intersection of the given vectors
        """
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)
        assert vectora[0] == vectorb[0]
        assert vectora[-1] == vectorb[-1]
        all_knots = tuple(sorted(set(vectora) & set(vectorb)))
        all_mults = [9999] * len(all_knots)
        for vector in [vectora, vectorb]:
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
        final_vector = tuple(sorted(final_vector))
        assert KnotVector.is_valid_vector(final_vector)
        return final_vector

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
        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert knotvector[0] <= node
            assert node <= knotvector[-1]
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

    @staticmethod
    def nodes_to_insert(vectora: Tuple[float], vectorb: Tuple[float]) -> Tuple[float]:
        """
        It's the inverse of
            vectorb = KnotVector.insert_knots(vectora, nodes)
        """
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)

        knotsa = set(vectora)
        knotsb = set(vectorb)
        assert len(knotsa - knotsb) == 0
        degreea = KnotVector.find_degree(vectora)
        degreeb = KnotVector.find_degree(vectorb)
        assert degreea == degreeb

        nodes = []
        for knot in tuple(sorted(knotsb)):
            multa = KnotVector.find_mult(knot, vectora)
            multb = KnotVector.find_mult(knot, vectorb)
            nodes += [knot] * (multb - multa)
        return tuple(nodes)

    @staticmethod
    def derivate(vector: Tuple[float]) -> Tuple[float]:
        """
        Returns the new vector of derivative of spline
        """
        assert KnotVector.is_valid_vector(vector)
        degree = KnotVector.find_degree(vector)
        knots = KnotVector.find_knots(vector)
        if degree == 0:
            return (vector[0], vector[-1])
        vector = list(vector)
        for knot in knots:
            mult = vector.count(knot)
            if mult == degree + 1:
                vector.remove(knot)
        vector = tuple(vector)
        assert KnotVector.is_valid_vector(vector)
        return vector


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
        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(reqdegree, int)
        assert 0 <= reqdegree
        maxdegree = KnotVector.find_degree(knotvector)
        assert reqdegree <= maxdegree
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
        assert KnotVector.is_valid_vector(vector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert vector[0] <= node
            assert node <= vector[-1]
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

    def one_knot_insert_once(vector: Tuple[float], node: float) -> "Matrix2D":
        """
        Given the knotvector and a node to be inserted, this function
        returns a matrix of transformation T of control points

        Let
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_j N_j(u) * Q_j

        This function returns T such
            [Q] = [T] @ [P]
        """
        assert KnotVector.is_valid_vector(vector)
        float(node)
        assert vector[0] <= node
        assert node <= vector[-1]

        oldnpts = KnotVector.find_npts(vector)
        degree = KnotVector.find_degree(vector)
        oldspan = KnotVector.find_span(node, vector)
        oldmult = KnotVector.find_mult(node, vector)
        one = node / node
        matrix = np.zeros((oldnpts + 1, oldnpts), dtype="object")
        for i in range(oldspan - degree + 1):
            matrix[i, i] = one
        for i in range(oldspan - oldmult, oldnpts):
            matrix[i + 1, i] = one
        for i in range(oldspan - degree + 1, oldspan + 1):
            alpha = node - vector[i]
            alpha /= vector[i + degree] - vector[i]
            matrix[i, i] = alpha
            matrix[i, i - 1] = 1 - alpha
        return totuple(matrix)

    def one_knot_insert(vector: Tuple[float], node: float, times: int) -> "Matrix2D":
        """
        Given the knotvector and a node to be inserted, this function
        returns a matrix of transformation T of control points

        Let
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_j N_j(u) * Q_j

        This function returns T such
            [Q] = [T] @ [P]
        """
        assert KnotVector.is_valid_vector(vector)
        float(node)
        assert vector[0] <= node
        assert node <= vector[-1]
        assert isinstance(times, int)
        assert 0 < times
        oldnpts = KnotVector.find_npts(vector)
        matrix = np.eye(oldnpts, dtype="object")
        for i in range(times):
            incmatrix = Operations.one_knot_insert_once(vector, node)
            matrix = incmatrix @ matrix
            vector = KnotVector.insert_knots(vector, [node])
        return totuple(matrix)

    def knot_insert(vector: Tuple[float], nodes: Tuple[float]) -> "Matrix2D":
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

        assert KnotVector.is_valid_vector(vector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert vector[0] <= node
            assert node <= vector[-1]
        nodes = tuple(nodes)
        setnodes = tuple(sorted(set(nodes) - set([vector[0], vector[-1]])))
        oldnpts = KnotVector.find_npts(vector)
        matrix = np.eye(oldnpts, dtype="object")
        if len(nodes) == 0:
            return totuple(matrix)
        for node in setnodes:
            times = nodes.count(node)
            incmatrix = Operations.one_knot_insert(vector, node, times)
            matrix = incmatrix @ matrix
            vector = KnotVector.insert_knots(vector, [node] * times)
        return totuple(matrix)

    def knot_remove(vector: Tuple[float], nodes: Tuple[float]) -> "Matrix2D":
        """ """
        assert KnotVector.is_valid_vector(vector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert vector[0] <= node
            assert node <= vector[-1]
        nodes = tuple(nodes)
        newvector = KnotVector.remove_knots(vector, nodes)
        matrix, _ = LeastSquare.spline2spline(vector, newvector)
        return totuple(matrix)

    def degree_increase_bezier_once(vector: Tuple[float]) -> "Matrix2D":
        assert KnotVector.is_valid_vector(vector)
        degree = KnotVector.find_degree(vector)
        matrix = np.zeros((degree + 2, degree + 1), dtype="object")
        matrix[0, 0] = 1
        for i in range(1, degree + 1):
            alpha = i / (degree + 1)
            matrix[i, i - 1] = alpha
            matrix[i, i] = 1 - alpha
        matrix[degree + 1, degree] = 1
        return totuple(matrix)

    def degree_increase_bezier(vector: Tuple[float], times: int) -> "Matrix2D":
        """
        Given a bezier curve A(u) of degree p, we want a new bezier curve B(u)
        of degree (p+t) such B(u) = A(u) for every u
        Then, this function returns the matrix of transformation T
            [Q] = [T] @ [P]
            A(u) = sum_{i=0}^{p} B_{i,p}(u) * P_i
            B(u) = sum_{i=0}^{p+t} B_{i,p+t}(u) * Q_i
        """
        assert KnotVector.is_valid_vector(vector)
        assert isinstance(times, int)
        assert times >= 0
        degree = KnotVector.find_degree(vector)
        matrix = np.eye(degree + 1, dtype="object")
        for i in range(times):
            elevateonce = Operations.degree_increase_bezier_once(vector)
            matrix = elevateonce @ matrix
            vector = KnotVector.insert_knots(vector, [vector[0], vector[-1]])
        return totuple(matrix)

    def degree_increase(vector: Tuple[float], times: int) -> "Matrix2D":
        """
        Given a curve A(u) associated with control points P, we want
        to do a degree elevation
        """
        assert KnotVector.is_valid_vector(vector)
        assert isinstance(times, int)
        assert times >= 0
        degree = KnotVector.find_degree(vector)
        npts = KnotVector.find_npts(vector)
        if times == 0:
            return totuple(np.eye(npts, dtype="object"))
        if degree + 1 == npts:
            return Operations.degree_increase_bezier(vector, times)
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
        return totuple(finalmatrix)

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
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)

        degreea = KnotVector.find_degree(vectora)
        degreeb = KnotVector.find_degree(vectorb)
        knotsa = KnotVector.find_knots(vectora)
        assert degreea <= degreeb
        matrix_deginc = Operations.degree_increase(vectora, degreeb - degreea)
        vectora = KnotVector.insert_knots(vectora, knotsa * (degreeb - degreea))

        nodes2ins = KnotVector.nodes_to_insert(vectora, vectorb)
        matrix_knotins = Operations.knot_insert(vectora, nodes2ins)

        finalresult = np.array(matrix_knotins) @ matrix_deginc
        return totuple(finalresult)


class MathOperations:
    @staticmethod
    def knotvector_add(vectora: Tuple[float], vectorb: Tuple[float]) -> Tuple[float]:
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)
        assert vectora[0] == vectorb[0]
        assert vectora[-1] == vectorb[-1]

        vectorc = KnotVector.unite_vectors(vectora, vectorb)
        return tuple(vectorc)

    @staticmethod
    def knotvector_mul(vectora: Tuple[float], vectorb: Tuple[float]) -> Tuple[float]:
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)
        assert vectora[0] == vectorb[0]
        assert vectora[-1] == vectorb[-1]

        degreea = KnotVector.find_degree(vectora)
        degreeb = KnotVector.find_degree(vectorb)
        allknots = tuple(sorted(set(vectora) | set(vectorb)))
        classes = [0] * len(allknots)
        for i, knot in enumerate(allknots):
            multa = KnotVector.find_mult(knot, vectora)
            multb = KnotVector.find_mult(knot, vectorb)
            classes[i] = min(degreea - multa, degreeb - multb)
        degreec = degreea + degreeb
        vectorc = [vectora[0]] * (degreec + 1)
        for knot, classe in zip(allknots[1:-1], classes):
            vectorc += [knot] * (degreec - classe)
        vectorc += [vectora[-1]] * (degreec + 1)
        return tuple(vectorc)

    @staticmethod
    def add_spline_curve(
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
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)
        assert vectora[0] == vectorb[0]
        assert vectora[-1] == vectorb[-1]

        vectorc = MathOperations.knotvector_add(vectora, vectorb)
        matrixa = Operations.matrix_transformation(vectora, vectorc)
        matrixb = Operations.matrix_transformation(vectorb, vectorc)
        return totuple(matrixa), totuple(matrixb)

    @staticmethod
    def sub_spline_curve(
        vectora: Tuple[float], vectorb: Tuple[float]
    ) -> Tuple["Matrix2D"]:
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)
        assert vectora[0] == vectorb[0]
        assert vectora[-1] == vectorb[-1]

        matrixa, matrixb = MathOperations.add_spline_curve(vectora, vectorb)
        matrixb = -np.array(matrixb, dtype="object")
        return matrixa, totuple(matrixb)

    @staticmethod
    def mul_spline_curve(
        vectora: Tuple[float], vectorb: Tuple[float]
    ) -> Tuple["Matrix3D"]:
        """
        Given two spline curves, called A(u) and B(u), it computes and returns
        a new curve C(u) such C(u) = A(u) * B(u) for every u
        Restrictions: The limits of B(u) must be the same as the limits of A(u)
        The parameter `simplify` shows if the function try to reduce at maximum
        the degree and the knots inside.
        """
        assert KnotVector.is_valid_vector(vectora)
        assert KnotVector.is_valid_vector(vectorb)
        assert vectora[0] == vectorb[0]
        assert vectora[-1] == vectorb[-1]

        vectorc = MathOperations.knotvector_mul(vectora, vectorb)
        degreea = KnotVector.find_degree(vectora)
        degreeb = KnotVector.find_degree(vectorb)
        degreec = KnotVector.find_degree(vectorc)
        nptsa = KnotVector.find_npts(vectora)
        nptsb = KnotVector.find_npts(vectorb)
        nptsc = KnotVector.find_npts(vectorc)
        allknots = KnotVector.find_knots(vectorc)

        nptseval = 2 * (degreec + 1)
        nptstotal = nptseval * (len(allknots) - 1)
        allevalnodes = np.empty(nptstotal, dtype="object")
        for i in range(len(allknots) - 1):
            start, end = allknots[i : i + 2]
            chebynodes = LeastSquare.uniform_nodes(nptseval, start, end)
            allevalnodes[i * nptseval : (i + 1) * nptseval] = chebynodes
        allevalnodes = tuple(allevalnodes)

        avals = eval_spline_nodes(vectora, allevalnodes, degreea)
        bvals = eval_spline_nodes(vectorb, allevalnodes, degreeb)
        cvals = eval_spline_nodes(vectorc, allevalnodes, degreec)
        avals = np.array(avals)
        bvals = np.array(bvals)
        cvals = np.array(cvals)

        if isinstance(cvals[0, 0], Fraction):
            matrix2d = cvals @ np.transpose(cvals)
            invmatrix2d = invert_fraction_matrix(matrix2d)
            lstsqmat = invmatrix2d @ cvals
        else:
            cvals = np.array(cvals, dtype="float64")
            lstsqmat = np.linalg.solve(cvals @ cvals.T, cvals)

        matrix3d = np.empty((nptsc, nptsa, nptsb), dtype="object")
        # matrix3d = np.einsum("iz,jz,kz->ijk", lstsqmat, avals, bvals)
        for i, linei in enumerate(lstsqmat):
            for j, linej in enumerate(avals):
                for k, linek in enumerate(bvals):
                    matrix3d[i, j, k] = np.sum(linei * linej * linek)
        return totuple(matrix3d)
