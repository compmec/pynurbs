"""
This module contains very low level functions that can be easily change to another language such as C/C++ (further may be).
They are 'heavy' functions that are called many times and don't require any special package 
Most of these functions works only with integers, floats and tuples.
"""

import math
from copy import deepcopy
from fractions import Fraction
from typing import Optional, Tuple, Union

import numpy as np


class Math:
    @staticmethod
    def gcd(*numbers: Tuple[int]) -> int:
        lenght = len(numbers)
        if lenght == 1:
            return abs(numbers[0])
        if lenght == 2:
            x, y = numbers
        else:
            middle = lenght // 2
            x = Math.gcd(*numbers[:middle])
            y = Math.gcd(*numbers[middle:])
        while y:
            x, y = y, x % y
        return abs(x)

    @staticmethod
    def lcm(*numbers: Tuple[int]) -> int:
        lenght = len(numbers)
        if lenght == 1:
            return numbers[0]
        if lenght == 2:
            x, y = numbers
        else:
            middle = lenght // 2
            x = Math.lcm(*numbers[:middle])
            y = Math.lcm(*numbers[middle:])
        if x == 0 or y == 0:
            return y if x == 0 else y
        return x * y // Math.gcd(x, y)

    @staticmethod
    def factorial(number: int) -> int:
        if number < 2:
            return 1
        prod = 1
        for i in range(2, number + 1):
            prod *= i
        return prod

    @staticmethod
    def comb(upper: int, lower: int) -> int:
        numerator = Math.factorial(upper)
        denominator = Math.factorial(lower)
        denominator *= Math.factorial(upper - lower)
        return numerator // denominator


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
        if isinstance(number, Fraction):
            return Fraction
        return float


def find_roots(knotvector: Tuple[float], ctrlvalues: Tuple[float]) -> Tuple[float]:
    """
    Finds the roots of given a spline function
    Each subinterval [u_{k}, u_{k+1}] can be interpoled
    by a polynomial of degree p.
    Taking out the case of constant equal

    We do it by sampling
    """
    assert KnotVector.is_valid_vector(knotvector)
    assert isinstance(ctrlvalues, tuple)
    tolerance = 1e-8
    for value in ctrlvalues:
        float(value)
    ctrlvalues = np.array(ctrlvalues, dtype="float64")
    knots = KnotVector.find_knots(knotvector)
    degree = KnotVector.find_degree(knotvector)
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


class Linalg:
    @staticmethod
    def solve(matrix: Tuple[Tuple[float]], force: Tuple[Tuple[float]]):
        numbtype = number_type((matrix, force))
        if numbtype not in (int, Fraction):
            matrix = np.array(matrix, dtype="float64")
            force = np.array(force, dtype="float64")
            return totuple(np.linalg.solve(matrix, force))
        matrix = [[deepcopy(elem) for elem in line] for line in matrix]
        inverse = Linalg.invert(matrix)
        result = np.dot(inverse, force)
        if numbtype is int:
            all_int = True
            for i, line in enumerate(result):
                for j, elem in enumerate(line):
                    if elem.denominator == 1:
                        result[i, j] = int(elem)
                    else:
                        all_int = False
            result = result.astype("int64") if all_int else result
        return totuple(result)

    @staticmethod
    def invert(matrix: Tuple[Tuple[float]]):
        numbtype = number_type(matrix)
        if numbtype not in (int, Fraction):
            matrix = np.array(matrix, dtype="float64")
            return totuple(np.linalg.inv(matrix))
        matrix = [[deepcopy(elem) for elem in line] for line in matrix]
        denomins = [1] * len(matrix)
        for i, line in enumerate(matrix):
            lcm = Math.lcm(*[Fraction(elem).denominator for elem in line])
            denomins[i] *= lcm
            for j, elem in enumerate(line):
                line[j] = lcm * elem
        matrix = tuple(tuple(int(elem) for elem in line) for line in matrix)
        diagonal, inverse = Linalg.invert_integer_matrix(matrix)
        inverse = np.array(inverse, dtype="object")
        for i, diag in enumerate(diagonal):
            for j, denom in enumerate(denomins):
                inverse[i, j] = Fraction(denom * inverse[i, j], diag)
        return inverse

    @staticmethod
    def lstsq(matrix: Tuple[Tuple[float]]):
        """
        Given a matrix A of shape (n, m), with n >= m
        We want the best solution X for
            [A] * [X] approx [B]
        To do it, we first transform into a square matrix and solve:
            [A]^T * [A] * [X] = [A]^T * [B]
        This function in fact returns the matrix [M] such
            [X] = [M] * [B]
            [M] = (A^T * A)^{-1} * A^T
        """
        matrix = np.array(matrix)
        assert matrix.shape[0] >= matrix.shape[1]
        if matrix.shape[0] == matrix.shape[1]:
            ident = totuple(np.eye(len(matrix), dtype="object"))
            return Linalg.solve(matrix, ident)
        return Linalg.solve(matrix.T @ matrix, matrix.T)

    def invert_integer_matrix(
        matrix: Tuple[Tuple[int]],
    ) -> Tuple[Tuple[int], Tuple[Tuple[int]]]:
        """
        Given a matrix A with integer entries, this function computes the
        inverse of this matrix by gaussian elimination.

        # Input:
            matrix: Tuple[Tuple[int]]
                Square matrix A of size (m, m) of integer values

        # Output:
            diagonal: Tuple[int]
                The final diagonal D after gaussian elimination, with values d_i
            inverse: Tuple[Tuple[int]]
                The final inversed matrix M = diag(D) * A^{-1}
        """
        side = len(matrix)
        inverse = np.eye(side, dtype="object")
        matrix = np.column_stack((matrix, inverse))

        # Eliminate lower triangle
        for k in range(side):
            # Swap pivos
            if matrix[k, k] == 0:
                for i in range(k + 1, side):
                    if matrix[i, k] != 0:
                        matrix[[k, i]] = matrix[[i, k]]
                        break
            # Eliminate lines bellow
            if matrix[k, k] < 0:
                matrix[k] *= -1
            for i in range(k + 1, side):
                matrix[i] = matrix[i] * matrix[k, k] - matrix[k] * matrix[i, k]
                gdcline = Math.gcd(*matrix[i])
                if gdcline != 1:
                    matrix[i] = matrix[i] // gdcline

        # Eliminate upper triangle
        for k in range(side - 1, 0, -1):
            for i in range(k - 1, -1, -1):
                matrix[i] = matrix[i] * matrix[k, k] - matrix[k] * matrix[i, k]
                gdcline = Math.gcd(*matrix[i])
                if gdcline != 1:
                    matrix[i] = matrix[i] // gdcline
        diagonal = list(np.diag(matrix[:, :side]))
        inverse = matrix[:, side:]
        return totuple(diagonal), totuple(inverse)


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
        knotvector: Tuple[float],
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
        assert KnotVector.is_valid_vector(knotvector)
        npts = KnotVector.find_npts(knotvector)
        degree = KnotVector.find_degree(knotvector)
        assert len(nodes) >= npts
        if weights is None:
            funcvals = eval_spline_nodes(knotvector, nodes, degree)
        else:
            funcvals = eval_rational_nodes(knotvector, weights, nodes, degree)
        return Linalg.lstsq(np.transpose(funcvals))

    @staticmethod
    def spline2spline(
        oldknotvector: Tuple[float],
        newknotvector: Tuple[float],
        fit_nodes: Tuple[float] = None,
    ) -> Tuple["Matrix2D"]:
        """
        Given two bspline curves A(u) and B(u), this
        function returns a matrix [M] such
            [Q] = [M] * [P]
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_i N_i(u) * Q_i
        """
        assert KnotVector.is_valid_vector(oldknotvector)
        assert KnotVector.is_valid_vector(newknotvector)
        oldnpts = KnotVector.find_npts(oldknotvector)
        newnpts = KnotVector.find_npts(newknotvector)
        oldweights = [Fraction(1) for i in range(oldnpts)]
        newweights = [Fraction(1) for i in range(newnpts)]
        result = LeastSquare.func2func(
            oldknotvector, oldweights, newknotvector, newweights, fit_nodes
        )
        return totuple(result)

    @staticmethod
    def func2func(
        oldknotvector: Tuple[float],
        oldweights: Tuple[float],
        newknotvector: Tuple[float],
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
        assert KnotVector.is_valid_vector(oldknotvector)
        assert isinstance(oldweights, (tuple, list))
        assert KnotVector.is_valid_vector(newknotvector)
        assert isinstance(newweights, (tuple, list))

        oldknotvector = tuple(oldknotvector)
        olddegree = KnotVector.find_degree(oldknotvector)
        oldnpts = KnotVector.find_npts(oldknotvector)
        oldknots = KnotVector.find_knots(oldknotvector)

        newknotvector = tuple(newknotvector)
        newdegree = KnotVector.find_degree(newknotvector)
        newnpts = KnotVector.find_npts(newknotvector)
        newknots = KnotVector.find_knots(newknotvector)

        oldknotvector = tuple(
            Fraction(node) if isinstance(node, int) else node for node in oldknotvector
        )
        newknotvector = tuple(
            Fraction(node) if isinstance(node, int) else node for node in newknotvector
        )

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
        integrator = np.array(integrator)

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
    def unite_vectors(
        knotvectora: Tuple[float], knotvectorb: Tuple[float]
    ) -> Tuple[float]:
        """
        Given two vectors with equal limits, returns a valid knotvector such
        it's the union of the given vectors
        """
        assert KnotVector.is_valid_vector(knotvectora)
        assert KnotVector.is_valid_vector(knotvectorb)
        assert knotvectora[0] == knotvectorb[0]
        assert knotvectora[-1] == knotvectorb[-1]
        all_knots = list(set(knotvectora) | set(knotvectorb))
        all_mults = [0] * len(all_knots)
        for vector in [knotvectora, knotvectorb]:
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
    def intersect_vectors(
        knotvectora: Tuple[float], knotvectorb: Tuple[float]
    ) -> Tuple[float]:
        """
        Given two vectors with equal limits, returns a knotvector such
        it's the intersection of the given vectors
        """
        assert KnotVector.is_valid_vector(knotvectora)
        assert KnotVector.is_valid_vector(knotvectorb)
        assert knotvectora[0] == knotvectorb[0]
        assert knotvectora[-1] == knotvectorb[-1]
        all_knots = tuple(sorted(set(knotvectora) & set(knotvectorb)))
        all_mults = [9999] * len(all_knots)
        for vector in [knotvectora, knotvectorb]:
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
    def nodes_to_insert(
        knotvectora: Tuple[float], knotvectorb: Tuple[float]
    ) -> Tuple[float]:
        """
        It's the inverse of
            knotvectorb = KnotVector.insert_knots(knotvectora, nodes)
        """
        assert KnotVector.is_valid_vector(knotvectora)
        assert KnotVector.is_valid_vector(knotvectorb)

        knotsa = set(knotvectora)
        knotsb = set(knotvectorb)
        assert len(knotsa - knotsb) == 0
        degreea = KnotVector.find_degree(knotvectora)
        degreeb = KnotVector.find_degree(knotvectorb)
        assert degreea == degreeb

        nodes = []
        for knot in tuple(sorted(knotsb)):
            multa = KnotVector.find_mult(knot, knotvectora)
            multb = KnotVector.find_mult(knot, knotvectorb)
            nodes += [knot] * (multb - multa)
        return tuple(nodes)

    @staticmethod
    def derivate(knotvector: Tuple[float]) -> Tuple[float]:
        """
        Returns the new vector of derivative of spline
        """
        assert KnotVector.is_valid_vector(knotvector)
        degree = KnotVector.find_degree(knotvector)
        knots = KnotVector.find_knots(knotvector)
        if degree == 0:
            return (knotvector[0], knotvector[-1])
        knotvector = list(knotvector)
        for knot in knots:
            mult = knotvector.count(knot)
            if mult == degree + 1:
                knotvector.remove(knot)
        knotvector = tuple(knotvector)
        assert KnotVector.is_valid_vector(knotvector)
        return knotvector


class BasisFunction:
    @staticmethod
    def horner_method(coefs: Tuple[float], value: float) -> float:
        """
        Horner method is a efficient method of computing polynomials
        Let's say you have a polynomial
            P(x) = a_0 + a_1 * x + ... + a_n * x^n
        A way to compute P(x_0) is
            P(x_0) = a_0 + a_1 * x_0 + ... + a_n * x_0^n
        But a more efficient way is to use
            P(x_0) = ((...((x_0 * a_n + a_{n-1})*x_0)...)*x_0 + a_1)*x_0 + a_0

        Input:
            coefs : Tuple[float] = (a_0, a_1, ..., a_n)
            value : float = x_0
        """
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

    def split_curve(knotvector: Tuple[float], nodes: Tuple[float]):
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
        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert knotvector[0] <= node
            assert node <= knotvector[-1]
        degree = KnotVector.find_degree(knotvector)
        nodes = set(nodes)  # Remove repeted nodes
        nodes -= set([knotvector[0], knotvector[-1]])  # Take out extremities
        nodes = tuple(nodes)
        manynodes = []
        for node in nodes:
            mult = KnotVector.find_mult(node, knotvector)
            manynodes += [node] * (degree + 1 - mult)
        bigvector = KnotVector.insert_knots(knotvector, manynodes)
        bigmatrix = Operations.knot_insert(knotvector, manynodes)
        newvectors = KnotVector.split(knotvector, nodes)
        matrices = []
        for newknotvector in newvectors:
            span = KnotVector.find_span(newknotvector[0], bigvector)
            lowerind = span - degree
            upperind = lowerind + len(newknotvector) - degree - 1
            newmatrix = bigmatrix[lowerind:upperind]
            matrices.append(newmatrix)
        return matrices

    def one_knot_insert_once(knotvector: Tuple[float], node: float) -> "Matrix2D":
        """
        Given the knotvector and a node to be inserted, this function
        returns a matrix of transformation T of control points

        Let
            A(u) = sum_i N_i(u) * P_i
            B(u) = sum_j N_j(u) * Q_j

        This function returns T such
            [Q] = [T] @ [P]
        """
        assert KnotVector.is_valid_vector(knotvector)
        float(node)
        assert knotvector[0] <= node
        assert node <= knotvector[-1]

        oldnpts = KnotVector.find_npts(knotvector)
        degree = KnotVector.find_degree(knotvector)
        oldspan = KnotVector.find_span(node, knotvector)
        oldmult = KnotVector.find_mult(node, knotvector)
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
        knotvector: Tuple[float], node: float, times: int
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
        assert KnotVector.is_valid_vector(knotvector)
        float(node)
        assert knotvector[0] <= node
        assert node <= knotvector[-1]
        assert isinstance(times, int)
        assert 0 < times
        oldnpts = KnotVector.find_npts(knotvector)
        matrix = np.eye(oldnpts, dtype="object")
        for i in range(times):
            incmatrix = Operations.one_knot_insert_once(knotvector, node)
            matrix = incmatrix @ matrix
            knotvector = KnotVector.insert_knots(knotvector, [node])
        return totuple(matrix)

    def knot_insert(knotvector: Tuple[float], nodes: Tuple[float]) -> "Matrix2D":
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

        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert knotvector[0] <= node
            assert node <= knotvector[-1]
        nodes = tuple(nodes)
        setnodes = tuple(sorted(set(nodes) - set([knotvector[0], knotvector[-1]])))
        oldnpts = KnotVector.find_npts(knotvector)
        matrix = np.eye(oldnpts, dtype="object")
        if len(nodes) == 0:
            return totuple(matrix)
        for node in setnodes:
            times = nodes.count(node)
            incmatrix = Operations.one_knot_insert(knotvector, node, times)
            matrix = incmatrix @ matrix
            knotvector = KnotVector.insert_knots(knotvector, [node] * times)
        return totuple(matrix)

    def knot_remove(knotvector: Tuple[float], nodes: Tuple[float]) -> "Matrix2D":
        """ """
        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(nodes, (tuple, list))
        for node in nodes:
            float(node)
            assert knotvector[0] <= node
            assert node <= knotvector[-1]
        nodes = tuple(nodes)
        newknotvector = KnotVector.remove_knots(knotvector, nodes)
        matrix, _ = LeastSquare.spline2spline(knotvector, newknotvector)
        return totuple(matrix)

    def degree_increase_bezier_once(knotvector: Tuple[float]) -> "Matrix2D":
        assert KnotVector.is_valid_vector(knotvector)
        one = knotvector[-1] - knotvector[0]
        one /= one
        degree = KnotVector.find_degree(knotvector)
        matrix = np.zeros((degree + 2, degree + 1), dtype="object")
        matrix[0, 0] = one
        for i in range(1, degree + 1):
            alpha = (one * i) / (degree + 1)
            matrix[i, i - 1] = alpha
            matrix[i, i] = one - alpha
        matrix[degree + 1, degree] = one
        return totuple(matrix)

    def degree_increase_bezier(knotvector: Tuple[float], times: int) -> "Matrix2D":
        """
        Given a bezier curve A(u) of degree p, we want a new bezier curve B(u)
        of degree (p+t) such B(u) = A(u) for every u
        Then, this function returns the matrix of transformation T
            [Q] = [T] @ [P]
            A(u) = sum_{i=0}^{p} B_{i,p}(u) * P_i
            B(u) = sum_{i=0}^{p+t} B_{i,p+t}(u) * Q_i
        """
        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(times, int)
        assert times >= 0
        degree = KnotVector.find_degree(knotvector)
        matrix = np.eye(degree + 1, dtype="object")
        for i in range(times):
            elevateonce = Operations.degree_increase_bezier_once(knotvector)
            matrix = elevateonce @ matrix
            knotvector = KnotVector.insert_knots(
                knotvector, [knotvector[0], knotvector[-1]]
            )
        return totuple(matrix)

    def degree_increase(knotvector: Tuple[float], times: int) -> "Matrix2D":
        """
        Given a curve A(u) associated with control points P, we want
        to do a degree elevation
        """
        assert KnotVector.is_valid_vector(knotvector)
        assert isinstance(times, int)
        assert times >= 0
        degree = KnotVector.find_degree(knotvector)
        npts = KnotVector.find_npts(knotvector)
        if times == 0:
            return totuple(np.eye(npts, dtype="object"))
        if degree + 1 == npts:
            return Operations.degree_increase_bezier(knotvector, times)
        nodes = KnotVector.find_knots(knotvector)
        newvectors = KnotVector.split(knotvector, nodes)
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
            mult = KnotVector.find_mult(node, knotvector)
            insertednodes += (degree + 1 - mult) * [node]
        bigvector = KnotVector.insert_knots(knotvector, insertednodes)
        incbigvector = KnotVector.insert_knots(bigvector, times * nodes)
        removematrix = Operations.knot_remove(incbigvector, insertednodes)

        bigmatrix = np.array(bigmatrix)
        removematrix = np.array(removematrix)
        finalmatrix = removematrix @ bigmatrix
        return totuple(finalmatrix)

    def matrix_transformation(knotvectora: Tuple[float], knotvectorb: Tuple[float]):
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
        assert KnotVector.is_valid_vector(knotvectora)
        assert KnotVector.is_valid_vector(knotvectorb)

        degreea = KnotVector.find_degree(knotvectora)
        degreeb = KnotVector.find_degree(knotvectorb)
        knotsa = KnotVector.find_knots(knotvectora)
        assert degreea <= degreeb
        matrix_deginc = Operations.degree_increase(knotvectora, degreeb - degreea)
        knotvectora = KnotVector.insert_knots(knotvectora, knotsa * (degreeb - degreea))

        nodes2ins = KnotVector.nodes_to_insert(knotvectora, knotvectorb)
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
        assert KnotVector.is_valid_vector(knotvectora)
        assert KnotVector.is_valid_vector(knotvectorb)
        assert knotvectora[0] == knotvectorb[0]
        assert knotvectora[-1] == knotvectorb[-1]
        return MathOperations.mul_spline_curve(knotvectora, knotvectorb)

    @staticmethod
    def knotvector_mul(
        knotvectora: Tuple[float], knotvectorb: Tuple[float]
    ) -> Tuple[float]:
        assert KnotVector.is_valid_vector(knotvectora)
        assert KnotVector.is_valid_vector(knotvectorb)
        assert knotvectora[0] == knotvectorb[0]
        assert knotvectora[-1] == knotvectorb[-1]

        degreea = KnotVector.find_degree(knotvectora)
        degreeb = KnotVector.find_degree(knotvectorb)
        allknots = tuple(sorted(set(knotvectora) | set(knotvectorb)))
        classes = [0] * len(allknots)
        for i, knot in enumerate(allknots):
            multa = KnotVector.find_mult(knot, knotvectora)
            multb = KnotVector.find_mult(knot, knotvectorb)
            classes[i] = min(degreea - multa, degreeb - multb)
        degreec = degreea + degreeb
        knotvectorc = [knotvectora[0]] * (degreec + 1)
        for knot, classe in zip(allknots[1:-1], classes):
            knotvectorc += [knot] * (degreec - classe)
        knotvectorc += [knotvectora[-1]] * (degreec + 1)
        return tuple(knotvectorc)

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
        assert KnotVector.is_valid_vector(knotvectora)
        assert KnotVector.is_valid_vector(knotvectorb)
        assert knotvectora[0] == knotvectorb[0]
        assert knotvectora[-1] == knotvectorb[-1]

        knotvectorc = KnotVector.unite_vectors(knotvectora, knotvectorb)
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
        assert KnotVector.is_valid_vector(knotvectora)
        assert KnotVector.is_valid_vector(knotvectorb)
        assert knotvectora[0] == knotvectorb[0]
        assert knotvectora[-1] == knotvectorb[-1]

        knotvectorc = MathOperations.knotvector_mul(knotvectora, knotvectorb)
        degreea = KnotVector.find_degree(knotvectora)
        degreeb = KnotVector.find_degree(knotvectorb)
        degreec = KnotVector.find_degree(knotvectorc)
        nptsa = KnotVector.find_npts(knotvectora)
        nptsb = KnotVector.find_npts(knotvectorb)
        nptsc = KnotVector.find_npts(knotvectorc)
        allknots = KnotVector.find_knots(knotvectorc)

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
    def difference_vector(knotvector: Tuple[float]) -> Tuple[float]:
        assert KnotVector.is_valid_vector(knotvector)
        degree = KnotVector.find_degree(knotvector)
        assert degree > 0
        npts = KnotVector.find_npts(knotvector)
        avals = np.zeros(npts, dtype="float64")
        for i in range(npts):
            diff = knotvector[i + degree] - knotvector[i]
            if diff != 0:
                avals[i] = degree / diff
        return totuple(avals)

    @staticmethod
    def difference_matrix(knotvector: Tuple[float]) -> np.ndarray:
        assert KnotVector.is_valid_vector(knotvector)

        avals = Calculus.difference_vector(knotvector)
        npts = len(avals)
        matrix = np.diag(avals)
        for i in range(npts - 1):
            matrix[i, i + 1] = -avals[i + 1]
        return totuple(matrix)

    @staticmethod
    def derivate_nonrational_bezier(
        knotvector: Tuple[float], reduce: bool = True
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
        assert KnotVector.is_valid_vector(knotvector)
        degree = KnotVector.find_degree(knotvector)
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
    def derivate_nonrational_spline(knotvector: Tuple[float]) -> Tuple[Tuple[float]]:
        """
        Given a spline C(u) of degree p, this function returns matrix [M] such
            [Q] = [M] * [P]
            C(u) = sum_{i=0}^{n} N_{i,p}(u) * P_i
            C'(u) = sum_{i=0}^{m} N_{i,q-1}(u) * Q_i
        The matrix size if (m, n)

        Normally q = p-1, since it decreases the degree.
        If reduce is False, it does a degree elevation and keeps the same degree
        """
        assert KnotVector.is_valid_vector(knotvector)
        matrix = Calculus.difference_matrix(knotvector)
        matrix = np.transpose(matrix)[1:]
        return totuple(matrix)

    @staticmethod
    def derivate_rational_bezier(knotvector: Tuple[float]) -> Tuple[Tuple[float]]:
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
        matrixmult = MathOperations.mult_nonrat_bezier(knotvector, knotvector)
        # matrixderi = Calculus.derivate_nonrational_bezier(knotvector, False)
        matrixderi = Calculus.derivate_nonrational_bezier(knotvector)
        elevate = Operations.degree_increase_bezier_once(knotvector[1:-1])
        matrixderi = np.dot(elevate, matrixderi)
        matrixleft = np.tensordot(np.transpose(matrixderi), matrixmult, axes=1)
        matrixrigh = matrixmult @ matrixderi
        return totuple(matrixleft - matrixrigh), totuple(matrixmult)


class NodeSample:
    __cheby = {1: (Fraction(1, 2),)}
    __gauss = {1: (Fraction(1, 2),)}

    @staticmethod
    def closed_linspace(npts: int, cls: Optional[type] = Fraction) -> Tuple[float]:
        """Returns equally distributed nodes in [0, 1]
        Include the extremities

        Example
        ------------
        >>> NodeSample.closed_linspace(2)
        (0, 1)
        >>> NodeSample.closed_linspace(3)
        (0, 1/2, 1)
        >>> NodeSample.closed_linspace(4)
        (0, 1/3, 2/3, 1)
        >>> NodeSample.closed_linspace(5)
        (0, 1/4, 2/4, 3/4, 1)
        >>> NodeSample.closed_linspace(6)
        (0, 1/5, 2/5, 3/5, 4/5, 1)
        """
        assert isinstance(npts, int)
        assert npts > 1
        nums = tuple(range(0, npts))
        nums = tuple(cls(num) / (npts - 1) for num in nums)
        return nums

    @staticmethod
    def open_linspace(npts: int, cls: Optional[type] = Fraction) -> Tuple[float]:
        """Returns equally distributed nodes in (0, 1)
        Exclude the extremities

        Example
        ------------
        >>> NodeSample.open_linspace(1)
        (1/2, )
        >>> NodeSample.open_linspace(2)
        (1/4, 3/4)
        >>> NodeSample.open_linspace(3)
        (1/6, 3/6, 5/6)
        >>> NodeSample.open_linspace(4)
        (1/8, 3/8, 5/8, 7/8)
        >>> NodeSample.open_linspace(5)
        (1/10, 3/10, 5/10, 7/10, 9/10)
        """
        assert isinstance(npts, int)
        assert npts > 0
        nums = range(1, 2 * npts, 2)
        nums = tuple(cls(num) / (2 * npts) for num in nums)
        return nums

    @staticmethod
    def chebyshev(npts: int) -> Tuple[float]:
        """
        Returns chebyshev nodes in the space [0, 1]
        `Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`_


        >>> NodeSample.chebyshev(1)
        (0.5,)
        >>> NodeSample.chebyshev(2)
        (0.146, 0.854)
        >>> NodeSample.chebyshev(3)
        (0.067, 0.5, 0.933)
        >>> NodeSample.chebyshev(4)
        (0.038, 0.309, 0.691, 0.962)
        >>> NodeSample.chebyshev(5)
        (0.024, 0.206, 0.5, 0.794, 0.976)
        """
        assert isinstance(npts, int)
        assert npts > 0
        if npts not in NodeSample.__cheby:
            nums = NodeSample.open_linspace(npts)
            nums = tuple(math.sin(0.5 * math.pi * num) ** 2 for num in nums)
            NodeSample.__cheby[npts] = nums
        return NodeSample.__cheby[npts]

    @staticmethod
    def gauss_legendre(npts: int) -> Tuple[float]:
        """
        Returns gauss legendre quadrature nodes in the space [0, 1]
        `Gauss-Legendre quadrature <https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature>`_

        >>> NodeSample.gauss_legendre(1)
        (0.5,)
        >>> NodeSample.gauss_legendre(2)
        (0.146, 0.854)
        >>> NodeSample.gauss_legendre(3)
        (0.067, 0.5, 0.933)
        >>> NodeSample.gauss_legendre(4)
        (0.038, 0.309, 0.691, 0.962)
        >>> NodeSample.gauss_legendre(5)
        (0.024, 0.206, 0.5, 0.794, 0.976)
        """
        assert isinstance(npts, int)
        assert npts > 0
        if npts not in NodeSample.__gauss:
            nums, _ = np.polynomial.legendre.leggauss(npts)
            nums = (1 + nums) / 2
            NodeSample.__gauss[npts] = tuple(nums)
        return NodeSample.__gauss[npts]


class IntegratorArray:
    __closed_newton = {
        2: (Fraction(1, 2), Fraction(1, 2)),
        3: (Fraction(1, 6), Fraction(2, 3), Fraction(1, 6)),
        4: (Fraction(1, 8), Fraction(3, 8), Fraction(3, 8), Fraction(1, 8)),
    }
    __open_newton = {
        1: (Fraction(1),),
        2: (Fraction(1, 2), Fraction(1, 2)),
        3: (Fraction(3, 8), Fraction(1, 4), Fraction(3, 8)),
    }
    __cheby = {
        1: (Fraction(1),),
        2: (Fraction(1, 2), Fraction(1, 2)),
        3: (Fraction(2, 9), Fraction(5, 9), Fraction(2, 9)),
    }
    __gauss = {
        1: (Fraction(1),),
        2: (Fraction(1, 2), Fraction(1, 2)),
        3: (Fraction(5, 18), Fraction(4, 9), Fraction(5, 18)),
    }

    @staticmethod
    def interpolate_bezier(nodes: Tuple[float]) -> Tuple[Tuple[float]]:
        """Returns a matrix that interpolates a function at given nodes using bezier

        This function returns the inverse of matrix [M] which
        interpolates a bezier curve C at the given nodes
            C(u) = sum_{i=0}^{p} B_{i,p}(u) * P_{i}
            B_{i,p}(u) = binom(p, i) * (1-u)^{p-i} * u^i
            [M]_{i,k} = B_{i,p}(u_k)
            [M] * [P] = [f(x_k)]

        Example
        ------------
        >>> nodes = (0, 0.2, 1)
        >>> IntegratorArray.interpolate_bezier(nodes)
        ((1, -2, 0), (0, 25/8, 0), (0, -1/8, 1))
        >>> nodes = (0, 0.5, 1)
        >>> IntegratorArray.interpolate_bezier(nodes)
        ((1, -1/2, 0), (0, 2, 0), (0, -1/2, 1))

        """
        assert isinstance(nodes, tuple)
        for node in nodes:
            float(node)
            assert 0 <= node
            assert node <= 1
        degree = len(nodes) - 1
        matrix_bezier = np.zeros((degree + 1, degree + 1), dtype="object")
        for k, uk in enumerate(nodes):
            for i in range(degree + 1):
                matrix_bezier[i, k] = (
                    Math.comb(degree, i) * (1 - uk) ** (degree - i) * (uk**i)
                )
        matrix_bezier = totuple(matrix_bezier)
        inverse = Linalg.invert(matrix_bezier)
        inverse = tuple(map(tuple, inverse))
        return inverse

    @staticmethod
    def bezier_integrator_array(nodes: Tuple[float]) -> Tuple[float]:
        """Computes the weights to integrate at given nodes

        Given ``nodes`` the positions of ``n`` values of ``x_i``,
        this function returns ``n`` values of ``w_i`` such

        int_{0}^{1} f(u) du = sum_{i=0}^{n-1} w_i * f(x_i)

        Example
        ------------
        >>> nodes = (0, 0.2, 1)
        >>> IntegratorArray.bezier_integrator_array(nodes)
        (1/3, 1/3, 1/3)
        >>> nodes = (0, 0.5, 1)
        >>> IntegratorArray.bezier_integrator_array(nodes)
        (1/3, 1/3, 1/3)
        """
        matrix = IntegratorArray.interpolate_bezier(nodes)
        array = [sum(line) / len(nodes) for line in matrix]
        return tuple(array)

    @staticmethod
    def closed_newton_cotes(npts: int) -> Tuple[Tuple[float]]:
        """Returns the weight array for closed newton-cotes formula
        in the interval [0, 1]

        Example
        ------------
        >>> IntegratorArray.closed_newton_cotes(2)
        (1/2, 1/2)
        >>> IntegratorArray.closed_newton_cotes(3)
        (1/6, 4/6, 1/6)
        >>> IntegratorArray.closed_newton_cotes(4)
        (1/8, 3/8, 3/8, 1/8)
        >>> IntegratorArray.closed_newton_cotes(5)
        (7/90, 16/45, 2/15, 16/45, 7/90)
        """
        assert isinstance(npts, int)
        assert npts > 1
        if npts not in IntegratorArray.__closed_newton:
            nodes = NodeSample.closed_linspace(npts, Fraction)
            weights = IntegratorArray.bezier_integrator_array(nodes)
            IntegratorArray.__closed_newton[npts] = weights
        return IntegratorArray.__closed_newton[npts]

    @staticmethod
    def open_newton_cotes(npts: int) -> Tuple[Tuple[float]]:
        """Returns the weight array for open newton-cotes formula
        in the interval (0, 1)

        Example
        ------------
        >>> IntegratorArray.open_newton_cotes(1)
        (1, )
        >>> IntegratorArray.open_newton_cotes(2)
        (1/2, 1/2)
        >>> IntegratorArray.open_newton_cotes(3)
        (3/8, 1/4, 3/8)
        >>> IntegratorArray.open_newton_cotes(4)
        (13/48, 11/48, 11/48, 13/48)
        >>> IntegratorArray.open_newton_cotes(5)
        (275/1152, 25/288, 67/192, 25/288, 275/1152)

        """
        assert isinstance(npts, int)
        assert npts > 0
        if npts not in IntegratorArray.__open_newton:
            nodes = NodeSample.open_linspace(npts, Fraction)
            weights = IntegratorArray.bezier_integrator_array(nodes)
            IntegratorArray.__open_newton[npts] = weights
        return IntegratorArray.__open_newton[npts]

    @staticmethod
    def chebyshev(npts: int) -> Tuple[float]:
        """Returns the weight array for integrate at chebyshev nodes

        Example
        ------------
        >>> IntegratorArray.chebyshev(1)
        (1, )
        >>> IntegratorArray.chebyshev(2)
        (1/2, 1/2)
        >>> IntegratorArray.chebyshev(3)
        (3/8, 1/4, 3/8)
        >>> IntegratorArray.chebyshev(4)
        (13/48, 11/48, 11/48, 13/48)
        >>> IntegratorArray.chebyshev(5)
        (275/1152, 25/288, 67/192, 25/288, 275/1152)

        """
        assert isinstance(npts, int)
        assert 0 < npts
        if npts not in IntegratorArray.__cheby:
            nodes = NodeSample.chebyshev(npts)
            weights = IntegratorArray.bezier_integrator_array(nodes)
            IntegratorArray.__cheby[npts] = weights
        return IntegratorArray.__cheby[npts]

    @staticmethod
    def gauss_legendre(npts: int) -> Tuple[float]:
        """Returns the weight array for integrate at gauss nodes

        Example
        ------------
        >>> IntegratorArray.chebyshev(1)
        (1, )
        >>> IntegratorArray.chebyshev(2)
        (1/2, 1/2)
        >>> IntegratorArray.chebyshev(3)
        (3/8, 1/4, 3/8)
        >>> IntegratorArray.chebyshev(4)
        (13/48, 11/48, 11/48, 13/48)
        >>> IntegratorArray.chebyshev(5)
        (275/1152, 25/288, 67/192, 25/288, 275/1152)

        """
        assert isinstance(npts, int)
        assert 0 < npts
        if npts not in IntegratorArray.__gauss:
            _, weights = np.polynomial.legendre.leggauss(npts)
            IntegratorArray.__gauss[npts] = tuple(weights / 2)
        return IntegratorArray.__gauss[npts]
