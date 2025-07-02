import math
from copy import deepcopy
from fractions import Fraction
from numbers import Real
from typing import Any, Optional, Tuple, Union

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


def totuple(array):
    """
    Convert recursively an array to tuples
    """
    try:
        return tuple(map(tuple, array))
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


def isnumber(obj: Any) -> bool:
    """
    Tells if an object is a number
    """
    if isinstance(obj, Real):
        return True
    if isinstance(obj, (str, tuple, list, set, dict)):
        return False
    try:
        (1.0 * float(obj) + 0) / 4.0
        return True
    except Exception:
        return False


def supports_linear_operation(obj: Any) -> bool:
    """
    Tells if an object suports a linear operations like
    sum and multiplication by scalar
    """
    if isinstance(obj, Real):
        return True
    if isinstance(obj, (str, tuple, list, set, dict)):
        return False
    try:
        0 * obj + 0.4 * obj + (-4) * obj
        return True
    except Exception:
        return False


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
