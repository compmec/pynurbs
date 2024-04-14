"""
Defines the KnotVector class and GeneratorKnotVector which are responsible to
define the parametric behavior of the curve.
For example, the valid limits to evaluate curve, the polynomial degree, continuity
and smoothness
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np

from pynurbs.__classes__ import Intface_KnotVector
from pynurbs.heavy import ImmutableKnotVector


class KnotVector(Intface_KnotVector):
    """Creates a KnotVector instance

    Examples
    --------

    >>> KnotVector([0, 1])
    (0, 1)
    >>> KnotVector([0, 0, 1, 1])
    (0, 0, 1, 1)
    >>> KnotVector([0, 0, 0.5, 1, 1])
    (0, 0, 0.5, 1, 1)

    """

    def __new__(cls, vector: Tuple[float], degree: Optional[int] = None):
        if isinstance(vector, cls):
            return vector
        instance = super(KnotVector, cls).__new__(cls)
        if not isinstance(vector, ImmutableKnotVector):
            vector = ImmutableKnotVector(vector, degree)
        instance.internal = vector
        return instance

    def __str__(self) -> str:
        return "(" + ", ".join(map(str, self)) + ")"

    def __repr__(self) -> str:
        return str(self)
        return f"KV({self.degree}, {self.npts})"

    def __iter__(self):
        for item in self.internal:
            yield item

    def __getitem__(self, index: int):
        return self.internal[index]

    def __len__(self):
        return len(self.internal)

    def __iadd__(self, other: float) -> KnotVector:
        try:
            return self.shift(other)
        except TypeError:
            return self.insert(other)

    def __isub__(self, other: Union[float, Tuple[float]]):
        try:
            return self.shift(-other)
        except TypeError:
            return self.remove(other)

    def __imul__(self, other: float):
        return self.scale(other)

    def __itruediv__(self, other: float):
        return self.scale(1 / other)

    def __ior__(self, other: KnotVector) -> KnotVector:
        self.internal |= other
        return self

    def __iand__(self, other: KnotVector) -> KnotVector:
        self.internal &= other
        return self

    def __add__(self, other: Union[float, Tuple[float]]):
        """Shifts all the knots by same given amout"""
        return deepcopy(self).__iadd__(other)

    def __sub__(self, other: Union[float, Tuple[float]]):
        return deepcopy(self).__isub__(other)

    def __mul__(self, other: float):
        return deepcopy(self).__imul__(other)

    def __rmul__(self, other: float):
        return deepcopy(self).__imul__(other)

    def __truediv__(self, other: float):
        return deepcopy(self).__itruediv__(other)

    def __or__(self, other: float):
        return deepcopy(self).__ior__(other)

    def __and__(self, other: float):
        return deepcopy(self).__iand__(other)

    def __eq__(self, other: object):
        try:
            other = self.__class__(other)
        except ValueError:
            return False
        return self.internal == other.internal

    def __copy__(self) -> KnotVector:
        return self.__deepcopy__(None)

    def __deepcopy__(self, memo) -> KnotVector:
        knotvector = [deepcopy(knot) for knot in self]
        return self.__class__(knotvector)

    @property
    def internal(self) -> ImmutableKnotVector:
        """Internal immutable knotvector

        :getter: Returns the immutable knot vector instance
        :setter: Sets as new knot vector instance
        :type: ImmutableKnotVector

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0., 1.])
        >>> knotvector.internal
        (0., 1.)

        """
        return self.__internal

    @property
    def degree(self) -> int:
        """Polynomial degree

        :getter: Returns the degree of the curve
        :setter: Increases or decreases curve's degree
        :type: int

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0., 1.])
        >>> knotvector.degree
        0
        >>> knotvector.degree = 1
        >>> print(knotvector)
        (0., 0., 1., 1.)
        >>> knotvector = KnotVector([1, 1, 2, 3, 3])
        >>> knotvector.degree
        1
        >>> knotvector.degree = 3
        >>> print(knotvector)
        (1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3)

        """
        return self.internal.degree

    @property
    def npts(self) -> int:
        """Number of control points

        :getter: Returns the number of control points
        :type: int


        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0., 1.])
        >>> knotvector.npts
        1
        >>> knotvector = KnotVector([1, 1, 2, 3, 3])
        >>> knotvector.npts
        3

        """
        return self.internal.npts

    @property
    def knots(self) -> Tuple[float]:
        """Non-repeted knots

        :getter: Non-repeted knots
        :type: tuple[float]

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0., 1.])
        >>> knotvector.knots
        (0., 1.)
        >>> knotvector = KnotVector([1, 1, 2, 3, 3])
        >>> knotvector.knots
        (1, 2, 3)
        >>> knotvector = KnotVector([0, 0, 0, 1, 2, 2, 3, 3, 3])
        >>> knotvector.knots
        (0, 1, 2, 3)

        """
        return self.internal.knots

    @property
    def limits(self) -> Tuple[float]:
        """The knotvector limits

        :getter: Returns the tuple [Umin, Umax]
        :type: tuple[float]


        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0., 1.])
        >>> knotvector.limits
        (0., 1.)
        >>> knotvector = KnotVector([1, 1, 2, 3, 3])
        >>> knotvector.degree
        (1, 3)

        """
        return self.internal.limits

    @degree.setter
    def degree(self, value: int):
        diff = int(value) - self.degree
        if diff < 0:  # Decrease degree
            self.decrease(-diff)
        if 0 < diff:  # Increase degree
            self.increase(diff)

    @internal.setter
    def internal(self, vector: Tuple[float]):
        if not isinstance(vector, ImmutableKnotVector):
            vector = ImmutableKnotVector(vector)
        self.__internal = vector

    def shift(self, value: float) -> KnotVector:
        """Add ``value`` to each knot

        :param value: The amount to shift every knot
        :type value: float
        :raises TypeError: If ``value`` is not a number
        :return: The same instance
        :rtype: KnotVector

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 1, 1])
        >>> knotvector.shift(1)
        (1, 1, 2, 2)
        >>> knotvector.shift(-1)
        (0, 0, 1, 1)
        >>> knotvector.shift(1.0)
        (1., 1., 2., 2.)
        >>> knotvector += 1  # same as shift(1)
        (2., 2., 3., 3.)
        >>> knotvector -= 1  # same as shift(-1)
        (1., 1., 2., 2.)

        """
        vector = tuple(knoti + value for knoti in self)
        self.internal = ImmutableKnotVector(vector)
        return self

    def scale(self, value: float) -> KnotVector:
        """Multiplies every knot by amount ``value``

        :param value: The amount to scale every knot
        :type value: float
        :raises TypeError: If ``value`` is not a number
        :raises AssertionError: If ``value`` is not positive
        :return: The same instance
        :rtype: KnotVector

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([1, 1, 2, 2])
        >>> knotvector.scale(2)
        (2, 2, 4, 4)
        >>> knotvector *= 2  # same as scale(2)
        (4, 4, 8, 8)
        >>> knotvector.scale(1/2)
        (2., 2., 4., 4.)
        >>> knotvector /= 2  # same as scale(1/2)
        (1., 1., 2., 2.)

        """
        float(value)  # Verify if it's a number
        assert value > 0
        self.internal = ImmutableKnotVector(knoti * value for knoti in self)
        return self

    def convert(self, cls: type, tolerance: Optional[float] = 1e-9) -> KnotVector:
        """Convert the knots from current type to given type.

        If ``tolerance`` is too small, it raises a ValueError cause cannot convert.

        :param cls: The class to convert the knots
        :type cls: type
        :param tolerance: The tolerance to check if each node is very far from other
        :type tolerance: float
        :raises ValueError: If cannot convert all knots to given type for given tolerance
        :return: The same instance
        :rtype: KnotVector

        Example use
        -----------

        >>> from fractions import Fraction
        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([1., 1., 2., 3., 3.])
        >>> knotvector.convert(int)
        (1, 1, 2, 3, 3)
        >>> knotvector.convert(float)
        (1., 1., 2., 3., 3.)
        >>> knotvector.convert(Fraction)
        (Fraction(1, 1), Fraction(1, 1), Fraction(2, 1), Fraction(3, 1), Fraction(3, 1))
        >>> knotvector = KnotVector([0., 0., 0.5, 1., 1.])
        >>> knotvector.convert(int)
        ValueError: Cannot convert knot 0.5 from type <class 'float'> to type <class 'int'>

        """
        new_vector = []
        for knot in self:
            new_knot = cls(knot)
            if abs(new_knot - knot) > tolerance:
                error_msg = "Cannot convert knot %s from type %s to type %s"
                error_msg %= str(knot), str(type(knot)), str(cls)
                raise ValueError(error_msg)
            new_vector.append(new_knot)
        self.internal = new_vector
        return self

    def normalize(self) -> KnotVector:
        """Shift and scale the vector to match the interval [0, 1]

        :return: The same instance
        :rtype: KnotVector

        Example use
        -----------

        >>> from fractions import Fraction
        >>> from pynurbs import KnotVector
        >>> vector = [1, 1, 2, 3, 3]
        >>> knotvector = KnotVector(vector)
        >>> knotvector.normalize()
        (0., 0., 0.5, 1., 1.)
        >>> vector = [2, 2, 3, 4, 4]
        >>> knotvector = KnotVector(vector)
        >>> knotvector.convert(Fraction)
        >>> knotvector.normalize(Fraction)
        (Fraction(0, 1), Fraction(0, 1), Fraction(1, 2), Fraction(1, 1), Fraction(1, 1))

        """
        self.shift(-self[0])
        self.scale(1 / self[-1])
        return self

    def insert(self, nodes: Tuple[float]) -> KnotVector:
        """Insert given nodes inside knotvector

        :param nodes: The nodes to be inserted
        :type nodes: tuple[float]
        :raises ValueError: If cannot insert knots
        :return: The same instance
        :rtype: KnotVector

        Example use
        -----------

        >>> from fractions import Fraction
        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 2, 3, 3])
        >>> knotvector.insert([2])
        (0, 0, 2, 2, 3, 3)
        >>> knotvector.insert([2])
        ValueError: Cannot insert nodes [2] in knotvector (0, 0, 2, 2, 3, 3)"
        >>> knotvector += [1]  # Same as insert([1])
        (0, 0, 1, 2, 2, 3, 3)

        """
        self.internal = self.internal.insert(nodes)
        return self

    def remove(self, nodes: Tuple[float]) -> KnotVector:
        """Remove given nodes inside knotvector

        :param nodes: The nodes to be remove
        :type nodes: tuple[float]
        :raises ValueError: If cannot remove knots
        :return: The same instance
        :rtype: KnotVector

        Example use
        -----------

        >>> from fractions import Fraction
        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 1, 2, 3, 3])
        >>> knotvector.remove([2])
        (0, 0, 1, 3, 3)
        >>> knotvector.remove([2])
        ValueError: Cannot remove nodes [2] in knotvector (0, 0, 1, 3, 3)"
        >>> knotvector -= [1]  # Same as remove([1])
        (0, 0, 3, 3)

        """
        self.internal = self.internal.remove(nodes)
        return self

    def increase(self, times: int) -> KnotVector:
        self.internal = self.internal.increase(times)
        return self

    def decrease(self, times: int) -> KnotVector:
        self.internal = self.internal.decrease(times)
        return self

    def span(self, nodes: Union[float, Tuple[float]]) -> Union[int, Tuple[int]]:
        """Finds the index position of a ``node`` such
        ``knotvector[span] <= node < knotvector[span+1]``

        If ``nodes`` is a vector of numbers, it returns a vector of indexs

        :param nodes: A node to compute the span, or a list of nodes
        :type nodes: float | tuple[float]
        :raises TypeError: If ``nodes`` is not a list of numbers
        :raises ValueError: If at least one node is outside ``[umin, umax]``
        :return: The index of the node
        :rtype: int | tuple[int]

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> vector = [0, 0, 1, 2, 2]
        >>> knotvector = KnotVector(vector)
        >>> knotvector.span(0)
        1
        >>> knotvector.span(0.5)
        1
        >>> knotvector.span(1)
        2
        >>> knotvector.span([0, 0.5, 1, 1.5, 2])
        (1, 1, 2, 2, 2)
        """
        return self.internal.span(nodes)

    def mult(self, nodes: Union[float, Tuple[float]]) -> Union[int, Tuple[int]]:
        """Counts how many times a node is inside the knotvector

        If ``nodes`` is a vector of numbers, it returns a list of mult

        :param nodes: The node to count, or a list of nodes
        :type nodes: float | tuple[float]
        :raises TypeError: If ``nodes`` is not a number or a list of numbers
        :raises ValueError: If the node is outside ``[umin, umax]``
        :return: The index of the node
        :rtype: int | tuple[int]

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> vector = [0, 0, 1, 2, 2]
        >>> knotvector = KnotVector(vector)
        >>> knotvector.mult(0)
        2
        >>> knotvector.mult(1)
        1
        >>> knotvector.mult(0.5)
        0
        >>> knotvector.mult([0, 0.5, 1, 1.2, 1.8, 2])
        (2, 0, 1, 0, 0, 2)
        """
        return self.internal.mult(nodes)

    def valid(self, nodes: Tuple[float]) -> bool:
        """Tells if all given nodes are valid

        :param nodes: The list of nodes
        :type nodes: tuple[float]
        :raises TypeError: If ``nodes`` is not a list of numbers
        :return: If all the nodes are in the interval ``[umin, umax]``
        :rtype: bool

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 1, 1])
        >>> knotvector.valid([0, 0.5, 1])
        True
        >>> knotvector.valid([-1, 0.5, 1])
        False
        """
        return self.internal.valid(nodes)

    def split(self, nodes: Tuple[float]) -> Tuple[KnotVector]:
        """Split the knot vector at given nodes

        :param nodes: The list of nodes
        :type nodes: tuple[float]
        :raises TypeError: If ``nodes`` is not a list of numbers
        :return: The list of splited knot vectors
        :rtype: tuple[KnotVector]

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 1, 1])
        >>> knotvector.split([0.5])
        ((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1))

        """
        vectors = self.internal.split(nodes)
        return tuple(map(self.__class__, vectors))


class GeneratorKnotVector:
    """
    Set of static functions to help creating KnotVector
    of given ``degree`` and ``npts``
    """

    @staticmethod
    def bezier(degree: int, cls: Optional[type] = int) -> KnotVector:
        """Creates the KnotVector of a bezier curve.

        :param degree: The degree of the bezier curve, non-negative
        :type degree: int
        :param cls: The class to convert the number, defaults to ``int``
        :type cls: int(, optional)
        :raises AssertionError: If ``degree`` is not a non-negative integer
        :return: The bezier knot vector
        :rtype: KnotVector

        Example use
        -----------

        >>> from pynurbs import GeneratorKnotVector
        >>> GeneratorKnotVector.bezier(1)
        (0, 0, 1, 1)
        >>> GeneratorKnotVector.bezier(2)
        (0, 0, 0, 1, 1, 1)
        >>> GeneratorKnotVector.bezier(3)
        (0, 0, 0, 0, 1, 1, 1, 1)
        >>> GeneratorKnotVector.bezier(3, float)
        (0., 0., 0., 0., 1., 1., 1., 1.)
        """
        assert isinstance(degree, int)
        assert degree >= 0
        knotvector = (degree + 1) * [cls(0)] + (degree + 1) * [cls(1)]
        return KnotVector(knotvector)

    @staticmethod
    def integer(degree: int, npts: int, cls: Optional[type] = int) -> KnotVector:
        """Creates a KnotVector of equally integer spaced.

        :param degree: The degree of the curve, non-negative
        :type degree: int
        :param npts: The number of control points of the curve
        :type npts: int
        :param cls: The class to convert the number, defaults to ``int``
        :type cls: int(, optional)
        :raises AssertionError: If ``degree`` or ``npts`` is not a non-negative integer
        :raises AssertionError: If ``npts`` is not greater than ``degree``
        :return: The bezier knot vector
        :rtype: KnotVector

        Example use
        -----------

        >>> from pynurbs import GeneratorKnotVector
        >>> GeneratorKnotVector.integer(1, 2)
        (0, 0, 1, 1)
        >>> GeneratorKnotVector.integer(1, 3)
        (0, 0, 1, 2, 2)
        >>> GeneratorKnotVector.integer(2, 5)
        (0, 0, 0, 1, 2, 3, 3, 3)
        >>> GeneratorKnotVector.integer(2, 5, float)
        (0., 0., 0., 1., 2., 3., 3., 3.)
        """
        assert isinstance(degree, int)
        assert isinstance(npts, int)
        assert degree >= 0
        assert npts > degree
        nintknots = npts - degree - 1
        knotvector = degree * [cls(0)]
        knotvector += [cls(i) for i in range(nintknots + 2)]
        knotvector += degree * [cls(nintknots + 1)]
        knotvector = KnotVector(knotvector)
        return knotvector

    @staticmethod
    def uniform(degree: int, npts: int, cls: Optional[type] = int) -> KnotVector:
        """Creates a equally distributed knotvector between [0, 1]

        :param degree: The degree of the curve, non-negative
        :type degree: int
        :param npts: The number of control points of the curve
        :type npts: int
        :param cls: The class to convert the number, defaults to ``int``
        :type cls: int(, optional)
        :raises AssertionError: If ``degree`` or ``npts`` is not a non-negative integer
        :raises AssertionError: If ``npts`` is not greater than ``degree``
        :return: The uniform knotvector of given degree and number of control points
        :rtype: KnotVector

        Example use
        -----------

        >>> from fractions import Fraction
        >>> from pynurbs import GeneratorKnotVector
        >>> GeneratorKnotVector.uniform(1, 2)
        (0, 0, 1, 1)
        >>> GeneratorKnotVector.uniform(1, 3)
        (0, 0, 0.5, 1, 1)
        >>> GeneratorKnotVector.uniform(2, 6)
        (0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1)
        >>> GeneratorKnotVector.uniform(1, 3, Fraction)
        (Fraction(0, 1), Fraction(0, 1), Fraction(1, 2), Fraction(1, 1), Fraction(1, 1))
        """
        knotvector = GeneratorKnotVector.integer(degree, npts, cls)
        knotvector.normalize()
        return knotvector

    @staticmethod
    def random(degree: int, npts: int, cls: Optional[type] = float) -> KnotVector:
        """Creates a random distributed knotvector between [0, 1]

        :param degree: The degree of the curve, non-negative
        :type degree: int
        :param npts: The number of control points of the curve
        :type npts: int
        :param cls: The class to convert the number, defaults to ``float``
        :type cls: float(, optional)
        :raises AssertionError: If ``degree`` or ``npts`` is not a non-negative integer
        :raises AssertionError: If ``npts`` is not greater than ``degree``
        :return: The random knotvector of given degree and number of control points
        :rtype: KnotVector

        Example use
        -----------

        >>> from pynurbs import GeneratorKnotVector
        >>> GeneratorKnotVector.random(1, 2)
        (0, 0, 1, 1)
        >>> GeneratorKnotVector.random(1, 3)
        (0, 0, 0.4, 1, 1)
        >>> GeneratorKnotVector.random(2, 6)
        [0, 0, 0, 0.21, 0.57, 0.61, 1, 1, 1]

        """
        assert isinstance(degree, int)
        assert isinstance(npts, int)
        assert degree >= 0
        assert npts > degree
        weights = np.random.randint(1, 1000, npts - degree)
        weights = [cls(int(weight)) for weight in weights]
        knotvector = GeneratorKnotVector.weight(degree, weights)
        knotvector.normalize()
        return knotvector

    @staticmethod
    def weight(degree: int, weights: Tuple[float]) -> KnotVector:
        """Creates a knotvector of degree ``degree`` based on
        given ``weights`` vector.

        :param degree: The degree of the curve, non-negative
        :type degree: int
        :param weights: The vector of weights
        :type weights: tuple[float]
        :raises AssertionError: If ``degree`` is not a non-negative integer
        :raises AssertionError: If the weights is not an array of positive numbers
        :return: A knotvector of degree ``degree``
        :rtype: KnotVector

        Example use
        -----------

        >>> from pynurbs import GeneratorKnotVector
        >>> GeneratorKnotVector.weights(1, [1])
        (0, 0, 1, 1)
        >>> GeneratorKnotVector.weights(2, [1])
        (0, 0, 0, 1, 1, 1)
        >>> GeneratorKnotVector.weights(2, [2])
        (0, 0, 0, 2, 2, 2)
        >>> GeneratorKnotVector.weights(1, [2, 2])
        (0, 0, 2, 4, 4)
        >>> GeneratorKnotVector.weights(1, [1, 2])
        (0, 0, 1, 3, 3)
        >>> GeneratorKnotVector.weights(1, [1., 2.])
        (0., 0., 1., 3., 3.)

        """
        assert isinstance(degree, int)
        assert degree >= 0
        assert len(weights) > 0
        cls = type(weights[0])
        weights = tuple(weights)
        listknots = [cls(0) for i in range(1 + len(weights))]
        for i, weight in enumerate(weights):
            listknots[i + 1] = listknots[i] + weight
        knotvector = degree * [cls(0)] + listknots + degree * [listknots[-1]]
        knotvector = KnotVector(knotvector)
        return knotvector
