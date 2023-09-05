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

from compmec.nurbs import heavy
from compmec.nurbs.__classes__ import Intface_KnotVector


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

    def __init__(self, knotvector: Tuple[float]):
        """Constructor method of KnotVector

        :raises ValueError: Input is not a valid knot vector
        """
        self.__safe_init(knotvector)

    def __safe_init(self, vector: Tuple[float]) -> KnotVector:
        """Private method to initialize a instance with verifications

        :param newvector: The vector of values
        :type file_loc: tuple[float]
        :raises ValueError: If the given knotvector is not valid
        :return: The same instance
        :rtype: KnotVector
        """
        if isinstance(vector, self.__class__):
            self.__unsafe_init(tuple(vector))
        elif not heavy.KnotVector.is_valid_vector(vector):
            msg = f"Invalid KnotVector of type {type(vector)} = {vector}"
            raise ValueError(msg)
        return self.__unsafe_init(vector)

    def __unsafe_init(self, vector: Tuple[float]) -> KnotVector:
        """Private method to initialize a instance without verifications

        :param newvector: The vector of values
        :type file_loc: tuple[float]
        :return: The same instance
        :rtype: KnotVector
        """
        self.__degree = None
        self.__npts = None
        self.__internal_vector = tuple(vector)
        return self

    def __str__(self) -> str:
        items = [str(item) for item in self]
        return "(" + ", ".join(items) + ")"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self) -> float:
        for knot in self.__internal_vector:
            yield knot

    def __getitem__(self, index):
        return self.__internal_vector[index]

    def __len__(self) -> int:
        return len(self.__internal_vector)

    def __iadd__(self, other: float) -> KnotVector:
        try:
            return self.shift(other)
        except TypeError:
            return self.insert(tuple(other))

    def __isub__(self, other: Union[float, Tuple[float]]):
        try:
            return self.shift(-other)
        except TypeError:
            return self.remove(tuple(other))

    def __imul__(self, other: float):
        return self.scale(other)

    def __itruediv__(self, other: float):
        return self.scale(1 / other)

    def __ior__(self, other: KnotVector) -> KnotVector:
        try:
            newvector = heavy.KnotVector.unite_vectors(tuple(self), tuple(other))
        except AssertionError:
            error_msg = f"Cannot use __or__ between {self} and {other}"
            raise ValueError(error_msg)
        return self.__safe_init(newvector)

    def __iand__(self, other: KnotVector) -> KnotVector:
        try:
            newvector = heavy.KnotVector.intersect_vectors(tuple(self), tuple(other))
        except AssertionError:
            error_msg = f"Cannot use __and__ between {self} and {other}"
            raise ValueError(error_msg)
        return self.__safe_init(newvector)

    def __add__(self, other: Union[float, Tuple[float]]):
        """Shifts all the knots by same given amout"""
        return self.copy().__iadd__(other)

    def __sub__(self, other: Union[float, Tuple[float]]):
        return self.copy().__isub__(other)

    def __mul__(self, other: float):
        return self.copy().__imul__(other)

    def __rmul__(self, other: float):
        return self.copy().__imul__(other)

    def __truediv__(self, other: float):
        return self.copy().__itruediv__(other)

    def __or__(self, other: float):
        return self.copy().__ior__(other)

    def __and__(self, other: float):
        return self.copy().__iand__(other)

    def __eq__(self, other: object):
        if not isinstance(other, self.__class__):
            # Will try to cenvert other
            try:
                other = self.__class__(other)
            except ValueError:
                return False
        if self.npts != other.npts:
            return False
        if self.degree != other.degree:
            return False
        for i, val in enumerate(self):
            if val != other[i]:
                return False
        return True

    def __ne__(self, other: object) -> bool:
        return not self == other

    @property
    def degree(self) -> int:
        """Polynomial degree

        :getter: Returns the degree of the curve
        :setter: Increases or decreases curve's degree
        :type: int

        Example use
        -----------

        >>> from compmec.nurbs import KnotVector
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
        if self.__degree is None:
            self.__degree = heavy.KnotVector.find_degree(tuple(self))
        return int(self.__degree)

    @property
    def npts(self) -> int:
        """Number of control points

        :getter: Returns the number of control points
        :type: int


        Example use
        -----------

        >>> from compmec.nurbs import KnotVector
        >>> knotvector = KnotVector([0., 1.])
        >>> knotvector.npts
        1
        >>> knotvector = KnotVector([1, 1, 2, 3, 3])
        >>> knotvector.npts
        3

        """
        if self.__npts is None:
            self.__npts = heavy.KnotVector.find_npts(tuple(self))
        return int(self.__npts)

    @property
    def knots(self) -> Tuple[float]:
        """Non-repeted knots

        :getter: Non-repeted knots
        :type: tuple[float]

        Example use
        -----------

        >>> from compmec.nurbs import KnotVector
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
        return heavy.KnotVector.find_knots(tuple(self))

    @property
    def limits(self) -> Tuple[float]:
        """The knotvector limits

        :getter: Returns the tuple [Umin, Umax]
        :type: tuple[float]


        Example use
        -----------

        >>> from compmec.nurbs import KnotVector
        >>> knotvector = KnotVector([0., 1.])
        >>> knotvector.limits
        (0., 1.)
        >>> knotvector = KnotVector([1, 1, 2, 3, 3])
        >>> knotvector.degree
        (1, 3)

        """
        return (self[0], self[-1])

    @degree.setter
    def degree(self, value: int):
        diff = int(value) - self.degree
        knots = tuple(self.knots)
        if diff < 0:  # Decrease degree
            self.remove((-diff) * knots)
        if 0 < diff:  # Increase degree
            self.insert(diff * knots)

    def __span(self, nodes: Tuple[float]) -> Tuple[int]:
        """
        Private method of self.span, doesn't raise errors
        """
        spans = []
        vector = tuple(self)
        for node in nodes:
            newspan = heavy.KnotVector.find_span(node, vector)
            spans.append(newspan)
        return tuple(spans)

    def __mult(self, nodes: Tuple[float]) -> Tuple[int]:
        """Private method of mult"""
        mults = []
        vector = tuple(self)
        for node in nodes:
            newmult = heavy.KnotVector.find_mult(node, vector)
            mults.append(newmult)
        return tuple(mults)

    def __valid_node(self, node: float):
        """
        Method of to verify if a node is
        - Is a number
        - inside the interval
        """
        if isinstance(node, (str, dict, tuple, list)):
            raise TypeError
        float(node)  # Verify if it's a number
        if node < self[0] or self[-1] < node:
            return False
        return True

    def __insert(self, nodes: Tuple[float]):
        try:
            assert self.valid_nodes(nodes)
            newvector = heavy.KnotVector.insert_knots(tuple(self), nodes)
        except AssertionError:
            error_msg = f"Cannot insert nodes {nodes} in knotvector {self}"
            raise ValueError(error_msg)
        return self.__safe_init(newvector)

    def __remove(self, nodes: Tuple[float]):
        try:
            assert self.valid_nodes(nodes)
            newvector = heavy.KnotVector.remove_knots(tuple(self), nodes)
        except AssertionError:
            error_msg = f"Cannot remove nodes {nodes} in knotvector {self}"
            raise ValueError(error_msg)
        return self.__safe_init(newvector)

    def copy(self) -> KnotVector:
        """Returns a copy of the object. The internal knots are also copied.

        :return: An exact copy of the instance.
        :rtype: KnotVector

        Example use
        -----------

        >>> from compmec.nurbs import KnotVector
        >>> vector0 = GeneratorKnotVector.random(2, 5)
        >>> vector1 = vector0.copy()
        >>> vector1 == vector0
        True
        >>> id(vector1) == id(vector0)
        False

        """
        knotvector = [deepcopy(knot) for knot in self]
        return self.__class__(knotvector)

    def shift(self, value: float) -> KnotVector:
        """Add ``value`` to each knot

        :param value: The amount to shift every knot
        :type value: float
        :raises TypeError: If ``value`` is not a number
        :return: The same instance
        :rtype: KnotVector

        Example use
        -----------

        >>> from compmec.nurbs import KnotVector
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
        float(value)  # Verify if it's a number
        newvector = tuple([knoti + value for knoti in self])
        return self.__safe_init(newvector)

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

        >>> from compmec.nurbs import KnotVector
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
        newvector = tuple([knoti * value for knoti in self])
        return self.__safe_init(newvector)

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
        >>> from compmec.nurbs import KnotVector
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
        self.__safe_init(new_vector)

    def normalize(self) -> KnotVector:
        """Shift and scale the vector to match the interval [0, 1]

        :return: The same instance
        :rtype: KnotVector

        Example use
        -----------

        >>> from fractions import Fraction
        >>> from compmec.nurbs import KnotVector
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
        >>> from compmec.nurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 2, 3, 3])
        >>> knotvector.insert([2])
        (0, 0, 2, 2, 3, 3)
        >>> knotvector.insert([2])
        ValueError: Cannot insert nodes [2] in knotvector (0, 0, 2, 2, 3, 3)"
        >>> knotvector += [1]  # Same as insert([1])
        (0, 0, 1, 2, 2, 3, 3)

        """
        return self.__insert(nodes)

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
        >>> from compmec.nurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 1, 2, 3, 3])
        >>> knotvector.remove([2])
        (0, 0, 1, 3, 3)
        >>> knotvector.remove([2])
        ValueError: Cannot remove nodes [2] in knotvector (0, 0, 1, 3, 3)"
        >>> knotvector -= [1]  # Same as remove([1])
        (0, 0, 3, 3)

        """
        return self.__remove(nodes)

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

        >>> from compmec.nurbs import KnotVector
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
        onevalue = True
        try:
            iter(nodes)
            onevalue = False
        except TypeError:
            nodes = (nodes,)
        if not self.valid_nodes(nodes):
            error_msg = f"The nodes {nodes} are not valid"
            raise ValueError(error_msg)
        spans = self.__span(nodes)
        return spans[0] if onevalue else spans

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

        >>> from compmec.nurbs import KnotVector
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
        onevalue = True
        try:
            iter(nodes)
            onevalue = False
        except TypeError:
            nodes = (nodes,)
        if not self.valid_nodes(nodes):
            error_msg = f"The nodes {nodes} are not valid"
            raise ValueError(error_msg)
        mults = self.__mult(nodes)
        return mults[0] if onevalue else mults

    def valid_nodes(self, nodes: Tuple[float]) -> bool:
        """Tells if all given nodes are valid

        :param nodes: The list of nodes
        :type nodes: tuple[float]
        :raises TypeError: If ``nodes`` is not a list of numbers
        :return: If all the nodes are in the interval ``[umin, umax]``
        :rtype: bool

        Example use
        -----------

        >>> from compmec.nurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 1, 1])
        >>> knotvector.valid_nodes([0, 0.5, 1])
        True
        >>> knotvector.valid_nodes([-1, 0.5, 1])
        False
        """
        for node in nodes:
            if not self.__valid_node(node):
                return False
        return True


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

        >>> from compmec.nurbs import GeneratorKnotVector
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

        >>> from compmec.nurbs import GeneratorKnotVector
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
        >>> from compmec.nurbs import GeneratorKnotVector
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

        >>> from compmec.nurbs import GeneratorKnotVector
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
        weights = [cls(weight) for weight in weights]
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

        >>> from compmec.nurbs import GeneratorKnotVector
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
