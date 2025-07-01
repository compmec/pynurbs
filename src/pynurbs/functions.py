from __future__ import annotations

from copy import copy
from typing import Tuple, Union

import numpy as np

from pynurbs.core.basisfunction import ImmutableBasisFunction
from pynurbs.knotspace import KnotVector

from .operations.tools import vectorize


class BaseFunction:
    def __init__(self, knotvector: KnotVector):
        self.knotvector = knotvector
        self.weights = None

    def __eq__(self, other: BaseFunction) -> bool:
        if not isinstance(other, BaseFunction):
            return NotImplemented
        if self.knotvector != other.knotvector:
            return False
        weightleft = self.weights
        weightrigh = other.weights
        weightleft = np.ones(self.npts) if self.weights is None else self.weights
        weightrigh = np.ones(self.npts) if weightrigh is None else weightrigh
        return np.all(weightleft == weightrigh)

    @property
    def knotvector(self) -> KnotVector:
        """The knotvector of the current basis function

        :getter: knotvector of current basis function
        :setter: -
        :type: KnotVector

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 2, 3, 3])
        >>> basis = Function(knotvector)
        >>> basis.knotvector
        (0, 0, 2, 3, 3)
        >>> type(basis.knotvector)
        <class 'pynurbs.knotspace.KnotVector'>

        """
        return self.__knotvector

    @property
    def degree(self) -> int:
        """Polynomial degree of basis function

        :getter: The polynomial degree
        :setter: -
        :type: int

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 2, 3, 3])
        >>> basis = Function(knotvector)
        >>> basis.degree
        1

        """
        return self.knotvector.degree

    @property
    def npts(self) -> int:
        """Number of control points

        :getter: The number of control points
        :setter: -
        :type: int

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 2, 3, 3])
        >>> basis = Function(knotvector)
        >>> basis.npts
        3

        """
        return self.knotvector.npts

    @property
    def knots(self) -> Tuple[float]:
        """The knots of the knotvector

        :getter: knot of the knotvector
        :setter: -
        :type: tuple[float]

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 2, 3, 3])
        >>> basis = Function(knotvector)
        >>> basis.knots
        (0, 2, 3)

        """
        return self.knotvector.knots

    @property
    def weights(self) -> Union[Tuple[float], None]:
        """Weights of the current function. If it's ``None``, it means
        the basis function is not rational

        :getter: Returns the tuple of the weights, or None if there are no weights
        :setter: Set the weights of rational bspline basis functions
        :type: None | tuple[float]

        Example use
        -----------

        >>> from pynurbs import KnotVector
        >>> knotvector = KnotVector([0, 0, 2, 3, 3])
        >>> basis = Function(knotvector)
        >>> basis.weights
        None

        >>> basis.weights = [1, 2, 1]
        >>> basis.weights
        (1, 2, 1)

        """
        return self.__weights

    @degree.setter
    def degree(self, value: int):
        value = int(value)
        self.knotvector.degree = value

    @knotvector.setter
    def knotvector(self, value: KnotVector):
        if not isinstance(value, KnotVector):
            value = KnotVector(value)
        self.__knotvector = value

    @weights.setter
    def weights(self, value: Tuple[float]):
        if value is None:
            self.__weights = None
            return
        value = np.array(value, dtype="object")
        if not np.all(value > 0):
            error_msg = "All weights must be positive!"
            raise ValueError(error_msg)
        if value.shape != (self.npts,):
            error_msg = f"Weights shape invalid! {value.shape} != ({self.npts})"
            raise ValueError(error_msg)
        self.__weights = value

    def __copy__(self) -> BaseFunction:
        return self.__deepcopy__(None)

    def __deepcopy__(self, memo) -> BaseFunction:
        knotvector = copy(self.knotvector)
        newfunc = self.__class__(knotvector)
        if self.weights is not None:
            newfunc.weights = [copy(weight) for weight in self.weights]
        return newfunc


class FunctionEvaluator:
    def __init__(self, func: BaseFunction, i: Union[int, slice], j: int):
        vector = func.knotvector
        self.__weights = func.weights
        self.__first_index = i
        self.__basis = ImmutableBasisFunction(vector.internal, j)

    @vectorize(1, 0)
    def __call__(self, node: float) -> Union[float, Tuple[float]]:
        result = self.__basis(node)
        if self.__weights is not None:
            result *= self.__weights
            result *= 1 / sum(result)
        return result[self.__first_index]


class IndexableFunction(BaseFunction):
    """
    Allows BaseFunction to be indexable
    """

    def __init__(self, knotvector: KnotVector):
        super().__init__(knotvector)

    def __valid_first_index(self, index: Union[int, slice]):
        if not isinstance(index, (int, slice)):
            raise TypeError
        if isinstance(index, int):
            npts = self.npts
            if not (-npts <= index < npts):
                raise IndexError

    def __valid_second_index(self, index: int):
        if not isinstance(index, int):
            raise TypeError
        if not (0 <= index <= self.degree):
            error_msg = f"Second index (={index}) "
            error_msg += f"must be in [0, {self.degree}]"
            raise IndexError(error_msg)

    def __getitem__(self, index) -> FunctionEvaluator:
        if isinstance(index, tuple):
            if len(index) > 2:
                raise IndexError
            i, j = index
        else:
            i, j = index, self.degree
        self.__valid_first_index(i)
        self.__valid_second_index(j)
        return FunctionEvaluator(self, i, j)

    @vectorize(1, 0)
    def __call__(self, node: float) -> Union[float, Tuple[float]]:
        return self[:, self.degree](node)


class Function(IndexableFunction):
    """Basis Function class, to evaluate functions

    Example use
    -----------

    >>> import numpy as np
    >>> from pynurbs import Function
    >>> knotvector = [0, 0, 1, 1]
    >>> basis = Function(knotvector)
    >>> basis.degree
    1
    >>> basis.npts
    2
    >>> basis(0.5)  # same as basis[:, degree](0.5)
    (0.5, 0.5)
    >>> basis([0, 0.5, 1])
    ((0, 0.5, 1), (1, 0.5, 0))

    """

    def __repr__(self) -> str:
        """Official printing"""
        if self.npts == self.degree + 1:
            return f"Bezier function of degree {self.degree}"
        elif self.weights is None:
            msg = "Spline"
        else:
            msg = "Rational"
        msg += f" function of degree {self.degree} "
        msg += f"and {self.npts} points"
        return msg
