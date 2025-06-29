from numbers import Real
from typing import Any, Generic, Iterable, Tuple, Union

import numpy as np

from .basisfunction import ImmutableBasisFunction


def permutations(numbers: Tuple[int, ...]) -> Iterable[Tuple[int, ...]]:
    """
    Computes the permutations of the numbers

    Example
    -------
    >>> permutations([2])
    [(0, ), (1, )]
    >>> permutations([2, 3])
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    """
    if len(numbers) > 1:
        for index in range(numbers[0]):
            for permu in permutations(numbers[1:]):
                yield (index,) + permu
    else:
        for index in range(numbers[0]):
            yield (index,)


class Container(Generic[Any]):

    @staticmethod
    def __find_ndim(ctrlpoints: Any) -> int:
        ndim = 0
        try:
            while True:
                iter(ctrlpoints)
                ndim += 1
                ctrlpoints = ctrlpoints[0]
        except Exception:
            return ndim

    @staticmethod
    def __find_shape(ctrlpoints: Any, ndim: int) -> Tuple[int, ...]:
        shape = [0] * ndim
        for i in range(ndim):
            shape[i] = len(ctrlpoints)
            ctrlpoints = ctrlpoints[0]
        return tuple(shape)

    def __init__(self, ctrlpoints: Any, ndim: Union[None, int] = None):
        if ndim is None:
            ndim = Container.__find_ndim(ctrlpoints)
        self.__shape = Container.__find_shape(ctrlpoints, ndim)
        self.__ctrlpoints = ctrlpoints

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__shape

    def __getitem__(self, indexs: Tuple[int, ...]) -> Any:
        ctrlpoint = self.__ctrlpoints
        for index in indexs:
            ctrlpoint = ctrlpoint[index]
        return ctrlpoint


class ImmuntableManifold:
    """
    f(x1, x2, ..., xn)
    """

    def __init__(
        self, allbasis: Iterable[ImmutableBasisFunction], ctrlpoints: Container[Any]
    ):

        allbasis = tuple(allbasis)
        if not all(isinstance(fun, ImmutableBasisFunction) for fun in allbasis):
            raise TypeError
        if isinstance(ctrlpoints, Container):
            if ctrlpoints.ndim != len(allbasis):
                raise ValueError
        else:
            ctrlpoints = Container(ctrlpoints, len(allbasis))
        self.__allbasis = allbasis
        self.__ctrlpoints = ctrlpoints

    @property
    def ndim(self) -> int:
        return len(self.__allbasis)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(basis.npts for basis in self.__allbasis)

    def __call__(self, node: Tuple[Real, ...]) -> Any:
        node = tuple(node)
        if len(node) != self.ndim:
            raise ValueError
        result = 0 * self.__ctrlpoints[*(0,) * self.ndim]
        basivalues = [basis.eval(nodei) for nodei, basis in zip(node, self.__allbasis)]
        for indexs in permutations(self.shape):
            scalar = 1
            for i, index in enumerate(indexs):
                scalar *= basivalues[i][index]
            if scalar:
                result += scalar * self.__ctrlpoints[*indexs]
        return result
