import abc
from typing import Tuple, Union

import numpy as np


class Intface_KnotVector(abc.ABC):
    @abc.abstractproperty
    def degree(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def npts(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def knots(self) -> Tuple[float]:
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, obj: object) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __ne__(self, obj: object) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def shift(self, value: float):
        raise NotImplementedError

    @abc.abstractmethod
    def span(self, nodes: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def mult(self, nodes: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        raise NotImplementedError


class Intface_Evaluator(abc.ABC):
    @abc.abstractproperty
    def eval(self) -> Union[int, slice]:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Intface_BaseFunction_BaseCurve(abc.ABC):
    @abc.abstractmethod
    def __eq__(self, obj: object) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __ne__(self, obj: object) -> bool:
        raise NotImplementedError

    @abc.abstractproperty
    def knotvector(self) -> Intface_KnotVector:
        raise NotImplementedError

    @abc.abstractproperty
    def degree(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def npts(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def knots(self) -> Tuple[float]:
        raise NotImplementedError

    @abc.abstractproperty
    def weights(self) -> Tuple[float]:
        raise NotImplementedError


class Intface_BaseFunction(Intface_BaseFunction_BaseCurve):
    @abc.abstractmethod
    def __init__(self, knotvector: Intface_KnotVector):
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, nodes: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index) -> Intface_Evaluator:
        raise NotImplementedError


class Intface_BaseCurve(Intface_BaseFunction_BaseCurve):
    @abc.abstractmethod
    def __init__(self, knotvector: Intface_KnotVector, ctrlpoints: np.ndarray):
        raise NotImplementedError

    @abc.abstractproperty
    def ctrlpoints(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def knot_clean(self):
        raise NotImplementedError

    @abc.abstractmethod
    def degree_clean(self):
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self):
        raise NotImplementedError
