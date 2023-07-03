import abc
from typing import Any, Optional, Tuple, Union

import numpy as np


class Intface_KnotVector(abc.ABC):
    @abc.abstractproperty
    def degree(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def npts(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def verify_input_init(self, knotvector: Tuple[float]):
        raise NotImplementedError

    @abc.abstractmethod
    def verify_valid_span(self, u: Tuple[float]):
        raise NotImplementedError

    @abc.abstractmethod
    def span(self, knot: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def insert_knot(self, knot: float, times: Optional[int] = 1):
        raise NotImplementedError

    @abc.abstractmethod
    def remove_knot(self, knot: float, times: Optional[int] = 1):
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, obj: object) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __ne__(self, obj: object) -> bool:
        raise NotImplementedError


class Intface_BaseFunction_Evaluator_BaseCurve(abc.ABC):
    @abc.abstractproperty
    def knotvector(self) -> Intface_KnotVector:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Intface_Evaluator(Intface_BaseFunction_Evaluator_BaseCurve):
    @abc.abstractproperty
    def first_index(self) -> Union[int, slice]:
        raise NotImplementedError

    @abc.abstractproperty
    def second_index(self) -> Union[int, slice]:
        raise NotImplementedError


class Intface_BaseFunction_BaseCurve(Intface_BaseFunction_Evaluator_BaseCurve):
    @abc.abstractproperty
    def degree(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def npts(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def knot_insert(self, knot: float, times: Optional[int] = 1):
        raise NotImplementedError

    @abc.abstractmethod
    def knot_remove(self, knot: float, times: Optional[int] = 1):
        raise NotImplementedError


class Intface_BaseFunction(Intface_BaseFunction_BaseCurve):
    @abc.abstractmethod
    def __init__(self, knotvector: Intface_KnotVector):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, tup: Any) -> Union[float, np.ndarray]:
        raise NotImplementedError


class Intface_BaseCurve(Intface_BaseFunction_BaseCurve):
    @abc.abstractmethod
    def __init__(self, knotvector: Intface_KnotVector, ctrlpoints: np.ndarray):
        raise NotImplementedError

    @abc.abstractproperty
    def F(self) -> Intface_BaseFunction:
        raise NotImplementedError

    @abc.abstractproperty
    def ctrlpoints(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, obj: object) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __ne__(self, obj: object) -> bool:
        raise NotImplementedError
