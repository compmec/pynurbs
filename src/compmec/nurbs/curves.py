from compmec.nurbs.basefunctions import BaseFunction, SplineBaseFunction, RationalBaseFunction
import numpy as np
from typing import Iterable

class BaseCurve(object):
    def __init__(self, f: BaseFunction, controlpoints: np.ndarray):
        self.f = f
        self.P = controlpoints

    def __call__(self, u: Iterable[float]) -> np.ndarray:
        L = self.f(u)
        return L.T @ self.P


class SplineBaseCurve(BaseCurve):
    def __init__(self, f: SplineBaseFunction, controlpoints: np.ndarray):
        super().__init__(f, controlpoints)


class RationalBaseCurve(BaseCurve):
    def __init__(self, f: RationalBaseFunction, controlpoints: np.ndarray):
        super().__init__(f, controlpoints)

    
