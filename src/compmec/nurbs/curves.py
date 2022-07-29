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

    def derivate(self):
        df = self.f.derivate()
        return self.__class__(df, self.P)

    def insert_knot(self, knot: float, times: int = 1):
        npts, dim = self.P.shape
        knot = float(knot)
        times = int(times)
        spot = self.f.U.spot(knot)
        Unew = list(self.f.U)
        for i in range(times):
            Unew.insert(spot, knot)
        


class SplineCurve(BaseCurve):
    def __init__(self, f: SplineBaseFunction, controlpoints: np.ndarray):
        super().__init__(f, controlpoints)


class RationalCurve(BaseCurve):
    def __init__(self, f: RationalBaseFunction, controlpoints: np.ndarray):
        super().__init__(f, controlpoints)

class BaseXYFunction(object):
    def __init__(self, f: BaseFunction, xconpoints: Iterable[float], yconpoints: Iterable[float]):
        self.f = f
        self.Y = np.array(yconpoints)
        
        nsample = 129
        self.usample = np.linspace(0, 1, nsample)
        L = self.f(self.usample)
        self.xsample = L.T @ np.array(xconpoints)
        if self.xsample[0] < self.xsample[-1]: # Always increase
            for i in range(nsample-1):
                if not self.xsample[i] < self.xsample[i+1]:
                    print("self.xsample[%d] = " % i, self.xsample[i])
                    print("self.xsample[%d] = " % (i+1), self.xsample[i+1])
                    print("self.usample[%d] = " % i, self.usample[i])
                    print("self.usample[%d] = " % (i+1), self.usample[i+1])
                    raise ValueError("1: The x values must be increasing or decreasing")
        else: # Always decrease
            for i in range(nsample-1):
                if not self.xsample[i] > self.xsample[i+1]:
                    print("self.xsample[%d] = " % i, self.xsample[i])
                    print("self.xsample[%d] = " % (i+1), self.xsample[i+1])
                    print("self.usample[%d] = " % i, self.usample[i])
                    print("self.usample[%d] = " % (i+1), self.usample[i+1])
                    raise ValueError("2: The x values must be increasing or decreasing")

    def find_indexs(self, x: Iterable[float]) -> np.ndarray:
        ind = np.zeros(len(x), dtype="int32")
        maxxsample = np.max(self.xsample)
        for i, xi in enumerate(x):
            value = np.where((xi < self.xsample[1:])*(self.xsample[:-1] <= xi))[0]
            if len(value) != 0:
                ind[i] = value[0]
            elif xi == maxxsample:
                ind[i] = np.where(xi == self.xsample)[0][0]
        return ind

    def inverse(self, x: Iterable[float]) -> np.ndarray:
        u = np.zeros(len(x))
        ind = self.find_indexs(x)
        for i, xi in enumerate(x):
            if ind[i] == len(self.xsample)-1:
                u[i] = self.usample[ind[i]]
                continue
            ua, ub = self.usample[ind[i]], self.usample[ind[i]+1]
            xa, xb = self.xsample[ind[i]], self.xsample[ind[i]+1]
            u[i] += ub*(xi - xa)/(xb - xa)
            u[i] += ua*(xb - xi)/(xb - xa)
        return u

    def __call__(self, x: Iterable[float]) -> np.ndarray:
        u = self.inverse(x)
        L = self.f(u)
        return L.T @ self.Y
    
class SplineXYFunction(BaseXYFunction):
    def __init__(self, f: SplineBaseFunction, xconpoints: Iterable[float], yconpoints: Iterable[float]):
        super().__init__(f, xconpoints, yconpoints)

class RationalXYFunction(BaseXYFunction):
    def __init__(self, f: RationalBaseFunction, xconpoints: Iterable[float], yconpoints: Iterable[float]):
        super().__init__(f, xconpoints, yconpoints)