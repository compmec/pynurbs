import pytest
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs.spaceu import getU_uniform
import numpy as np


def test_1():
    n, p = 5, 2
    U = getU_uniform(n, p)
    N = SplineBaseFunction(U)
    dN = N.derivate()
    u = np.linspace(0, 1, 129)
    Nu = N(u)
    dNu = dN(u)
    
def main():
    test_1()

if __name__ == "__main__":
    main()
