import pytest
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs.spaceu import getU_uniform, getU_random
import numpy as np
from matplotlib import pyplot as plt


def test_1():
    n, p = 5, 2
    U = getU_uniform(n, p)
    N = SplineBaseFunction(U)
    dN = N.derivate()

    u = np.linspace(0, 1, 129)

    plt.figure()
    Nu = N(u)
    print("N.e = ", N.p)
    for i in range(n):
        plt.plot(u, Nu[i], label=r"$N_{%d%d}$"%(i, p))
    plt.legend()

    plt.figure()
    print("dN.e = ", dN.p)
    dNu = dN(u)
    print(dNu.shape)
    for i in range(n):
        plt.plot(u, dNu[i], label=r"$N'_{%d%d}$"%(i, p))
    plt.legend()
    plt.show()

def main():
    test_1()

if __name__ == "__main__":
    main()
