import numpy as np
from matplotlib import pyplot as plt
from Bspline import SplineBaseFunction
from numpy import linalg as la

def f(x):
    return np.sin(x)

if __name__ == "__main__":
    a, b = 0, 2*np.pi
    p, n = 2, 10
    x = np.linspace(a, b, n)
    y = f(x)
    xplot = np.linspace(a, b, 1025)
    uplot = (xplot-xplot[0])/(xplot[-1]-xplot[0])
    yplot = f(xplot)

    ubar = (x-x[0])/(x[-1]-x[0])
    U = np.zeros(n+p+1)
    for j in range(1, n-p):
        U[j+p] = np.sum(ubar[j:j+p])/p
    U[n:] = 1
    N = SplineBaseFunction(U)
    L = N[:, p](ubar)

    P = la.solve(L @ L.T, L @ y)
    Lplot = N[:, p](uplot)
    ysuposed = Lplot.T @ P

    print("Error = Sqrt(Sum (ysuposed - yplot)^2) = %.3e" % np.sqrt(np.sum((ysuposed-yplot)**2)))
    print("Error/nplot = %.3e" % (np.sqrt(np.sum((ysuposed-yplot)**2))/len(uplot)))

    plt.plot(uplot, yplot, ls="dotted", label="exact")
    plt.plot(uplot, ysuposed, ls="dashed", label="regression")
    plt.scatter(ubar, y)
    plt.legend()
    plt.show()
