import numpy as np
from Bspline import compute_np

def getU_uniform(n, p):
    if n < p:
        n = p
    U = np.linspace(0, 1, n - p + 1)
    U = np.concatenate((np.zeros(p), U))
    U = np.concatenate((U, np.ones(p)))
    return U

def getU_random(n, p):
    if n < p:
        n = p
    hs = np.random.random(n - p)
    hs *= 1.2
    hs[hs> 1] = 0
    U = np.cumsum(hs)
    if U[0] == 0:
        U += 0.1*np.random.random(1)
    
    U /= U[-1]*(1+0.4*np.random.random(1))
    
    U = np.concatenate((np.zeros(p+1), U))
    U = np.concatenate((U, np.ones(p+1)))
    return U


def getU_plot(U, number_divisions=128):
    zero = 1e-11
    n, p = compute_np(U)
    newU = np.linspace(U[p] + zero, U[p + 1] - zero, number_divisions + 1)
    for iiii in range(p + 1, n):
        new_interval = np.linspace(
            U[iiii] + zero, U[iiii + 1] - zero, number_divisions + 1)
        newU = np.concatenate((newU, new_interval))
    return newU


if __name__ == "__main__":
    U = getU_uniform(n=5, p=2)
    U = getU_random(n=5, p=2)
    uplot = getU_plot(U)
    print(U)

