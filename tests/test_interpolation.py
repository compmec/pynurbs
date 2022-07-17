import numpy as np
from matplotlib import pyplot as plt
from compmec.nurbs.interpolate import curve_spline, function_spline
from compmec.nurbs.curves import SplineXYFunction
from numpy import linalg as la


def test_polynomial1():
    ntests = 10
    for i in range(ntests):
        a, b = 4*np.random.rand(2)
        a -= 2
        b += 2
        
        a0, a1 = np.random.rand(2)
        nsample = 10
        nrefine = 100
        xsample = np.linspace(a, b, nsample)
        ysample = a0 + a1 * xsample
        xrefine = np.linspace(a, b, nrefine)
        yrefine = a0 + a1 * xrefine

        F = function_spline(xsample, ysample, p=1)
        ysuposed = F(xrefine)

        L2y = la.norm(ysuposed - yrefine)
        assert L2y < 1e-9


def test_polynomial2():
    ntests = 10
    for i in range(ntests):
        a, b = 4*np.random.rand(2)
        a -= 2
        b += 2
        
        a0, a1, a2 = np.random.rand(3)
        nsample = 10
        nrefine = 100
        xsample = np.linspace(a, b, nsample)
        ysample = a0 + a1 * xsample + a2 * xsample**2
        xrefine = np.linspace(a, b, nrefine)
        yrefine = a0 + a1 * xrefine + a2 * xrefine**2

        F = function_spline(xsample, ysample, p=2)
        ysuposed = F(xrefine)

        L2y = la.norm(ysuposed - yrefine)
        assert L2y < 1e-9

def test_polynomial3():
    ntests = 10
    for i in range(ntests):
        a, b = 4*np.random.rand(2)
        a -= 2
        b += 2
        
        a0, a1, a2, a3 = np.random.rand(4)
        nsample = 10
        nrefine = 100
        xsample = np.linspace(a, b, nsample)
        ysample = a0 + a1 * xsample + a2 * xsample**2 + a3 * xsample**3
        xrefine = np.linspace(a, b, nrefine)
        yrefine = a0 + a1 * xrefine + a2 * xrefine**2 + a3 * xrefine**3

        F = function_spline(xsample, ysample, p=3)
        ysuposed = F(xrefine)

        L2y = la.norm(ysuposed - yrefine)
        assert L2y < 1e-9


def test_functionXYsin():
    a, b = 0, 2*np.pi
    f = np.sin

    p = 2
    nsample = 10
    nplot = 1024
    x = np.linspace(a, b, nsample)
    y = f(x)
    xplot = np.linspace(a, b, nplot)
    yplot = f(xplot)
    
    F = function_spline(x, y, p)
    ysuposed = F(xplot)

    L2y = la.norm(ysuposed - yplot)
    assert L2y < 0.2

def main():
    test_polynomial1()
    test_polynomial2()
    test_polynomial3()
    test_functionXYsin()

if __name__ == "__main__":
    main()
