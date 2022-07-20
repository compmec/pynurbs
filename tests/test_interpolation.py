import numpy as np
from matplotlib import pyplot as plt
from compmec.nurbs.interpolate import curve_spline, function_spline
from compmec.nurbs.curves import SplineXYFunction
from numpy import linalg as la


def test_functionspline_smallpolynomials():
    ntests = 10
    for p in range(1, 5):
        for i in range(ntests):
            a, b = 4*np.random.rand(2)
            a -= 2.1
            b += 2.1
            
            coefs = np.random.rand(p+1)

            nsample = 10
            nrefine = 100
            xsample = np.linspace(a, b, nsample)
            ysample = np.zeros(xsample.shape)
            xrefine = np.linspace(a, b, nrefine)
            yrefine = np.zeros(xrefine.shape)
            for j in range(p+1):
                ysample += coefs[j]*(xsample**j)
                yrefine += coefs[j]*(xrefine**j)

            F = function_spline(xsample, ysample, p=p)
            ysuposed = F(xrefine)

            L2y = la.norm(ysuposed - yrefine)
            assert L2y < 1e-9

def test_curvespline_pointsinterpolation():
    ntests = 1
    for p in range(1, 5):
        for i in range(ntests):
            nsample = 10
            qsample = np.linspace(0, 1, nsample)
            pointssample = np.random.rand(nsample, 3)
            
            F = curve_spline(p, [qsample], [pointssample])
            pointssuposed = F(qsample)

            
            L2y = la.norm(pointssuposed - pointssample)
            assert L2y < 1e-9
            qplot = np.linspace(0, 1, 129)
            plt.figure()
            plt.scatter(qsample, pointssample[:, 0], color="b", ls="dotted", label="sample x")
            plt.scatter(qsample, pointssample[:, 1], color="r", ls="dotted", label="sample y")
            plt.scatter(qsample, pointssample[:, 2], color="g", ls="dotted", label="sample z")
            plt.plot(qplot, F(qplot)[:, 0], color="b", label="interp x")
            plt.plot(qplot, F(qplot)[:, 1], color="r", label="interp y")
            plt.plot(qplot, F(qplot)[:, 2], color="g", label="interp z")
            plt.legend()
            plt.show()

def test_curvespline_derivativeinterpolation():
    ntests = 1
    for p in range(1, 5):
        for i in range(ntests):
            nsample = 10
            qsample = np.linspace(0, 1, nsample)
            pointssample = np.random.rand(nsample, 3)
            derivssample = np.random.rand(nsample, 3)
            
            F = curve_spline(p, [qsample, qsample], [pointssample, derivssample])
            dF = F.derivate()
            pointssuposed = F(qsample)
            derivssuposed = dF(qsample)

            
            L2pos = la.norm(pointssuposed - pointssample)
            L2der = la.norm(derivssuposed - derivssample)
            assert L2pos < 1e-9
            assert L2der < 1e-9
            qplot = np.linspace(0, 1, 129)
            plt.figure()
            plt.scatter(qsample, pointssample[:, 0], color="b", ls="dotted", label="sample x")
            plt.scatter(qsample, pointssample[:, 1], color="r", ls="dotted", label="sample y")
            plt.scatter(qsample, pointssample[:, 2], color="g", ls="dotted", label="sample z")
            plt.plot(qplot, F(qplot)[:, 0], color="b", label="interp x")
            plt.plot(qplot, F(qplot)[:, 1], color="r", label="interp y")
            plt.plot(qplot, F(qplot)[:, 2], color="g", label="interp z")
            plt.plot("Points")
            plt.legend()

            plt.figure()
            plt.scatter(qsample, derivssample[:, 0], color="b", ls="dotted", label="sample x")
            plt.scatter(qsample, derivssample[:, 1], color="r", ls="dotted", label="sample y")
            plt.scatter(qsample, derivssample[:, 2], color="g", ls="dotted", label="sample z")
            plt.plot(qplot, dF(qplot)[:, 0], color="b", label="interp x")
            plt.plot(qplot, dF(qplot)[:, 1], color="r", label="interp y")
            plt.plot(qplot, dF(qplot)[:, 2], color="g", label="interp z")
            plt.legend()
            plt.title("Derivates")
            plt.show()

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
    # test_functionspline_smallpolynomials()
    # test_curvespline_pointsinterpolation()
    test_curvespline_derivativeinterpolation()
    # test_functionXYsin()

if __name__ == "__main__":
    main()
