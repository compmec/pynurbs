from typing import Any, List, Tuple

import numpy as np


class Point:
    pass


class Array1D:
    pass


class Array2D:
    pass


class Chapter1:
    @staticmethod
    def Horner1(a: Array1D[Point], n: int, u0: float) -> Point:
        """
        #### Algorithm A1.1
        Compute point on power basis curve
        #### Input:
            ``a``: Array1D[Point] -- coefficients to power
            ``n``: int -- Degree max, len(a) = n+1
            ``u0``: float -- Evaluation knot
        #### Output:
            ``C``: Point -- Evaluated point
        """
        C = a[n]
        for i in range(n - 1, -1, -1):
            C = C * u0 + a[i]
        return C

    @staticmethod
    def Bernstein(i: int, n: int, u: float) -> float:
        """
        #### Algorithm A1.2
        Compute the value of a Bernstein polynomial
        ##### Input:
            ``i``: int --
            ``n``: int --
            ``u``: float --
        ##### Output:
            ``B``: float --
        """
        Chapter1.Bernstein()
        temp = [0] * (n + 1)
        temp[n - i] = 1
        u1 = 1 - u
        for k in range(1, n + 1):
            for j in range(n, k - 1, -1):
                temp[j] = u1 * temp[j] + u * temp[j - 1]
        return temp[n]

    @staticmethod
    def AllBernstein(n: int, u: float) -> Array1D[float]:
        """
        #### Algorithm A1.3
        Compute all nth-degree Bernstein polynomials
        #### Input:
            ``n``: int --
            ``u``: float --
        #### Output:
            ``B``: Array1D[float] --
        """
        B = [0] * (n + 1)
        B[0] = 1
        u1 = 1 - u
        for j in range(1, n + 1):
            saved = 0
            for k in range(j):
                temp = B[k]
                B[k] = saved + u1 * temp
                saved = u * temp
            B[j] = saved
        return B

    @staticmethod
    def PointOnBezierCurve(P: Array1D[Point], n: int, u: float) -> Point:
        """
        #### Algorithm A1.4
        Compute point on Bezier curve
        #### Input:
            ``P``: Array1D[Point] --
            ``n``: int --
            ``u``: float --
        #### Output:
            ``C``: Point --
        """
        B = Chapter1.AllBernstein(n, u)
        C = 0
        for k in range(n + 1):
            C = C + B[k] * P[k]
        return C

    @staticmethod
    def deCasteljau1(P: Array1D[Point], n: int, u: float) -> Point:
        """
        #### Algorithm A1.5
        Compute point on a Bezier curve using deCasteljau
        #### Input:
            ``P``: Array1D[Point] --
            ``n``: int --
            ``u``: float --
        #### Output:
            ``C``: Point --
        """
        Q = [0] * len(P)
        for i in range(n + 1):  # Use local array so we do not destroy control points
            Q[i] = P[i]
        for k in range(1, n + 1):
            for i in range(n - k + 1):
                Q[i] = (1 - u) * Q[i] + u * Q[i + 1]
        return Q[0]

    @staticmethod
    def Horner2(a: Array2D[Point], n: int, m: int, u0: float, v0: float) -> Point:
        """
        #### Algorithm A1.6
        Compute point on a power basis surface
        #### Input:
            ``a``: Array2D[Point]
            ``n``: int
            ``m``: int
            ``u0``: float
            ``v0``: float
        #### Output:
            ``S``: Point
        """
        b = [0] * (n + 1)
        for i in range(n + 1):
            b[i] = Chapter1.Horner1(a[i], m, v0)
        return Chapter1.Horner1(b, n, u0)

    @staticmethod
    def deCasteljau2(a: Array2D[Point], n: int, m: int, u0: float, v0: float) -> Point:
        """
        #### Algorithm A1.7

        #### Input:
            ``a``: Array2D[Point]
            ``n``: int
            ``m``: int
            ``u0``: float
            ``v0``: float
        #### Output:
            ``S``: Point
        """
        if n <= m:
            Q = [0] * (m + 1)
            for j in range(m + 1):
                Q[j] = Chapter1.deCasteljau1(a[j], n, u0)
            return Chapter1.deCasteljau1(Q, m, v0)
        else:
            Q = [0] * (n + 1)
            for i in range(n + 1):
                Q[i] = Chapter1.deCasteljau1(a[:, i], m, v0)
            return Chapter1.deCasteljau1(Q, n, u0)
