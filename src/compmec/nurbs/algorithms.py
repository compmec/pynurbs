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
            Description
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


class Chapter2:
    @staticmethod
    def FindSpan(n: int, p: int, u: float, U: Array1D[float]) -> int:
        """
        #### Algorithm A2.1
        Determine the knot span index
        #### Input:
            ``n``: int -- number of DOFs
            ``p``: int -- degree
            ``u``: float -- knot value
            ``U``: Array1D[float] -- knot vector
        #### Output:
            ``index``: int -- The span index
        """
        if u == U[n + 1]:  # Special case
            return n
        low, high = p, n + 1  # Do binary search
        mid = (low + high) // 2
        while True:
            if u < U[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
            if U[mid] <= u < U[mid + 1]:
                return mid

    @staticmethod
    def BasisFuns(i: int, u: float, p: int, U: Array1D[float]) -> Array1D:
        """
        #### Algorithm A2.2 - NURBs book - pag 68
        Compute the nonvanishing basis function
        #### Input:
            ``i``: int -- index of function
            ``u``: float -- knot value
            ``p``: int -- degree
            ``U``: Array1D[float] -- knot vector
        #### Output:
            ``N``: Array1D[float] --
        """
        left = [0] * (p + 1)
        right = [0] * (p + 1)
        N = [0] * (p + 1)
        N[0] = 1
        for j in range(1, p + 1):
            left[j] = u - U[i + 1 - j]
            right[j] = U[i + j] - u
            saved = 0
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N[j] = saved
        return N

    @staticmethod
    def DersBasisFuns(
        i: int, u: float, p: int, n: int, U: Array1D[float]
    ) -> Array2D[float]:
        """
        #### Algorithm A2.3 - NURBs book - pag 72
        Compute nonzero basis functions and their derivatives.
        First section is A2.2 modified to store functions and knot differences.
        #### Input:
            ``i``: int -- index of function
            ``u``: float -- knot value
            ``p``: int -- degree
            ``n``: int -- number of dofs
            ``U``: Array1D[float] -- knot vector
        #### Output:
            ``ders``: Array1D[float] --
        """
        ndu = [[0] * (p + 1)] * (p + 1)
        ders = [[0] * (p + 1)] * (p + 1)
        a = [[0] * (p + 1)] * (p + 1)
        left = [0] * (p + 1)
        right = [0] * (p + 1)
        ndu[0, 0] = 1
        for j in range(1, p + 1):
            left[j] = u - U[i + 1 - j]
            right[j] = U[i + j] - u
            saved = 0
            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]  # Lower triangle
                temp = ndu[r, j - 1] / ndu[j, r]
                ndu[r][j] = saved + right[r + 1] * temp  # Upper triangle
                saved = left[j - r] * temp
            ndu[j][j] = saved
        for j in range(p + 1):  # Load the basis functions
            ders[0][j] = ndu[j][p]
        for r in range(p + 1):  # Loop over function index
            s1, s2 = 0, 1
            a[0][0] = 0
            for k in range(1, n + 1):  # Loop to compute kth derivative
                d = 0
                rk, pk = r - k, p - k
                if r >= k:
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                    d = a[s2][0] * ndu[rk][pk]
                if rk >= -1:
                    j1 = 1
                else:
                    j1 = -rk
                if r - 1 < pk:
                    j2 = k - 1
                    j2 = p - r
                for j in range(j1, j2 + 1):
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                    d += a[s2][j] * ndu[rk + j][pk]
                if r <= pk:
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                    d += a[s2][k] * ndu[r][pk]
                ders[k][r] = d
                j = s1
                s1 = s2
                s2 = j
        r = p
        for k in range(1, n + 1):
            for j in range(p + 1):
                ders[k][j] *= r
            r *= p - k
        return ders

    @staticmethod
    def OneBasisFun(p: int, m: int, U: Array1D[float], i: int, u: float) -> float:
        """
        #### Algorithm A2.4 - NURBs book - pag 74
        Compute the basis function Nip
        #### Input:
            ``p``: int -- degree
            ``m``: int -- size of knot vector: len(U) = m+1
            ``U``: Array1D[float] -- knot vector
            ``i``: int -- index of which function desired
            ``u``: float -- knot to evaluate
        #### Output:
            ``Nip``: float --  the result of function N_{ip}(u)
        """
        if i == 0 and u == U[0]:  # Special case
            return 1
        if i == m - p - 1 and u == U[m]:  # Special case
            return 1
        if u < U[i]:  # Local property
            return 0
        if u >= U[i + p + 1]:
            return 0
        N = [0] * (p + 1)
        for j in range(p + 1):  # Initialize zeroth-degree functs
            if u > U[i + j] and u < U[i + j + 1]:
                N[j] = 1
            else:
                N[j] = 0
        for k in range(1, p + 1):  # Compute triangular table
            if N[0] == 0:
                saved = 0
            else:
                saved = (u - U[i]) * N[0] / (U[i + k] - U[i])
            for j in range(p - k + 1):
                Uleft = U[i + j + 1]
                Uright = U[i + j + k + 1]
                if N[j + 1] == 0:
                    N[j] = saved
                    saved = 0
                else:
                    temp = N[j + 1] / (Uright - Uleft)
                    N[j] = saved + (Uright - u) * temp
                    saved = (u - Uleft) * temp
        return N[0]

    @staticmethod
    def DersOneBasisFun(
        p: int, m: int, U: Array1D[float], i: int, u: float, n: int
    ) -> Array1D[float]:
        """
        #### Algorithm A2.5 - NURBs book - pag 76
        Compute derivatives of basis function Nip
        #### Input:
            ``p``: int -- degree
            ``m``: int -- size of knot vector: len(U) = m+1
            ``U``: Array1D[float] -- knot vector
            ``i``: int -- index of which function desired
            ``u``: float -- knot to evaluate
            ``n``: int -- the maximum value for derivative, <= p
        #### Output:
            ``ders``: Array1D[float] -- The result of N_{ip}^{(k)}(u), k=0, 1, ..., n
        """
        ders = [0] * (n + 1)
        N = [[0] * (p + 1)] * (p + 1)
        if u < U[i]:  # Local property
            return ders
        if U[i + p + 1] <= u:
            return ders
        for j in range(0, p + 1):  # Initialize zeroth-degree functs
            if U[i + j] <= u < U[i + j + 1]:
                N[j][0] = 1
            else:
                N[j][0] = 0
        for k in range(1, p + 1):  # Compute full triangular table
            if N[0][k - 1] == 0:
                saved = 0
            else:
                saved = (u - U[i]) * N[0][k - 1] / (U[i + k] - U[i])
            for j in range(p - k + 1):
                Uleft = U[i + j + 1]
                Uright = U[i + j + k + 1]
                if N[j + 1][k - 1] == 0:
                    N[j][k] = saved
                    saved = 0
                else:
                    temp = N[j + 1][k - 1] / (Uright - Uleft)
                    N[j][k] = saved + (Uright - u) * temp
                    saved = (u - Uleft) * temp
        ders[0] = N[0][p]  # The function value
        ND = [0] * (p + 1)
        for k in range(1, n + 1):  # Compute the derivatives
            for j in range(k + 1):  # Load appropriate column
                ND[j] = N[j][p - k]
            for jj in range(1, k + 1):
                if ND[0] == 0:
                    saved = 0
                else:
                    saved = ND[0] / (U[i + p - k + jj] - U[i])
                for j in range(k - jj + 1):
                    Uleft = U[i + j + 1]
                    Uright = U[i + j + p + jj + 1]
                    if ND[j + 1] == 0:
                        ND[j] = (p - k + jj) * saved
                        saved = 0
                    else:
                        temp = ND[j + 1] / (Uright - Uleft)
                        ND[j] = (p - k + jj) * (saved - temp)
                        saved = temp
            ders[k] = ND[0]
        return ders
