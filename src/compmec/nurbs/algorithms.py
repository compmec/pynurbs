import math
from typing import Any, List, Tuple


class Point:
    pass


class Array1D(Tuple):
    pass


class Array2D(Tuple):
    pass


class Chapter1:
    @staticmethod
    def Horner1(a: Array1D[Point], npts: int, u0: float) -> Point:
        """
        #### Algorithm A1.1
        Compute point on power basis curve
        #### Input:
            ``a``: Array1D[Point] -- coefficients to power
            ``npts``: int -- Degree max, len(a) = n+1
            ``u0``: float -- Evaluation knot
        #### Output:
            ``C``: Point -- Evaluated point
        """
        n = npts - 1
        C = a[n]
        for i in range(n - 1, -1, -1):
            C = C * u0 + a[i]
        return C

    @staticmethod
    def Bernstein(i: int, npts: int, u: float) -> float:
        """
        #### Algorithm A1.2
        Compute the value of a Bernstein polynomial
        ##### Input:
            ``i``: int --
            ``npts``: int --
            ``u``: float --
        ##### Output:
            ``B``: float --
        """
        n = npts - 1
        Chapter1.Bernstein()
        temp = [0] * (n + 1)
        temp[n - i] = 1
        u1 = 1 - u
        for k in range(1, n + 1):
            for j in range(n, k - 1, -1):
                temp[j] = u1 * temp[j] + u * temp[j - 1]
        return temp[n]

    @staticmethod
    def AllBernstein(npts: int, u: float) -> Array1D[float]:
        """
        #### Algorithm A1.3
        Compute all nth-degree Bernstein polynomials
        #### Input:
            ``npts``: int --
            ``u``: float --
        #### Output:
            ``B``: Array1D[float] --
        """
        n = npts - 1
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
    def PointOnBezierCurve(P: Array1D[Point], npts: int, u: float) -> Point:
        """
        #### Algorithm A1.4
        Compute point on Bezier curve
        #### Input:
            ``P``: Array1D[Point] --
            ``npts``: int --
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
    def deCasteljau1(P: Array1D[Point], npts: int, u: float) -> Point:
        """
        #### Algorithm A1.5
        Compute point on a Bezier curve using deCasteljau
        #### Input:
            ``P``: Array1D[Point] --
            ``npts``: int --
            ``u``: float --
        #### Output:
            ``C``: Point --
        """
        n = npts - 1
        Q = [0] * len(P)
        for i in range(n + 1):  # Use local array so we do not destroy control points
            Q[i] = P[i]
        for k in range(1, n + 1):
            for i in range(n - k + 1):
                Q[i] = (1 - u) * Q[i] + u * Q[i + 1]
        return Q[0]

    @staticmethod
    def Horner2(a: Array2D[Point], npts: int, m: int, u0: float, v0: float) -> Point:
        """
        #### Algorithm A1.6
        Compute point on a power basis surface
        #### Input:
            ``a``: Array2D[Point]
            ``npts``: int
            ``m``: int
            ``u0``: float
            ``v0``: float
        #### Output:
            ``S``: Point
        """
        n = npts - 1
        b = [0] * (n + 1)
        for i in range(n + 1):
            b[i] = Chapter1.Horner1(a[i], m, v0)
        return Chapter1.Horner1(b, n, u0)

    @staticmethod
    def deCasteljau2(
        a: Array2D[Point], npts: int, m: int, u0: float, v0: float
    ) -> Point:
        """
        #### Algorithm A1.7
            Description
        #### Input:
            ``a``: Array2D[Point]
            ``npts``: int
            ``m``: int
            ``u0``: float
            ``v0``: float
        #### Output:
            ``S``: Point
        """
        n = npts - 1
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
    def FindSpan(npts: int, degree: int, u: float, U: Array1D[float]) -> int:
        """
        #### Algorithm A2.1
        Determine the knot span index
        #### Input:
            ``npts``: int -- number of DOFs
            ``degree``: int -- degree
            ``u``: float -- knot value
            ``U``: Array1D[float] -- knot vector
        #### Output:
            ``index``: int -- The span index
        """
        n = npts - 1
        p = degree
        if u == U[n + 1]:  # Special case
            return n + 1
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
    def FindSpanMult(npts: int, degree: int, u: float, U: Array1D[float]) -> int:
        """
        #### Algorithm A2.1
        Determine the knot span index
        #### Input:
            ``npts``: int -- number of DOFs
            ``degree``: int -- degree
            ``u``: float -- knot value
            ``U``: Array1D[float] -- knot vector
        #### Output:
            ``k``: int -- The span index
            ``s``: int -- Multiplicity of the knot
        """
        k = Chapter2.FindSpan(npts, degree, u, U)
        s = 0
        for i, ui in enumerate(U):
            if ui == u:
                s += 1
        return k, s

    @staticmethod
    def BasisFuns(i: int, u: float, degree: int, U: Array1D[float]) -> Array1D[float]:
        """
        #### Algorithm A2.2 - NURBs book - pag 68
        Compute the nonvanishing basis function
        #### Input:
            ``i``: int -- index of function
            ``u``: float -- knot value
            ``degree``: int -- degree
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
        i: int, u: float, degree: int, npts: int, U: Array1D[float]
    ) -> Array2D[float]:
        """
        #### Algorithm A2.3 - NURBs book - pag 72
        Compute nonzero basis functions and their derivatives.
        First section is A2.2 modified to store functions and knot differences.
        #### Input:
            ``i``: int -- index of function
            ``u``: float -- knot value
            ``degree``: int -- degree
            ``npts``: int -- number of dofs
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
    def OneBasisFun(degree: int, m: int, U: Array1D[float], i: int, u: float) -> float:
        """
        #### Algorithm A2.4 - NURBs book - pag 74
        Compute the basis function Nip
        #### Input:
            ``degree``: int -- degree
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
        degree: int, m: int, U: Array1D[float], i: int, u: float, npts: int
    ) -> Array1D[float]:
        """
        #### Algorithm A2.5 - NURBs book - pag 76
        Compute derivatives of basis function Nip
        #### Input:
            ``degree``: int -- degree
            ``m``: int -- size of knot vector: len(U) = m+1
            ``U``: Array1D[float] -- knot vector
            ``i``: int -- index of which function desired
            ``u``: float -- knot to evaluate
            ``npts``: int -- the maximum value for derivative, <= p
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


class Chapter3:
    @staticmethod
    def CurvePoint(
        npts: int, degree: int, U: Array1D[float], P: Array1D[Point], u: float
    ) -> Point:
        """
        #### Algorithm A3.1 - NURBs book - pag 82
            Compute curve point
        #### Input:
            ``npts``: int -- number of dofs
            ``degree``: int -- degree
            ``U``: Array1D[float] -- knot vector
            ``P``: Array1D[Point] -- control points
            ``u``: float -- knot to evaluate
        #### Output:
            ``C``: Point -- evaluated point
        """
        span = Chapter2.FindSpan(npts, degree, u, U)
        N = Chapter2.BasisFuns(span, u, degree, U)
        C = 0
        for i in range(p + 1):
            C = C + N[i] * P[span - degree + i]
        return C

    @staticmethod
    def CurveDerivsAlg1(
        npts: int, degree: int, U: Array1D[float], P: Array1D[Point], u: float, d: int
    ) -> Point:
        """
        #### Algorithm A3.2 - NURBs book - pag 93
            Compute curve derivatives
        #### Input:
            ``npts``: int -- number of control points + 1
            ``degree``: int -- degree
            ``U``: Array1D[float] -- knot vector
            ``P``: Array1D[Point] -- control points
            ``u``: float -- knot to evaluate
        #### Output:
            ``CC``: Array1D[Points] -- Derivatives of the curve
        """
        n = npts - 1
        p = degree
        CK = [0] * (d + 1)
        du = min(d, p)
        span = Chapter2.FindSpan(npts, degree, u, U)
        nders = Chapter2.DersBasisFuns(span, u, p, du, U)
        for k in range(du + 1):
            for j in range(p + 1):
                CK[k] = CK[k] + nders[k][j] * P[span - p + j]
        return CK

    @staticmethod
    def CurveDerivCpts(
        npts: int,
        degree: int,
        U: Array1D[float],
        P: Array1D[Point],
        d: int,
        r1: int,
        r2: int,
    ) -> Array2D[Point]:
        """
        #### Algorithm A3.3 - NURBs book - pag 98
            Compute contorl points of curve derivatives
        #### Input:
            ``npts``: int -- number of control points + 1
            ``degree``: int -- degree
            ``U``: Array1D[float] -- knot vector
            ``P``: Array1D[Point] -- control points
            ``d``: int -- The maximum number of derivatives, 0 <= k <= d
            ``r1``: int -- The lower bound of derivative: r1 <= i <= r2 - k
            ``r2``: int -- The upper bound of derivative: r1 <= i <= r2 - k
        #### Output:
            ``PK``: Array2D[Point] -- PK[k][i] is the i-th control point of the k-th derivative
        """
        PK = [[0] * (d + 1)] * (r + 1)
        r = r2 - r1
        for i in range(r + 1):
            PK[0][i] = P[r1 + i]
        for k in range(1, d + 1):
            temp = p - k + 1
            for i in range(r - k + 1):
                PK[k][i] = (
                    temp
                    * (PK[k - 1][i + 1] - PK[k - 1][i])
                    / (U[r1 + i + p + 1] - U[r1 + i + k])
                )
        return PK

    @staticmethod
    def CurveDerivsAlg2(
        npts: int, degree: int, U: Array1D[float], P: Array1D[Point], u: float, d: int
    ) -> Array2D[Point]:
        """
        #### Algorithm A3.4 - NURBs book - pag 99
            Compute curve derivatives
        #### Input:
            ``npts``: int -- number of control points + 1
            ``degree``: int -- degree
            ``U``: Array1D[float] -- knot vector
            ``P``: Array1D[Point] -- control points
            ``u``: float -- knot to evaluate
            ``d``: int -- The maximum number of derivatives, 0 <= k <= d
        #### Output:
            ``CK``:
        """
        n = npts - 1
        p = degree
        du = min(d, p)
        CK = [0] * (d + 1)
        span = Chapter2.FindSpan(npts, degree, u, U)
        N = Chapter2.AllBasisFuns(span, u, degree, U, N)
        PK = Chapter3.CurveDerivCpts(npts, degree, U, P, du, span - p, span)
        for k in range(du + 1):
            for j in range(p - k + 1):
                CK[k] = CK[k] + N[j][p - k] * PK[k][j]
        return CK

    @staticmethod
    def SurfacePoint(
        npts: int,
        degree: int,
        U: Array1D[float],
        m: int,
        q: int,
        V: Array1D[float],
        P: Array2D[float],
        u: float,
        v: float,
    ) -> Point:
        """
        #### Algorithm A3.5 - NURBs book - pag 103
            Compute surface point
        #### Input:
            ``npts``: int -- Number of control points in first coordinate
            ``degree``: int -- Degree of curve in first coordinate
            ``U``: Array1D[float] -- knot vector of first coordinate
            ``m``: int -- Number of control points in first coordinate
            ``q``: int -- Degree of curve in second coordinate
            ``V``: Array1D[float] -- knot vector of second coordinate
            ``P``: Array2D[Point] -- control points
            ``u``: float -- knot to evaluate at first coordinate
            ``v``: float -- knot to evaluate at second coordinate
        #### Output:
            ``S``: Point -- Evaluated point
        """
        pass

    @staticmethod
    def SurfaceDerivsAlg1(
        npts: int,
        degree: int,
        U: Array1D[float],
        m: int,
        q: int,
        V: Array1D[float],
        P: Array2D[float],
        u: float,
        v: float,
        d: int,
    ) -> Point:
        """
        #### Algorithm A3.6 - NURBs book - pag 111
            Compute surface point
        #### Input:
            ``npts``: int -- Number of control points in first coordinate
            ``degree``: int -- Degree of curve in first coordinate
            ``U``: Array1D[float] -- knot vector of first coordinate
            ``m``: int -- Number of control points in first coordinate
            ``q``: int -- Degree of curve in second coordinate
            ``V``: Array1D[float] -- knot vector of second coordinate
            ``P``: Array2D[Point] -- control points
            ``u``: float -- knot to evaluate at first coordinate
            ``v``: float -- knot to evaluate at second coordinate
        #### Output:
            ``SKL``: Array2D[float] -- SKL[k][l] is the derivative of S(u, v) with respect to u k times and v l times
        """
        pass

    @staticmethod
    def SurfaceDerivCpts():
        """
        #### Algorithm A3.7 - NURBs book - pag 114
        """
        pass

    @staticmethod
    def SurfaceDerivsAlg2():
        """
        #### Algorithm A3.8 - NURBs book - pag 115
        """
        pass


class Chapter4:
    @staticmethod
    def CurvePoint(
        npts: int, degree: int, U: Array1D[float], Pw: Array1D[Point], u: float
    ) -> Point:
        """
        #### Algorithm A4.1 - NURBs book - pag 124
        """
        pass

    @staticmethod
    def RatCurveDerivs(
        Aders: Array1D[float], wders: Array1D[float], d: int
    ) -> Array1D[Point]:
        """
        #### Algorithm A4.2 - NURBs book - pag 127
        Compute C(u) derivatives from Cw(u) derivatives
        """
        pass

    @staticmethod
    def SurfacePoint(
        npts: int,
        degree: int,
        U: Array1D[float],
        m: int,
        q: int,
        V: Array1D[float],
        Pw: Array2D[Point],
        u: float,
        v: float,
    ) -> Point:
        """
        #### Algorithm A4.3 - NURBs book - pag 134
            Compute point on rational B-spline surface
        """
        pass

    @staticmethod
    def RatSurfaceDerivs(Aders: Array2D[float], wders: Array2D[float], d: int):
        """
        #### Algorithm A4.4 - NURBs book - pag 137
            Compute S(u, v) derivatives from Sw(u, v) derivatives
        """
        pass


class Chapter5:
    @staticmethod
    def Distance4D(P1, P2):
        value = 0
        if isinstance(P1, float):
            P1 = [P1]
            P2 = [P2]
        for p1, p2 in zip(P1, P2):
            value += (p1 - p2) ** 2
        return math.sqrt(value)

    @staticmethod
    def CurveKnotIns(
        npts: int,
        degree: int,
        UP: Array1D[float],
        Pw: Array1D[Point],
        u: float,
        k: int,
        s: int,
        r: int,
    ) -> Tuple:
        """
        #### Algorithm A5.1 - NURBs book - pag 151
            Compute new curve from knot insertion
        #### Input:
            ``npts``: int -- number of points before insertion
            ``degree``: int -- curver degree order
            ``UP``: Array1D[float] -- knot vector after knot insertion
            ``Pw``: Array1D[Point] -- Control points before knot insertion
            ``u``: float -- knot to be inserted
            ``k``: int -- span where U_k <= u < U_{k+1}
            ``s``: int -- multiplicity of knot 'u'
            ``r``: int -- number of insertions of u
        #### Output:
            ``nq``: int -- number of points after insertion
            ``UQ``: Array1D[float] -- knot vector after knot insertion
            ``Qw``: Array1D[Point] -- Control points after knot insertion
        """
        np = npts - 1
        p = degree
        if r + s > p:
            raise ValueError(f"r + s > p : {r} + {s} > {p}")
        mp = np + p + 1
        nq = np + r
        mq = nq + p + 1

        UQ = [0] * (mq + 1)
        Qw = [0] * (nq + 1)
        Rw = [0] * (p + 1)
        # Load new knot vector
        UQ[: k + 1] = UP[: k + 1]
        UQ[k + 1 : k + r + 1] = [u] * r
        UQ[r + k + 1 :] = UP[k + 1 :]
        # Save unaltered control points
        Qw[: k - p + 1] = Pw[: k - p + 1]  # begin points
        Qw[r + k - s :] = Pw[k - s :]  # end points
        for i in range(p - s + 1):
            Rw[i] = Pw[k - p + i]
        for j in range(1, r + 1):  # Insert the knot r times
            L = k - p + j
            for i in range(p - j - s + 1):
                alpha = (u - UP[L + i]) / (UP[i + k + 1] - UP[L + i])
                Rw[i] = alpha * Rw[i + 1] + (1 - alpha) * Rw[i]
            Qw[L] = Rw[0]
            Qw[k + r - j - s] = Rw[p - j - s]
        for i in range(L + 1, k - s):
            Qw[i] = Rw[i - L]
        return nq + 1, UQ, Qw

    @staticmethod
    def CurvePntByCornerCut(
        npts: int, degree: int, U: Array1D[float], Pw: Array1D[Point], u: float
    ) -> Point:
        """
        #### Algorithm A5.2 - NURBs book - pag 153
            Compute point on rational B-spline curve
        #### Input:
            ``npts``: int -- number of points
            ``degree``: int -- curver degree order
            ``U``: Array1D[float] -- knot vector
            ``Pw``: Array1D[Point] -- Control points
            ``u``: float -- evaluation knot
        #### Output:
            ``C``: Point -- Evaluated point
        """
        n = npts - 1
        p = degree
        if u == U[0]:
            return Pw[0] / w
        if u == U[n + p + 1]:
            return Pw[n] / w
        Rw = [None] * (11 * (n + p + 1))
        k, s = Chapter2.FindSpanMult(n, p, u, U)  # General case
        r = p - s
        for i in range(r + 1):
            Rw[i] = Pw[k - p + i]
        for j in range(1, r + 1):
            for i in range(r - j + 1):
                alfa = (u - U[k - p + j + i]) / (U[i + k + 1] - U[k - p + j + i])
                Rw[i] = alfa * Rw[i + 1] + (1 - alfa) * Rw[i]
        return Rw[0] / w

    @staticmethod
    def SurfaceKnotIns():
        """
        #### Algorith A5.3 - NURBs book - pag 155
        """
        pass

    @staticmethod
    def RefineKnotVectCurve(
        npts: int, degree: int, U: Array1D[float], Pw: Array1D[Point], X, r
    ) -> Tuple:
        """
        #### Algorith A5.4 - NURBs book - pag 155
            Refine curve knot vector
        #### Input:
            ``npts``: int -- number of points
            ``degree``: int -- curver degree order
            ``U``: Array1D[float] -- knot vector
            ``Pw``: Array1D[Point] -- Control points
            ``X``: float -- evaluation knot
            ``r``: float -- evaluation knot
        #### Output:
            ``Ubar``: Array1D[float] -- New knot vector
            ``Qw``: Array1D[Point] -- New control points
        """
        n = npts - 1
        p = degree
        m = n + p + 1
        Ubar = [None] * (11 * m)

        a = Chapter2.FindSpan(npts, degree, X[0], U)
        b = Chapter2.FindSpan(npts, degree, X[r], U)
        b = b + 1
        Qw = [0] * ()
        for j in range(a - p + 1):
            Qw[j] = Pw[j]
        for j in range(b - 1, n + 1):
            Qw[j + r + 1] = Pw[j]
        for j in range(a + 1):
            Ubar[j] = U[j]
        for j in range(b + p, m + 1):
            Ubar[j + r + 1] = U[j]
        i = b + p - 1
        k = b + p + r
        for j in range(r, -1, -1):
            while (X[j] <= U[i]) and (i > a):
                Qw[k - p - 1] = Pw[i - p - 1]
                Ubar[k] = U[i]
                k = k - 1
                i = i - 1
            Qw[k - p - 1] = Qw[k - p]
            for l in range(1, p + 1):
                ind = k - p + l
                alfa = Ubar[k + l] - X[j]
                if abs(alfa) == 0:
                    Qw[ind - 1] = Qw[ind]
                else:
                    alfa = alfa / (Ubar[k + l] - U[i - p + l])
                    Qw[ind - 1] = alfa * Qw[ind - 1] + (1 - alfa) * Qw[ind]
            Ubar[k] = X[j]
            k = k - 1

    @staticmethod
    def RefineKnotVectSurface():
        """
        #### Algorith A5.5 - NURBs book - pag 167
            Refine surface knot vector
        """
        pass

    @staticmethod
    def DecomposeCurve():
        """
        #### Algorith A5.6 - NURBs book - pag 173
            Decompose curve into bezier segments
        """
        pass

    @staticmethod
    def DecomposeSurface():
        """
        #### Algorith A5.7 - NURBs book - pag 173
            Decompose surface into bezier patches
        """
        pass

    @staticmethod
    def RemoveCurveKnot(
        npts: int,
        degree: int,
        U: Array1D[float],
        Pw: Array1D[Point],
        u: float,
        r: int,
        s: int,
        num: int,
    ):
        """
        #### Algorith A5.8 - NURBs book - pag 185
            Remove knot u (index r) num times.
        #### Input:
            ``npts``: int -- Number of control points = n+1
            ``degree``: int -- curver degree order
            ``U``: Array1D[float] -- knot vector
            ``Pw``: Array1D[Point] -- Control points
            ``u``: float -- The knot to remove
            ``r``: int -- span of knot to remove
            ``s``: int -- Multiplicity of the knot
            ``num``: int -- Number of times to remove the knot
        #### Output:
            ``t``: int -- indicator how many points took out: 0 <= t <= num
            ``Un``: Array1D[float] -- New knot vector
            ``Pw``: Array1D[Point] -- New control points
        """
        TOLERANCE = 1e-7
        n = npts - 1
        p = degree
        m = n + p + 1  # Careful. In original algorithm there's an +1 that shouldn't be
        ord = p + 1
        fout = (2 * r - s - p) // 2
        last = r - s
        first = r - p
        temp = [0] * len(Pw)
        t = 0
        while t < num:  # This loop is Eq. (5.28)
            off = first - 1  # Diff in index between temp and P
            temp[0] = Pw[off]
            temp[last + 1 - off] = Pw[last + 1]
            i = first
            j = last
            ii = 1
            jj = last - off
            remflag = 0
            while j - i > t:  # Compute new control points for one removal step
                alfi = (u - U[i]) / (U[i + ord + t] - U[i])
                alfj = (u - U[j - t]) / (U[j + ord] - U[j - t])
                temp[ii] = (Pw[i] - (1 - alfi) * temp[ii - 1]) / alfi
                temp[jj] = (Pw[j] - alfj * temp[jj + 1]) / (1 - alfj)
                i += 1
                ii += 1
                j -= 1
                jj -= 1
            if j - i < t:  # Check if knot removable
                distance = Chapter5.Distance4D(temp[ii - 1], temp[jj + 1])
                if distance < TOLERANCE:
                    remflag = 1
            else:
                alfi = (u - U[i]) / (U[i + ord + t] - U[i])
                second_point = alfi * temp[ii + t + 1] + (1 - alfi) * temp[ii - 1]
                distance = Chapter5.Distance4D(Pw[i], second_point)
                if distance < TOLERANCE:
                    remflag = 1
            if remflag == 0:  # Cannot remove any more knots
                break
            else:  # Successful removal. Save new cont. pts.
                i = first
                j = last
                while j - i > t:
                    Pw[i] = temp[i - off]
                    Pw[j] = temp[j - off]
                    i += 1
                    j -= 1
            first -= 1
            last += 1
            t += 1
        if t == 0:
            return t, U, Pw

        Uq = [0] * (len(U) - t)
        Qw = [0] * (len(Pw) - t)
        for k in range(r + 1):
            Uq[k] = U[k]
        for k in range(r, len(Uq)):
            Uq[k] = U[k + t]  # Shift knots

        j = fout
        i = j  # Pj thru Pi will be overwritten
        for k in range(1, t):
            if k % 2 == 1:  # k modulo 2
                i += 1
            else:
                j -= 1
        for k in range(i + 1):
            Qw[k] = Pw[k]
        for k in range(i + 1, len(Pw)):  # Shift
            Qw[j] = Pw[k]
            j += 1
        return t, Uq, Qw

    @staticmethod
    def DegreeElevateCurve(
        npts: int, degree: int, U: Array1D[float], Pw: Array1D[Point], t: int
    ):
        """
        #### Algorith A5.9 - NURBs book - pag 206
            Degree elevate a curve t times
        #### Input:
            ``npts``: int -- Number of control points = n+1
            ``degree``: int -- curver degree order
            ``U``: Array1D[float] -- knot vector
            ``Pw``: Array1D[Point] -- Control points
            ``t``: int -- Number of times to increase degree
        #### Output:
            ``nh``: int --
            ``Uh``: Array1D[float] -- New knot vector
            ``Qw``: Array1D[Point] -- New control points
        """
        p = degree
        n = npts - 1
        m = n + p + 1
        ph = p + t
        ph2 = ph // 2
        # Compute Bezier degree elevation coefficients
        bezalfs = [[None] * (ph + 1)] * (ph + 1)
        Qw = [None] * (11 * m)
        Uh = [None] * (11 * m)
        Nextbpts = [None] * (11 * m)
        bpts = [None] * (11 * m)
        ebpts = [None] * (11 * m)
        alfs = [None] * (11 * m)

        bezalfs[0][0] = 1
        bezalfs[ph][0] = 1
        for i in range(1, ph2 + 1):
            inv = 1 / math.comb(ph, i)
            mpi = min(p, i)
            for j in range(max(0, i - t), mpi + 1):
                bezalfs[i][j] = inv * math.comb(p, j) * math.comb(t, i - j)
        for i in range(ph2 + 1, ph):
            mpi = min(p, i)
            for j in range(max(0, i - t), mpi + 1):
                bezalfs[i][j] = bezalfs[ph - i][p - j]
        mh = ph
        kind = ph + 1
        r = -1
        a = p
        b = p + 1
        cind = 1
        ua = U[0]
        Qw[0] = Pw[0]
        for i in range(ph + 1):
            Uh[i] = ua
        for i in range(p + 1):
            bpts[i] = Pw[i]
        while b < m:  # big loop thru knot vector
            i = b
            while True:
                b += 1
                if b == m:
                    break
                if U[b] != U[b + 1]:
                    break
            mul = b - i + 1
            mh = mh + mul + t
            ub = U[b]
            oldr = r
            r = p - mul

            # Insert knot u(b) r times
            lbz = 1 if oldr <= 0 else (oldr // 2) + 1
            rbz = ph if r <= 0 else ph - (r + 1) // 2
            if r > 0:  # Insert knot to get bezier segment
                numer = ub - ua
                for k in range(p, mul, -1):
                    alfs[k - mul - 1] = numer / (U[a + k] - ua)
                for j in range(1, r + 1):
                    save = r - j
                    s = mul + j
                    for k in range(p, s - 1, -1):
                        bpts[k] = (
                            alfs[k - s] * bpts[k] + (1 - alfs[k - s]) * bpts[k - 1]
                        )
                    Nextbpts[save] = bpts[p]
            for i in range(lbz, ph + 1):  # Degree elevate bezier
                ebpts[i] = 0
                mpi = min(p, i)
                for j in range(max(0, i - t), mpi + 1):
                    ebpts[i] = ebpts[i] + bezalfs[i][j] * bpts[j]
            if oldr > 1:  # Must remove knot u = U[a] oldr times
                first = kind - 2
                last = kind
                den = ub - ua
                bet = (ub - Uh[kind - 1]) / den
                for tr in range(1, oldr):
                    i = first
                    j = last
                    kj = j - kind + 1
                    while j - i > tr:  # Loop and compute the new
                        if i < cind:
                            alf = (ub - Uh[i]) / (ua - Uh[i])
                            Qw[i] = alf * Qw[i] + (1 - alf) * Qw[i - 1]
                        if j >= lbz:
                            if j - tr <= kind - ph + oldr:
                                gam = (ub - Uh[j - tr]) / den
                                ebpts[kj] = gam * ebpts[kj] + (1 - gam) * ebpts[kj + 1]
                            else:
                                ebpts[kj] = bet * ebpts[kj] + (1 - bet) * ebpts[kj + 1]
                        i += 1
                        j -= 1
                        kj -= 1
                    first -= 1
                    last += 1
            if a != p:  # Load the knot ua
                for i in range(ph - oldr):
                    Uh[kind] = ua
                    kind += 1
            for j in range(lbz, rbz + 1):  # Load ctrl pts into Qw
                Qw[cind] = ebpts[j]
                cind += 1
            if b < m:  # Set up for next pass thru loop
                for j in range(r):
                    bpts[j] = Nextbpts[j]
                for j in range(r, p + 1):
                    bpts[j] = Pw[b - p + j]
                a = b
                b += 1
                ua = ub
            else:
                for i in range(ph + 1):
                    Uh[kind + i] = ub
        nh = mh - ph - 1
        return nh, Uh, Qw

    @staticmethod
    def DegreeElevateSurface():
        """
        #### Algorith A5.10 - NURBs book - pag 206
            Degree elevate a surface t times
        """
        pass

    @staticmethod
    def DegreeReduceCurve():
        """
        #### Algorith A5.11 - NURBs book - pag 223
            Degree reduce a curve from p to p-1
        """
        pass
