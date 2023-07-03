import math
from typing import Any, List, Tuple

import numpy as np

# from compmec.nurbs.functions import SplineFunction


def N(i: int, j: int, k: int, u: float, U: Tuple[float]) -> float:
    """
    Returns the value of N_{ij}(u) in the interval [u_{k}, u_{k+1}]
    Remember that N_{i, j}(u) = 0   if  ( u not in [U[i], U[i+j+1]] )
    """

    npts = KnotVector.find_npts(U)

    if k < i:
        return 0
    if j == 0:
        if i == k:
            return 1
        if i + 1 == npts and k == npts:
            return 1
        return 0
    if i + j < k:
        return 0

    if U[i] == U[i + j]:
        factor1 = 0
    else:
        factor1 = (u - U[i]) / (U[i + j] - U[i])

    if U[i + j + 1] == U[i + 1]:
        factor2 = 0
    else:
        factor2 = (U[i + j + 1] - u) / (U[i + j + 1] - U[i + 1])

    result = factor1 * N(i, j - 1, k, u, U)
    result += factor2 * N(i + 1, j - 1, k, u, U)
    return result


def R(i: int, j: int, k: int, u: float, U: Tuple[float], w: Tuple[float]) -> float:
    """
    Returns the value of R_{ij}(u) in the interval [u_{k}, u_{k+1}]
    """
    Niju = N(i, j, k, u, U)
    if Niju == 0:
        return 0
    npts = len(w)
    soma = 0
    for z in range(npts):
        soma += w[z] * N(z, j, k, u, U)
    return w[i] * Niju / soma


def binom(n: int, i: int):
    """
    Returns binomial (n, i)
    if n
    """
    if n == 0:
        return 1
    prod = 1
    if i == 0 or i == n:
        return 1
    for j in range(i):
        prod *= (n - j) / (i - j)
    if prod == 0:
        raise ValueError(f"(n, i) = ({n}, {i})")
    return int(prod)


class LeastSquare:
    """
    Given two hypotetic curves C0 and C1, which are associated
    with knotvectors U and V, and control points P and Q.
        C0(u) = sum_{i=0}^{n-1} N_{i}(u) * P_{i}
        C1(u) = sum_{i=0}^{m-1} M_{i}(u) * Q_{i}
    Then, this class has functions to return [T] and [E] such
        [Q] = [T] * [P]
        error = [P]^T * [E] * [P]
    Then, C1 keeps near to C1 by using galerkin projections.

    They minimizes the integral
        int_{a}^{b} abs(C0(u) - C1(u))^2 du
    The way it does it by using the norm of inner product:
        abs(X) = sqrt(< X, X >)
    Then finally it finds the matrix [A] and [B]
        [C] * [Q] = [B] * [P]
        [A]_{ij} = int_{0}^{1} < Ni(u), Nj(u) > du
        [B]_{ij} = int_{0}^{1} < Mi(u), Nj(u) > du
        [C]_{ij} = int_{0}^{1} < Mi(u), Mj(u) > du
        --> [T] = [C]^{-1} * [B]^T
        --> [E] = [A] - [B] * [T]
    """

    @staticmethod
    def chebyshev_nodes(npts: int, a: float = 0, b: float = 1) -> Tuple[float]:
        """
        Returns a list of floats which are spaced in [a, b]
        for integration purpose.
        """
        k = np.arange(0, npts)
        nodes = np.cos(np.pi * (2 * k + 1) / (2 * npts))
        nodes *= (b - a) / 2
        nodes += (a + b) / 2
        nodes.sort()
        return nodes

    @staticmethod
    def interpolator_matrix(npts: int) -> Tuple[float]:
        """
        This function helps computing the values of F_{i} such
            f(x) approx g(x) = sum_{i=0}^{z} F_{i} * B_{i, z}(x)
        Which B_{i, z}(x) is a bezier function
            B_{i, z}(x) = binom(z, i) * (1-x)^{i} * x^{z-i}
        Then, a linear system is obtained by setting f(x_j) = g(x_j)
        at (z+1) points, which are the chebyshev nodes:
            [B0(x0)   B1(x0)   ...   Bz(x0)][ F0 ]   [ f(x0) ]
            [B0(x1)   B1(x1)   ...   Bz(x1)][ F1 ]   [ f(x1) ]
            [B0(x2)   B1(x2)   ...   Bz(x2)][ F2 ] = [ f(x2) ]
            [  |        |       \      |   ][ |  ]   [   |   ]
            [B0(xz)   B1(xz)   ...   Bz(xz)][ Fz ]   [ f(xz) ]
        The return is the inverse of the matrix
        """
        z = npts - 1
        chebyshev0to1 = LeastSquare.chebyshev_nodes(npts)
        matrixbezier = np.zeros((npts, npts), dtype="float64")
        for i, xi in enumerate(chebyshev0to1):
            for j in range(z + 1):
                matrixbezier[i, j] = binom(z, j) * (1 - xi) ** (z - j) * (xi**j)
        return np.linalg.inv(matrixbezier)

    @staticmethod
    def integrator_array(npts: int) -> Tuple[float]:
        """
        This function helps computing the integral of f(x) at the
        interval [0, 1] by using a polynomial of degree ```npts```.
        It returns an 1D array [A] such
            int_{0}^{1} f(x) dx = [A] * [f]
        Which
            [f] = [f(x0)  f(x1)  ...  f(xz)]
        The points x0, x1, ... are the chebyshev nodes

        We suppose that
            int_{0}^{1} f(x) dx approx int_{0}^{1} g(x) dx
        Which g(x) is made by bezier curves
            g(x) = sum_{i=0}^{z} F_{i} * B_{i, z}(x)
            B_{i, z}(x) = binom(z, i) * (1-x)^{i} * x^{z-i}
        We see that
            int_{0}^{1} B_{i, z}(x) dx = 1/(z+1)
        Therefore
            int_{0}^{1} f(x) dx approx 1/(z+1) * sum F_{i}
        The values of F_{i} are gotten from ```interpolator_matrix```
        """
        matrix = LeastSquare.interpolator_matrix(npts)
        return np.einsum("ij->j", matrix) / npts

    @staticmethod
    def bezier(degree0: int, degree1: int) -> Tuple[np.ndarray]:
        """
        For bezier curves, the knotvector is not needed.
        Calling the degrees as p and q:
            N_{ip} = binom(p, i) * (1-u)^(p-i) * u^i
            M_{iq} = binom(q, i) * (1-u)^(q-i) * u^i
        """
        knotvect0 = (degree0 + 1) * [0] + (degree0 + 1) * [1]
        knotvect1 = (degree1 + 1) * [0] + (degree1 + 1) * [1]
        return LeastSquare.spline(knotvect0, knotvect1)

    @staticmethod
    def spline(knotvect0: Tuple[float], knotvect1: Tuple[float]) -> Tuple[np.ndarray]:
        """ """
        knotvect0 = tuple(knotvect0)
        knotvect1 = tuple(knotvect1)
        degree0 = KnotVector.find_degree(knotvect0)
        degree1 = KnotVector.find_degree(knotvect1)
        npts0 = KnotVector.find_npts(knotvect0)
        npts1 = KnotVector.find_npts(knotvect1)
        knotes0 = KnotVector.find_knots(knotvect0)
        knotes1 = KnotVector.find_knots(knotvect1)
        allknots = list(set(knotes0 + knotes1))
        allknots.sort()
        nptsinteg = degree0 + degree1 + 3  # Number integration points
        integrator = LeastSquare.integrator_array(nptsinteg)

        # Here we will store the values of N and M
        # for a certain interval [a, b], which will change
        Nvalues = np.empty((npts0, nptsinteg), dtype="float64")
        Mvalues = np.empty((npts1, nptsinteg), dtype="float64")

        A = np.zeros((npts0, npts0), dtype="float64")  # N*N
        B = np.zeros((npts0, npts1), dtype="float64")  # N*M
        C = np.zeros((npts1, npts1), dtype="float64")  # M*M
        for a, b in zip(allknots[:-1], allknots[1:]):
            chebynodes = LeastSquare.chebyshev_nodes(nptsinteg, a, b)
            # Integral of the functions in the interval [a, b]
            k0 = KnotVector.find_span(a, knotvect0)
            k1 = KnotVector.find_span(a, knotvect1)
            for i in range(npts0):
                for j, uj in enumerate(chebynodes):
                    Nvalues[i, j] = N(i, degree0, k0, uj, knotvect0)
            for i in range(npts1):
                for j, uj in enumerate(chebynodes):
                    Mvalues[i, j] = N(i, degree1, k1, uj, knotvect1)
            A += np.einsum("k,ik,jk->ij", integrator, Nvalues, Nvalues)
            B += np.einsum("k,ik,jk->ij", integrator, Nvalues, Mvalues)
            C += np.einsum("k,ik,jk->ij", integrator, Mvalues, Mvalues)

        Cinv = np.linalg.inv(C)
        T = Cinv @ B.T
        E = A - B @ T
        return T, E


class KnotVector:
    @staticmethod
    def find_degree(knotvector: Tuple[float]) -> int:
        degree = 0
        while knotvector[degree] == knotvector[degree + 1]:
            degree += 1
        return degree

    @staticmethod
    def find_npts(knotvector: Tuple[float]) -> int:
        degree = KnotVector.find_degree(knotvector)
        return len(knotvector) - degree - 1

    @staticmethod
    def find_span(node: float, knotvector: Tuple[float]) -> int:
        n = KnotVector.find_npts(knotvector) - 1
        degree = KnotVector.find_degree(knotvector)
        if node == knotvector[n + 1]:  # Special case
            return n + 1
        low, high = degree, n + 1  # Do binary search
        mid = (low + high) // 2
        while True:
            if node < knotvector[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
            if knotvector[mid] <= node < knotvector[mid + 1]:
                return mid

    @staticmethod
    def find_mult(node: float, knotvector: Tuple[float]) -> int:
        """
        #### Algorithm A2.1
            Determine the knot span index
        #### Input:
            ``npts``: int -- number of DOFs
            ``degree``: int -- degree
            ``u``: float -- knot value
            ``U``: Tuple[float] -- knot vector
        #### Output:
            ``s``: int -- Multiplicity of the knot
        """
        mult = 0
        for knot in knotvector:
            if abs(knot - node) < 1e-9:
                mult += 1
        return mult

    @staticmethod
    def find_knots(knotvector: Tuple[float]) -> Tuple[float]:
        """
        #### Algorithm A2.1
            Determine the knot span index
        #### Input:
            ``npts``: int -- number of DOFs
            ``degree``: int -- degree
            ``u``: float -- knot value
            ``U``: Tuple[float] -- knot vector
        #### Output:
            ``s``: int -- Multiplicity of the knot
        """
        knots = []
        for knot in knotvector:
            if knot not in knots:
                knots.append(knot)
        return knots

    def add_knot(node: float, knotvector: Tuple[float]):
        """
        Add the node inside the knotvector
        """
        span = KnotVector.find_span(node, knotvector)
        knotvector = list(knotvector)
        knotvector.insert(span, node)
        return tuple(knotvector)

    def split(knotvector: Tuple[float], nodes: Tuple[float]) -> List[Tuple[float]]:
        """
        It splits the knotvector at nodes.
        Example:
            >> U = [0, 0, 0.5, 1, 1]
            >> split(U, [0.5])
            [[0, 0, 0.5, 0.5],
             [0.5, 0.5, 1, 1]]
            >> split(U, [0.25])
            [[0, 0, 0.25, 0.25],
             [0.25, 0.25, 0.5, 1, 1]]
            >> split(U, [0.25, 0.75])
            [[0, 0, 0.25, 0.25],
             [0.25, 0.25, 0.5, 0.75, 0.75],
             [0.75, 0.75, 1, 1]]
        """
        retorno = []
        knotvector = np.array(knotvector)
        minu, maxu = min(knotvector), max(knotvector)
        nodes = list(nodes) + [minu, maxu]
        nodes = list(set(nodes))
        nodes.sort()
        degree = np.sum(knotvector == minu) - 1
        for a, b in zip(nodes[:-1], nodes[1:]):
            middle = list(knotvector[(a < knotvector) * (knotvector < b)])
            newknotvect = (degree + 1) * [a] + middle + (degree + 1) * [b]
            retorno.append(newknotvect)
        return retorno


class Bezier:
    @staticmethod
    def degree_increase(ctrlpoints: Tuple[Any], times: int) -> Tuple[Any]:
        """
        #### Algorithm to increase degree of bezier curve
            It uses Equations 5.41, 5.42, 5.45 and 5.46
        #### Input:
            ``ctrlpoints``: Tuple[Any] -- Old Control points, before increase
            ``times``: int -- Times to increase degree
        #### Output:
            ``ctrlpoints``: Tuple[Any] -- New control points, after increase
        """
        npts = len(ctrlpoints)
        degree = npts - 1
        newctrlpoints = [0] * (npts + times)
        newctrlpoints[0] = ctrlpoints[0]
        newctrlpoints[npts + times - 1] = ctrlpoints[npts - 1]
        for i in range(1, npts + times - 1):
            lower = max(0, i - times)
            upper = min(degree, i) + 1
            for j in range(lower, upper):
                coef = math.comb(degree, j) * math.comb(times, i - j)
                coef /= math.comb(degree + times, i)
                newctrlpoints[i] = newctrlpoints[i] + coef * ctrlpoints[j]
        return newctrlpoints

    @staticmethod
    def degree_decrease(
        ctrlpoints: Tuple[Any], times: int, tolerance: float = 1e-9
    ) -> Tuple[Any]:
        """
        #### Algorithm to increase degree of bezier curve
            It uses Equations 5.41, 5.42, 5.45 and 5.46
        #### Input:
            ``ctrlpoints``: Tuple[Any] -- Old Control points, before increase
            ``times``: int -- Times to increase degree
        #### Output:
            ``ctrlpoints``: Tuple[Any] -- New control points, after increase
        """
        raise NotImplementedError


class Point:
    pass


class Chapter2:
    @staticmethod
    def FindSpan(npts: int, degree: int, knot: float, knotvector: Tuple[float]) -> int:
        """
        #### Algorithm A2.1
        Determine the knot span index
        #### Input:
            ``npts``: int -- number of DOFs
            ``degree``: int -- degree
            ``u``: float -- knot value
            ``U``: Tuple[float] -- knot vector
        #### Output:
            ``index``: int -- The span index
        """
        u = knot
        U = knotvector
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
    def FindSpanMult(
        npts: int, degree: int, knot: float, knotvector: Tuple[float]
    ) -> Tuple[int, int]:
        """
        #### Algorithm A2.1
        Determine the knot span index
        #### Input:
            ``npts``: int -- number of DOFs
            ``degree``: int -- degree
            ``u``: float -- knot value
            ``U``: Tuple[float] -- knot vector
        #### Output:
            ``k``: int -- The span index
            ``s``: int -- Multiplicity of the knot
        """
        u = knot
        U = knotvector
        k = Chapter2.FindSpan(npts, degree, u, U)
        s = 0
        for i, ui in enumerate(U):
            if ui == u:
                s += 1
        return k, s


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
        knotvector: Tuple[float],
        ctrlpoints: Tuple[Point],
        knot: float,
        times: int,
    ) -> Tuple:
        """
        #### Algorithm A5.1 - NURBs book - pag 151
            Compute new curve from knot insertion
        #### Input:
            ``knotvector``: Tuple[float] -- knot vector after knot insertion
            ``ctrlpoints``: Tuple[Point] -- Control points before knot insertion
            ``knot``: float -- knot to be inserted
            ``times``: int -- number of insertions of u
        #### Output:
            ``UQ``: Tuple[float] -- knot vector after knot insertion
            ``Qw``: Tuple[Point] -- Control points after knot insertion
        """
        npts = len(ctrlpoints)
        degree = len(knotvector) - npts - 1
        r = times
        span, mult = Chapter2.FindSpanMult(npts, degree, knot, knotvector)
        k, s = span, mult
        Pw = ctrlpoints
        UP = knotvector
        u = knot
        np = npts - 1
        p = degree
        if r <= 0:
            return list(knotvector), list(ctrlpoints)
        if times + mult > degree:
            raise ValueError(f"times + mult > degree : {times} + {mult} > {degree}")
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
        return UQ, Qw

    @staticmethod
    def DecomposeCurve(knotvector: Tuple[float], ctrlpoints: Tuple[Point]):
        """
        #### Algorith A5.6 - NURBs book - pag 173
            Decompose curve into bezier segments
        """
        npts = len(ctrlpoints)
        degree = len(knotvector) - npts - 1
        U, Pw = knotvector, ctrlpoints
        n, p = npts - 1, degree
        m = n + p + 1
        a, b = p, p + 1
        nb = 0
        Qw = np.empty((m, degree + 1), dtype="object")
        alphas = [0] * (p + 1)
        for i in range(p + 1):
            Qw[nb, i] = Pw[i]
        while b < m:
            i = b
            while b < m:
                if U[b] != U[b + 1]:
                    break
                b += 1
            mult = b - i + 1
            if mult < p:
                numer = U[b] - U[a]  # Numerator of alpha
                # Compute and store alphas
                for j in range(p, mult, -1):
                    alphas[j - mult - 1] = numer / (U[a + j] - U[a])
                r = p - mult  # Insert knot r times
                for j in range(1, r + 1):
                    save = r - j
                    s = mult + j  # This many new points
                    for k in range(p, s - 1, -1):
                        alpha = alphas[k - s]
                        Qw[nb, k] = alpha * Qw[nb, k] + (1 - alpha) * Qw[nb, k - 1]
                    if b < m:  # Control point of next segment
                        Qw[nb + 1, save] = Qw[nb, p]
            nb += 1  # Bezier segment completed
            if b < m:  # Initialize for next segment
                for i in range(p - mult, p + 1):
                    Qw[nb, i] = Pw[b - p + i]
                a, b = b, b + 1
        return Qw[:nb].tolist()

    @staticmethod
    def RemoveCurveKnot(
        knotvector: Tuple[float],
        ctrlpoints: Tuple[Point],
        knot: float,
        times: int,
    ):
        """
        #### Algorith A5.8 - NURBs book - pag 185
            Remove knot u (index r) num times.
        #### Input:
            ``knotvector``: Tuple[float] -- knot vector
            ``ctrlpoints``: Tuple[Point] -- Control points
            ``knot``: float -- The knot to remove
            ``times``: int -- Number of times to remove the knot
        #### Output:
            ``t``: int -- indicator how many points took out: 0 <= t <= num
            ``Un``: Tuple[float] -- New knot vector
            ``Pw``: Tuple[Point] -- New control points
        """
        TOLERANCE = 1e-9
        npts = len(ctrlpoints)
        degree = len(knotvector) - npts - 1
        u = knot
        r, s = Chapter2.FindSpanMult(npts, degree, knot, knotvector)
        Pw = ctrlpoints
        num = times
        U = knotvector
        n = npts - 1
        p = degree
        m = n + p + 1
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
        Uq = np.array(Uq, dtype="float64").tolist()
        Qw = list(np.array(Qw, dtype="float64"))
        return t, Uq, Qw

    @staticmethod
    def DegreeElevateCurve(
        knotvector: Tuple[float], ctrlpoints: Tuple[Point], times: int
    ):
        """
        #### Algorith A5.9 - NURBs book - pag 206
            Degree elevate a curve t times
        #### Input:
            ``knotvector``: Tuple[float] -- knot vector
            ``ctrlpoints``: Tuple[Point] -- Control points
            ``times``: int -- Number of times to increase degree
        #### Output:
            ``Uh``: Tuple[float] -- New knot vector
            ``Qw``: Tuple[Point] -- New control points
        """

    @staticmethod
    def DegreeElevateCurve_nurbsbook(
        knotvector: Tuple[float], ctrlpoints: Tuple[Point], times: int
    ):
        """
        #### Algorith A5.9 - NURBs book - pag 206
            Degree elevate a curve t times
        #### Input:
            ``knotvector``: Tuple[float] -- knot vector
            ``ctrlpoints``: Tuple[Point] -- Control points
            ``times``: int -- Number of times to increase degree
        #### Output:
            ``Uh``: Tuple[float] -- New knot vector
            ``Qw``: Tuple[Point] -- New control points
        """
        ctrlpoints = list(np.array(ctrlpoints, dtype="float64"))
        knotvector = list(knotvector)
        npts = len(ctrlpoints)
        degree = len(knotvector) - npts - 1
        p = degree
        n = npts - 1
        t = times
        U = knotvector
        Pw = ctrlpoints
        # Init variables
        bezalfs = np.zeros((p + t + 1, p + 1))
        bpts = [0] * (p + 1)
        ebpts = [0] * (p + t + 1)
        Nextbpts = [0] * (p - 1)
        alfs = [0] * p
        Uh = [None] * (11 * npts)
        Qw = [None] * (11 * npts)

        # Init algorithm
        m = n + p + 1
        ph = p + t
        ph2 = ph // 2

        # Compute Bezier degree elevation coefficients
        bezalfs[0, 0] = 1.0
        bezalfs[ph, 0] = 1.0
        for i in range(1, ph2 + 1):
            inv = 1 / math.comb(ph, i)
            mpi = min(p, i)
            for j in range(max(0, i - t), mpi + 1):
                bezalfs[i, j] = inv * math.comb(p, j) * math.comb(t, i - j)
        for i in range(ph2 + 1, ph):
            mpi = min(p, i)
            for j in range(max(0, i - t), mpi + 1):
                bezalfs[i, j] = bezalfs[ph - i, p - j]
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
        # Initialize first Bezier seg
        for i in range(p + 1):
            bpts[i] = Pw[i]
        while b < m:  # big loop thru knot vector
            i = b
            while True:
                if b >= m:
                    break
                if U[b] != U[b + 1]:
                    break
                b += 1
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
                    ebpts[i] += bezalfs[i, j] * bpts[j]
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
        Uh = Uh[: mh + 1]
        nh = mh - ph - 1
        Qw = Qw[: nh + 1]
        try:
            Uh = np.array(Uh, dtype="float64")
            Qw = np.array(Qw, dtype="float64")
        except Exception as e:
            raise e
        return Uh, Qw

    @staticmethod
    def DegreeReduceCurve_nurbsbook(
        knotvector: Tuple[float], ctrlpoints: Tuple[Point], times: int
    ):
        """
        #### Algorith A5.11 - NURBs book - pag 223
            Degree reduce a curve from (degree) to (degree - times)
            Entry is not protected
        #### Input:
            ``knotvector``: Tuple[float] -- knot vector
            ``ctrlpoints``: Tuple[Point] -- Control points
            ``times``: Tuple[Point] -- Control points
        #### Output:
            ``Uh``: Tuple[float] -- New knot vector
            ``Pw``: Tuple[Point] -- New control points
        """
        assert times == 1
        TOLERANCE = 1e9
        ctrlpoints = list(np.array(ctrlpoints, dtype="float64"))
        knotvector = list(knotvector)
        npts = len(ctrlpoints)
        degree = len(knotvector) - npts - 1

        p = degree
        n = npts - 1
        m = n + p + 1
        U = list(knotvector)
        Qw = list(ctrlpoints)

        # Init vars
        bpts = [0] * (p + 1)
        Nextbpts = [0] * (p - 1)
        rbpts = [0] * p
        alphas = [0] * (p - 1)
        e = [0] * m
        Pw = [0] * npts  # It will be less than that
        Uh = [0] * (m + 1)  # It will be less than that

        # Init some variables
        ph = p - 1
        mh = ph
        kind = ph + 1
        r = -1
        a = p
        b = p + 1
        cind = 1
        mult = p
        m = n + p + 1
        Pw[0] = Qw[0]
        for i in range(ph + 1):  # Compute left end of knot vector
            Uh[i] = U[0]
        for i in range(p + 1):  # Initialize first Bezier segment
            bpts[i] = Qw[i]
        for i in range(m):  # Initialize error vector
            e[i] = 0.0
        # Loop through the knot vector
        while b < m:
            i = b
            while b < m:
                if U[b] != U[b + 1]:
                    break
                b += 1
            mult = b - i + 1
            mh = mh + mult - 1
            oldr = r
            r = p - mult
            # lbz = 1 + ((oldr // 2) if (oldr > 0) else 0)
            if oldr > 0:
                lbz = (oldr + 2) // 2
            else:
                lbz = 1
            # Insert knot U[b] r times
            if r > 0:
                numer = U[b] - U[a]
                for k in range(p, mult - 1, -1):
                    alphas[k - mult - 1] = numer / (U[a + k] - U[a])
                for j in range(1, r + 1):
                    save = r - j
                    s = mult + j
                    for k in range(p, s - 1, -1):
                        bpts[k] = alphas[k - s] * bpts[k]
                        bpts[k] += (1 - alphas[k - s]) * bpts[k - 1]
                    Nextbpts[save] = bpts[p]
            # Degree reduce bezier segment
            rbpts, MaxErr = Custom.BezDegreeReduce_nurbsbook(bpts)
            MaxErr = 0
            e[a] += MaxErr
            if e[a] > TOLERANCE:
                raise ValueError("Curve not degree reducible")
            # Remove knot U[a] oldr times
            if oldr > 0:
                first = kind
                last = kind
                for k in range(oldr):
                    i = first
                    j = last
                    kj = j - kind
                    while j - i > k:
                        alfa = (U[a] - Uh[i - 1]) / (U[b] - Uh[i - 1])
                        beta = (U[a] - Uh[j - k - 1]) / (U[b] - Uh[j - k - 1])
                        val = (Pw[i - 1] - (1 - alfa) * Pw[i - 2]) / alfa
                        Pw[i - 1] = val

                        rbpts[kj] = (rbpts[kj] - beta * rbpts[kj + 1]) / (1 - beta)
                        i += 1
                        j -= 1
                        kj -= 1
                    # Compute knot removal error bounds (Br)
                    if j - i < k:
                        Br = Chapter5.Distance4D(Pw[i - 2], rbpts[kj + 1])
                    else:
                        delta = (U[a] - Uh[i - 1]) / (U[b] - Uh[i - 1])
                        A = delta * rbpts[kj + 1] + (1 - delta) * Pw[i - 2]
                        Br = Chapter5.Distance4D(Pw[i - 1], A)
                    # Update the error vector
                    K = a + oldr - k
                    q = p + (1 - k) // 2
                    L = K - q
                    for ii in range(L, a + 1):  # These knot spans were affected
                        e[ii] += Br
                        if e[ii] > TOLERANCE:
                            pass  # raise ValueError
                        first -= 1
                        last += 1
                cind = i - 1
            # Load knot vector and control points
            if a != p:
                for i in range(ph - oldr):
                    Uh[kind] = U[a]
                    kind += 1
            for i in range(lbz, ph + 1):
                Pw[cind] = rbpts[i]
                cind += 1
            # Set up for next pass through
            if b < m:
                for i in range(r):
                    bpts[i] = Nextbpts[i]
                for i in range(r, p + 1):
                    bpts[i] = Qw[b - p + i]
                a = b
                b += 1
            else:
                for i in range(ph + 1):
                    Uh[kind + i] = U[b]

        Uh = Uh[: mh + 1]
        nh = mh - ph - 1
        Pw = Pw[: nh + 1]
        return Uh, Pw


class Custom:
    @staticmethod
    def BezDegreeIncrease(ctrlpoints: Tuple[Point], times: int):
        """
        #### Algorithm to increase degree of bezier curve
            It uses Equations 5.41, 5.42, 5.45 and 5.46
        #### Input:
            ``ctrlpoints``: Tuple[Point] -- Control points
            ``times``: int -- Times to increase degree
        #### Output:
            ``ctrlpoints``: Tuple[Point] -- New control points
        """
        npts = len(ctrlpoints)
        degree = npts - 1
        newctrlpoints = [0] * (npts + times)
        newctrlpoints[0] = ctrlpoints[0]
        newctrlpoints[npts + times - 1] = ctrlpoints[npts - 1]
        for i in range(1, npts + times - 1):
            lower = max(0, i - times)
            upper = min(degree, i) + 1
            for j in range(lower, upper):
                coef = math.comb(degree, j) * math.comb(times, i - j)
                coef /= math.comb(degree + times, i)
                newctrlpoints[i] = newctrlpoints[i] + coef * ctrlpoints[j]
        return newctrlpoints

    @staticmethod
    def FindMaximumDistanceBetweenBezier(Q: Tuple[Point], P: Tuple[Point]):
        degreeQ = len(Q) - 1
        degreeP = len(P) - 1
        us = np.linspace(0, 1, 129)
        maximum = 0
        for i, ui in enumerate(us):
            Cq, Cp = 0, 0
            ui1 = 1 - ui
            for j in range(degreeQ + 1):
                Cq += math.comb(degreeQ, j) * ui**j * ui1 ** (degreeQ - j) * Q[j]
            for j in range(degreeP + 1):
                Cp += math.comb(degreeP, j) * ui**j * ui1 ** (degreeP - j) * P[j]
            distance = Chapter5.Distance4D(Cp, Cq)
            if maximum < distance:
                maximum = distance
        return maximum

    @staticmethod
    def BezDegreeReduce(ctrlpoints: Tuple[Point], times: int):
        """
        #### Algorithm to reduce degree of bezier curve
            It's used in Alggorithm A5.11
            It finds the value of P[i], 0 < i < npts-1-times such
            it minimizes the integral
                I = int_0^1  abs(Ci(u) - Cd(u))^2 du
            Where Ci is the increased curve, and Cd the (wanted) decreased curve
                Ci = sum_{i=0}^{degree} B_{i,degree}(u) * Q[i]
                Cd = sum_{i=0}^{degree-times} B_{i,degree-times}(u) * P[i]
            We still have P[0] = Q[0] and P[degree-times] = Q[degree]

            The entries are not protected.
        #### Input:
            ``ctrlpoints``: Tuple[Point] -- Control points
            ``times``: int -- Number of times to reduce degree
        #### Output:
            ``ctrlpoints``: Tuple[Point] -- New control points
            ``MaxErr``: float -- Maximum error of bezier reduction
        """
        return Custom.BezDegreeReduce_leastsquare(ctrlpoints, times)

    @staticmethod
    def BezDegreeReduce_leastsquare(ctrlpoints: Tuple[Point], times: int):
        """
        #### Algorithm to reduce degree of bezier curve
            It's used in Alggorithm A5.11
            It finds the value of P[i], 0 < i < npts-2 such
            it minimizes the integral
                I = int_0^1  abs(Ci(u) - Cd(u))^2 du
            Where Ci is the increased curve, and Cd the (wanted) decreased curve
                Ci = sum_{i=0}^{degree} B_{i,degree}(u) * Q[i]
                Cd = sum_{i=0}^{degree-1} B_{i,degree-1}(u) * P[i]
            We still have P[0] = Q[0] and P[degree-1] = Q[degree]
        #### Input:
            ``ctrlpoints``: Tuple[Point] -- Control points
        #### Output:
            ``ctrlpoints``: Tuple[Point] -- New control points
            ``MaxErr``: float -- Maximum error of bezier reduction
        """

        Q = ctrlpoints
        npts = len(Q)
        degree = npts - 1
        p = degree
        t = times
        M = np.zeros((degree + 1 - times, degree + 1 - times), dtype="float64")
        K = np.zeros((degree + 1 - times, degree + 1), dtype="float64")
        for i in range(p + 1 - t):
            for j in range(p + 1 - t):
                M[i, j] = math.comb(p - t, i) * math.comb(p - t, j)
                M[i, j] /= (2 * (p - t) + 1) * math.comb(2 * (p - t), i + j)
            for j in range(p + 1):
                K[i, j] = math.comb(p - t, i) * math.comb(p, j)
                K[i, j] /= (2 * p + 1 - t) * math.comb(2 * p - t, i + j)
        M[0, 0] = 1
        M[0, 1:] = 0
        M[p - t, : p - t] = 0
        M[p - t, p - t] = 1
        K[0, 0] = 1
        K[0, 1:] = 0
        K[p - t, :p] = 0
        K[p - t, p] = 1
        A = np.linalg.solve(M, K)
        P = A @ Q
        P = np.array(P)
        error = Custom.FindMaximumDistanceBetweenBezier(Q, P)
        return P, error

    @staticmethod
    def BezDegreeReduce_nurbsbook(ctrlpoints: Tuple[Point]):
        """
        #### Algorithm to reduce degree of bezier curve
            It's used in Alggorithm A5.11
            It uses Equations 5.41, 5.42, 5.45 and 5.46
        #### Input:
            ``ctrlpoints``: Tuple[Point] -- Control points
        #### Output:
            ``ctrlpoints``: Tuple[Point] -- New control points
            ``MaxErr``: float -- Maximum error of bezier reduction
        """
        Q = ctrlpoints
        npts = len(ctrlpoints)
        degree = npts - 1
        P = [0] * degree

        P[0] = Q[0]
        P[degree - 1] = Q[degree]
        r = (degree - 1) // 2
        alpha = [i / (degree) for i in range(degree)]
        for i in range(1, r):
            P[i] = (Q[i] - alpha[i] * P[i - 1]) / (1 - alpha[i])
        for i in range(degree - 2, r, -1):
            val = (Q[i + 1] - (1 - alpha[i + 1]) * P[i + 1]) / alpha[i + 1]
            P[i] = val
        alpha = [i / (degree) for i in range(degree)]
        if degree % 2:  # degree is odd
            PrL = (Q[r] - alpha[r] * P[r - 1]) / (1 - alpha[r])
            PrR = (Q[r + 1] - (1 - alpha[r + 1]) * P[r + 1]) / alpha[r + 1]
            P[r] = 0.5 * (PrL + PrR)
            error = Chapter5.Distance4D(PrL, PrR)
        else:  # degree is even
            P[r] = (Q[r] - alpha[r] * P[r - 1]) / (1 - alpha[r])
            pointtocomputeerror = 0.5 * (P[r] + P[r + 1])
            error = Chapter5.Distance4D(P[r + 1], pointtocomputeerror)
        return P, error

    @staticmethod
    def UniteBezierCurvesSameDegree(all_knots: Tuple[float], allctrlpoints: Any):
        ncurves = len(allctrlpoints)
        degree = len(allctrlpoints[0]) - 1
        allctrlpoints = np.array(allctrlpoints)
        p = degree
        newknotvector = [0]
        for knot in all_knots:
            newknotvector += [knot] * p
        newknotvector += [1]
        finalnpts = len(newknotvector) - degree - 1
        ctrlpoints = [allctrlpoints[0, 0]] * finalnpts
        for i in range(ncurves):
            ctrlpoints[1 + i * p : 1 + (i + 1) * p] = allctrlpoints[i, 1:]
        return newknotvector, ctrlpoints

    @staticmethod
    def LeastSquareSpline(
        knotvector: Tuple[float],
        ctrlpoints: Tuple[Point],
        desknotvect: Tuple[float],
    ):
        """Takes time to compute, cause we integrate and solve system"""
        T, E = LeastSquare.spline(knotvector, desknotvect)
        return T @ ctrlpoints

    #     ndivsubint = 1049
    #     all_knots = list(set(knotvector))
    #     all_knots.sort()
    #     pairs = list(zip(all_knots[:-1], all_knots[1:]))
    #     Nq = SplineFunction(knotvector)
    #     Np = SplineFunction(desknotvect)
    #     M = np.zeros((Np.npts, Np.npts), dtype="float64")
    #     F = np.zeros((Np.npts, Nq.npts), dtype="float64")
    #     for i, (a, b) in enumerate(pairs):
    #         u = np.linspace(a, b, ndivsubint).tolist()
    #         Nqu = Nq(u)
    #         Npu = Np(u)
    #         M += Npu @ Npu.T
    #         F += Npu @ Nqu.T
    #     M[0, 0] = 1
    #     M[0, 1:] = 0
    #     M[-1, :-1] = 0
    #     M[-1, -1] = 1
    #     F[0, 0] = 1
    #     F[0, 1:] = 0
    #     F[-1, :-1] = 0
    #     F[-1, -1] = 1
    #     A = np.linalg.solve(M, F)
    #     P = A @ np.array(ctrlpoints)
    #     return P
