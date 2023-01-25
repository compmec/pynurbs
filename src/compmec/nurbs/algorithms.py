import math
from typing import Any, Tuple

import numpy as np

from compmec.nurbs import SplineBaseFunction


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
    def CurvePntByCornerCut(
        npts: int,
        degree: int,
        knotvector: Tuple[float],
        Pw: Tuple[Point],
        knot: float,
    ) -> Point:
        """
        #### Algorithm A5.2 - NURBs book - pag 153
            Compute point on rational B-spline curve
        #### Input:
            ``npts``: int -- number of points
            ``degree``: int -- curver degree order
            ``U``: Tuple[float] -- knot vector
            ``Pw``: Tuple[Point] -- Control points
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
    def RefineKnotVectCurve(
        npts: int, degree: int, knotvector: Tuple[float], Pw: Tuple[Point], X, r
    ) -> Tuple:
        """
        #### Algorith A5.4 - NURBs book - pag 155
            Refine curve knot vector
        #### Input:
            ``npts``: int -- number of points
            ``degree``: int -- curver degree order
            ``U``: Tuple[float] -- knot vector
            ``Pw``: Tuple[Point] -- Control points
            ``X``: float -- evaluation knot
            ``r``: float -- evaluation knot
        #### Output:
            ``Ubar``: Tuple[float] -- New knot vector
            ``Qw``: Tuple[Point] -- New control points
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
    def DegreeReduceCurve(
        knotvector: Tuple[float], ctrlpoints: Tuple[Point], times: int
    ):
        return Chapter5.DegreeReduceCurve_nurbsbook(knotvector, ctrlpoints, times)

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
        ndivsubint = 1049
        all_knots = list(set(knotvector))
        all_knots.sort()
        pairs = list(zip(all_knots[:-1], all_knots[1:]))
        Nq = SplineBaseFunction(knotvector)
        Np = SplineBaseFunction(desknotvect)
        M = np.zeros((Np.npts, Np.npts), dtype="float64")
        F = np.zeros((Np.npts, Nq.npts), dtype="float64")
        for i, (a, b) in enumerate(pairs):
            u = np.linspace(a, b, ndivsubint).tolist()
            Nqu = Nq(u)
            Npu = Np(u)
            M += Npu @ Npu.T
            F += Npu @ Nqu.T
        M[0, 0] = 1
        M[0, 1:] = 0
        M[-1, :-1] = 0
        M[-1, -1] = 1
        F[0, 0] = 1
        F[0, 1:] = 0
        F[-1, :-1] = 0
        F[-1, -1] = 1
        A = np.linalg.solve(M, F)
        P = A @ np.array(ctrlpoints)
        return P
