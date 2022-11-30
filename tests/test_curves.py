import math

import numpy as np
import pytest

from compmec.nurbs import KnotVector, RationalCurve, SplineCurve
from compmec.nurbs.curves import BaseCurve
from compmec.nurbs.knotspace import GeneratorKnotVector


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_basefunctions.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestSplineCurve:
    def create_random_knotvector(self, degree: int, npts: int):
        return GeneratorKnotVector.random(degree, npts)

    def create_random_controlpoints(self, npts: int, ndim: int):
        if ndim == 0:
            P = np.random.uniform(-1, 1, npts)
        else:
            P = np.random.uniform(-1, 1, (npts, ndim))
        return P

    def create_random_bspline(self, degree: int, npts: int, ndim: int):
        U = self.create_random_knotvector(degree, npts)
        P = self.create_random_controlpoints(npts, ndim)
        C = SplineCurve(U, P)
        return C

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestSplineCurve::test_begin"])
    def test_creation_scalar_curve(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 1, degree + 1):
                for i in range(ntests):
                    self.create_random_bspline(degree, npts, 0)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestSplineCurve::test_begin"])
    def test_creation_vectorial_curve(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 1, degree + 11):
                for ndim in range(1, 5):
                    for i in range(ntests):
                        self.create_random_bspline(degree, npts, ndim)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_creation_scalar_curve",
            "TestSplineCurve::test_creation_vectorial_curve",
        ]
    )
    def test_creationfail(self, ntests=10):
        for i in range(ntests):
            degree = np.random.randint(0, 5)
            npts = np.random.randint(degree + 1, degree + 11)
            ndim = np.random.randint(1, 4)
            U = self.create_random_knotvector(degree, npts)
            P = self.create_random_controlpoints(npts + 1, ndim)
            with pytest.raises(TypeError):
                SplineCurve(U, "asd")
            with pytest.raises(TypeError):
                SplineCurve(U, 1)
            with pytest.raises(ValueError):
                SplineCurve(U, P)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_creation_scalar_curve",
            "TestSplineCurve::test_creation_vectorial_curve",
        ]
    )
    def test_compare_two_curves(self, ntests=1):
        for i in range(ntests):
            degree = np.random.randint(0, 5)
            npts = np.random.randint(degree + 1, degree + 11)
            ndim = np.random.randint(0, 4)
            U = GeneratorKnotVector.random(degree, n=npts)
            if ndim == 0:
                P = np.random.uniform(-1, 1, npts)
                P3 = np.random.uniform(-1, 1, npts)
            else:
                P = np.random.uniform(-1, 1, (npts, ndim))
                P3 = np.random.uniform(-1, 1, (npts, ndim))
            C1 = SplineCurve(U, P)
            C2 = SplineCurve(U, P)
            C3 = SplineCurve(U, P3)
            assert id(C1) != id(C2)
            assert C1 == C2
            assert C1 != C3

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_creation_scalar_curve",
            "TestSplineCurve::test_creation_vectorial_curve",
        ]
    )
    def test_curve_is_callable(self):
        degree = np.random.randint(0, 5)
        npts = np.random.randint(degree + 1, degree + 11)
        U = GeneratorKnotVector.random(degree, n=npts)
        P = np.random.uniform(-1, 1, npts)
        C = SplineCurve(U, P)
        assert callable(C)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_curve_is_callable",
        ]
    )
    def test_shapetype_callscalar_scalarpoints(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 1, degree + 11):
                U = self.create_random_knotvector(degree, npts)
                P = self.create_random_controlpoints(npts, 0)
                C = SplineCurve(U, P)

                t = np.random.uniform(0, 1)
                Cval = C(t)
                assert type(Cval) == type(P[0])

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_curve_is_callable",
            "TestSplineCurve::test_shapetype_callscalar_scalarpoints",
        ]
    )
    def test_shapetype_callscalar_vectorialpoints(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 1, degree + 11):
                for ndim in range(1, 5):
                    for i in range(ntests):
                        U = self.create_random_knotvector(degree, npts)
                        P = self.create_random_controlpoints(npts, ndim)
                        C = SplineCurve(U, P)

                        t = np.random.uniform(0, 1)
                        Cval = C(t)
                        assert len(Cval) == ndim
                        assert type(Cval) == type(P[0])
                        assert type(Cval[0]) == type(P[0][0])

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_curve_is_callable",
        ]
    )
    def test_shapetype_callvectorial_scalarpoints(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 1, degree + 11):
                for i in range(ntests):
                    U = self.create_random_knotvector(degree, npts)
                    P = self.create_random_controlpoints(npts, 0)
                    C = SplineCurve(U, P)

                    nsample = np.random.randint(10, 129)
                    t = np.linspace(0, 1, nsample)
                    Cval = C(t)
                    assert len(Cval) == nsample
                    assert type(Cval[0]) == type(P[0])

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_curve_is_callable",
            "TestSplineCurve::test_shapetype_callvectorial_scalarpoints",
            "TestSplineCurve::test_shapetype_callscalar_vectorialpoints",
        ]
    )
    def test_shapetype_callvectorial_vectorialpoints(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 1, degree + 11):
                for ndim in range(1, 5):
                    for i in range(ntests):
                        U = self.create_random_knotvector(degree, npts)
                        P = self.create_random_controlpoints(npts, ndim)
                        C = SplineCurve(U, P)

                        nsample = np.random.randint(10, 129)
                        t = np.linspace(0, 1, nsample)
                        Cval = C(t)
                        assert len(Cval) == nsample
                        assert type(Cval[0]) == type(P[0])
                        assert np.array(Cval).shape == (nsample, ndim)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_curve_is_callable",
            "TestSplineCurve::test_shapetype_callscalar_scalarpoints",
            "TestSplineCurve::test_shapetype_callscalar_vectorialpoints",
            "TestSplineCurve::test_shapetype_callvectorial_scalarpoints",
            "TestSplineCurve::test_shapetype_callvectorial_vectorialpoints",
        ]
    )
    def test_bezier_curve_values(self):
        for degree in range(1, 5):
            for ndim in range(0, 5):
                npts = degree + 1
                ndim = np.random.randint(0, 4)
                U = self.create_random_knotvector(degree, npts)
                P = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(U, P)

                nsample = np.random.randint(10, 129)
                t = np.linspace(0, 1, nsample)
                Cval = C(t)
                assert len(Cval) == nsample
                assert type(Cval[0]) == type(P[0])
                if ndim == 0:
                    assert np.array(Cval).shape == (nsample,)
                else:
                    assert np.array(Cval).shape == (nsample, ndim)

                Cgood = np.zeros(np.array(Cval).shape)
                for i, ti in enumerate(t):
                    ti1 = 1 - ti
                    for j, Pj in enumerate(P):
                        Cgood[i] += (
                            math.comb(npts - 1, j)
                            * ti1 ** (npts - 1 - j)
                            * ti**j
                            * Pj
                        )
                np.testing.assert_almost_equal(Cval, Cgood)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_compare_two_curves",
        ]
    )
    def test_sum_and_diff_two_curves(self):
        for degree in range(1, 5):
            for npts in range(degree + 1, degree + 11):
                ndim = np.random.randint(0, 4)
                U = self.create_random_knotvector(degree, npts)
                P1 = self.create_random_controlpoints(npts, ndim)
                P2 = self.create_random_controlpoints(npts, ndim)
                C1 = SplineCurve(U, P1)
                C2 = SplineCurve(U, P2)
                Cs = SplineCurve(U, P1 + P2)
                Cd = SplineCurve(U, P1 - P2)
                assert (C1 + C2) == Cs
                assert (C1 - C2) == Cd

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_sum_and_diff_two_curves",
        ]
    )
    def test_sum_and_diff_two_curves_fail(self):
        degree = np.random.randint(1, 5)
        npts = np.random.randint(degree + 1, degree + 11)
        ndim = np.random.randint(0, 5)

        U1 = self.create_random_knotvector(degree, npts)
        P1 = self.create_random_controlpoints(npts, ndim)
        C1 = SplineCurve(U1, P1)

        U2 = self.create_random_knotvector(degree, npts + 1)
        P2 = self.create_random_controlpoints(npts + 1, ndim)
        C2 = SplineCurve(U2, P2)

        with pytest.raises(ValueError):
            C1 + C2
        with pytest.raises(ValueError):
            C1 - C2

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
        ]
    )
    def test_derivative(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
        ]
    )
    def test_knot_insert_known_case(self):
        Uorig = [0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5]  # Example 5.1 nurbs book
        Uorig = np.array(Uorig, dtype="float64") / 5  # degree = 3, n = 8
        Uorig = KnotVector(Uorig)
        degree, npts = 3, 8
        knot = 0.5
        ndim = 3
        P = np.random.uniform(-1, 1, (npts, ndim))
        Corig = SplineCurve(Uorig, P)

        Q = np.zeros((npts + 1, ndim), dtype="float64")
        Q[:3] = P[:3]
        Q[3] = (1 / 6) * P[2] + (5 / 6) * P[3]
        Q[4] = (1 / 2) * P[3] + (1 / 2) * P[4]
        Q[5] = (5 / 6) * P[4] + (1 / 6) * P[5]
        Q[6:] = P[5:]

        Uinse = list(Uorig)
        Uinse.insert(6, knot)
        Uinse = KnotVector(Uinse)
        Cinse = SplineCurve(Uinse, Q)

        assert Corig == Cinse

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_knot_insert_known_case",
        ]
    )
    def test_knot_insert_random(self, ntests=10):
        for i in range(ntests):
            degree = np.random.randint(1, 5)
            npts = np.random.randint(degree + 1, degree + 11)
            ndim = np.random.randint(0, 5)
            U = self.create_random_knotvector(degree, npts)
            P = self.create_random_controlpoints(npts, ndim)
            C = SplineCurve(U, P)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_knot_insert_random",
        ]
    )
    def test_knot_remove(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_knot_insert_random",
            "TestSplineCurve::test_knot_remove",
        ]
    )
    def test_knot_insert_remove_once_random(self, ntests=1):
        for degree in range(1, 6):
            for i in range(ntests):
                npts = np.random.randint(degree + 1, degree + 11)
                ndim = np.random.randint(0, 5)
                U = self.create_random_knotvector(degree, npts)
                P = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(U, P)

                knot = np.random.uniform(0, 1)
                C.knot_insert(knot)
                C.knot_remove(knot)

                assert C == SplineCurve(U, P)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.skip(reason="Needs implementation")
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
        ]
    )
    def test_degree_increase(self, ntests=10):
        for degree in range(1, 6):
            for i in range(ntests):
                npts = np.random.randint(degree + 1, degree + 11)
                ndim = np.random.randint(0, 5)
                U = self.create_random_knotvector(degree, npts)
                P = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(U, P)
                C.degree_increase()
                assert C.degree == (degree + 1)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=["TestSplineCurve::test_begin", "TestSplineCurve::test_degree_increase"]
    )
    def test_degree_decrease(self, ntests=10):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_degree_increase",
            "TestSplineCurve::test_degree_decrease",
        ]
    )
    def test_degree_increase_decrease(self, ntests=10):
        for degree in range(1, 6):
            for i in range(ntests):
                npts = np.random.randint(degree + 1, degree + 11)
                ndim = np.random.randint(0, 5)
                times = np.random.randint(1, 5)
                U = self.create_random_knotvector(degree, npts)
                P = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(U, P)
                C.degree_increase(times)
                C.degree_decrease(times)
                assert C.degree == degree

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_sum_and_diff_two_curves",
        ]
    )
    def test_somefails(self):
        degree = np.random.randint(1, 5)
        npts = np.random.randint(degree + 1, degree + 11)
        U = GeneratorKnotVector.random(degree, n=npts)
        P = np.random.uniform(-1, 1, npts)
        C = SplineCurve(U, P)
        with pytest.raises(TypeError):
            C == 1
        with pytest.raises(TypeError):
            C + 1

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_creation_scalar_curve",
            "TestSplineCurve::test_creation_vectorial_curve",
            "TestSplineCurve::test_compare_two_curves",
            "TestSplineCurve::test_curve_is_callable",
            "TestSplineCurve::test_shapetype_callscalar_scalarpoints",
            "TestSplineCurve::test_shapetype_callscalar_vectorialpoints",
            "TestSplineCurve::test_shapetype_callvectorial_scalarpoints",
            "TestSplineCurve::test_shapetype_callvectorial_vectorialpoints",
            "TestSplineCurve::test_bezier_curve_values",
            "TestSplineCurve::test_sum_and_diff_two_curves",
            "TestSplineCurve::test_sum_and_diff_two_curves_fail",
            "TestSplineCurve::test_knot_insert_random",
            "TestSplineCurve::test_knot_remove",
            "TestSplineCurve::test_knot_insert_remove_random",
            "TestSplineCurve::test_somefails",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestSplineCurve::test_end",
    ]
)
def test_end():
    pass
