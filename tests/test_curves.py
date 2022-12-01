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
            ctrlpoints = np.random.uniform(-1, 1, npts)
        else:
            ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
        return ctrlpoints

    def create_random_bspline(self, degree: int, npts: int, ndim: int):
        knotvector = self.create_random_knotvector(degree, npts)
        ctrlpoints = self.create_random_controlpoints(npts, ndim)
        C = SplineCurve(knotvector, ctrlpoints)
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
            for npts in range(degree + 1, degree + 11):
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
            knotvector = self.create_random_knotvector(degree, npts)
            ctrlpoints = self.create_random_controlpoints(npts + 1, ndim)
            with pytest.raises(TypeError):
                SplineCurve(knotvector, "asd")
            with pytest.raises(TypeError):
                SplineCurve(knotvector, 1)
            with pytest.raises(ValueError):
                SplineCurve(knotvector, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_creation_scalar_curve",
            "TestSplineCurve::test_creation_vectorial_curve",
        ]
    )
    def test_compare_two_curves(self, ntests=100):
        for i in range(ntests):
            degree = np.random.randint(0, 5)
            npts = np.random.randint(degree + 1, degree + 11)
            ndim = np.random.randint(0, 4)
            knotvector = self.create_random_knotvector(degree, npts)
            P1 = self.create_random_controlpoints(npts, ndim)
            P3 = self.create_random_controlpoints(npts, ndim)
            C1 = SplineCurve(knotvector, P1)
            C2 = SplineCurve(knotvector, P1)
            C3 = SplineCurve(knotvector, P3)
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
        knotvector = GeneratorKnotVector.random(degree, npts=npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        C = SplineCurve(knotvector, ctrlpoints)
        assert callable(C)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_creation_scalar_curve",
            "TestSplineCurve::test_creation_vectorial_curve",
        ]
    )
    def test_curve_attributes(self):
        degree = np.random.randint(0, 5)
        npts = np.random.randint(degree + 1, degree + 11)
        ndim = np.random.randint(0, 5)
        knotvector = self.create_random_knotvector(degree, npts)
        ctrlpoints = self.create_random_controlpoints(npts, ndim)
        C = SplineCurve(knotvector, ctrlpoints)
        assert C.degree == degree
        assert C.npts == npts
        assert C.knotvector == knotvector
        assert np.all(C.ctrlpoints == ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestSplineCurve::test_begin",
            "TestSplineCurve::test_curve_is_callable",
        ]
    )
    def test_shapetype_callscalar_scalarpoints(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 1, degree + 11):
                knotvector = self.create_random_knotvector(degree, npts)
                ctrlpoints = self.create_random_controlpoints(npts, 0)
                C = SplineCurve(knotvector, ctrlpoints)

                t = np.random.uniform(0, 1)
                Cval = C(t)
                assert type(Cval) == type(ctrlpoints[0])

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
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
                        knotvector = self.create_random_knotvector(degree, npts)
                        ctrlpoints = self.create_random_controlpoints(npts, ndim)
                        C = SplineCurve(knotvector, ctrlpoints)

                        t = np.random.uniform(0, 1)
                        Cval = C(t)
                        assert len(Cval) == ndim
                        assert type(Cval) == type(ctrlpoints[0])
                        assert type(Cval[0]) == type(ctrlpoints[0][0])

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
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
                    knotvector = self.create_random_knotvector(degree, npts)
                    ctrlpoints = self.create_random_controlpoints(npts, 0)
                    C = SplineCurve(knotvector, ctrlpoints)

                    nsample = np.random.randint(npts + degree + 2, npts + degree + 11)
                    t = np.linspace(0, 1, nsample)
                    Cval = C(t)
                    assert len(Cval) == nsample
                    assert type(Cval[0]) == type(ctrlpoints[0])

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
                        knotvector = self.create_random_knotvector(degree, npts)
                        ctrlpoints = self.create_random_controlpoints(npts, ndim)
                        C = SplineCurve(knotvector, ctrlpoints)

                        nsample = np.random.randint(
                            npts + degree + 2, npts + degree + 11
                        )
                        t = np.linspace(0, 1, nsample)
                        Cval = C(t)
                        assert len(Cval) == nsample
                        assert type(Cval[0]) == type(ctrlpoints[0])
                        assert np.array(Cval).shape == (nsample, ndim)

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
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
                knotvector = self.create_random_knotvector(degree, npts)
                ctrlpoints = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(knotvector, ctrlpoints)

                nsample = np.random.randint(npts + degree + 2, npts + degree + 11)
                t = np.linspace(0, 1, nsample)
                Cval = C(t)
                assert len(Cval) == nsample
                assert type(Cval[0]) == type(ctrlpoints[0])
                if ndim == 0:
                    assert np.array(Cval).shape == (nsample,)
                else:
                    assert np.array(Cval).shape == (nsample, ndim)

                Cgood = np.zeros(np.array(Cval).shape)
                for i, ti in enumerate(t):
                    ti1 = 1 - ti
                    for j, Pj in enumerate(ctrlpoints):
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
                knotvector = self.create_random_knotvector(degree, npts)
                P1 = self.create_random_controlpoints(npts, ndim)
                P2 = self.create_random_controlpoints(npts, ndim)
                C1 = SplineCurve(knotvector, P1)
                C2 = SplineCurve(knotvector, P2)
                Cs = SplineCurve(knotvector, P1 + P2)
                Cd = SplineCurve(knotvector, P1 - P2)
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
        Uorig = np.array(Uorig, dtype="float64") / 5  # degree = 3, npts = 8
        Uorig = KnotVector(Uorig)
        degree, npts = 3, 8
        knot = 0.5
        ndim = 3
        ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
        Corig = SplineCurve(Uorig, ctrlpoints)

        Q = np.zeros((npts + 1, ndim), dtype="float64")
        Q[:3] = ctrlpoints[:3]
        Q[3] = (1 / 6) * ctrlpoints[2] + (5 / 6) * ctrlpoints[3]
        Q[4] = (1 / 2) * ctrlpoints[3] + (1 / 2) * ctrlpoints[4]
        Q[5] = (5 / 6) * ctrlpoints[4] + (1 / 6) * ctrlpoints[5]
        Q[6:] = ctrlpoints[5:]

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
            knotvector = self.create_random_knotvector(degree, npts)
            ctrlpoints = self.create_random_controlpoints(npts, ndim)
            C = SplineCurve(knotvector, ctrlpoints)

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
                knotvector = self.create_random_knotvector(degree, npts)
                ctrlpoints = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(knotvector, ctrlpoints)

                knot = np.random.uniform(0, 1)
                C.knot_insert(knot)
                C.knot_remove(knot)

                assert C == SplineCurve(knotvector, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
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
                knotvector = self.create_random_knotvector(degree, npts)
                ctrlpoints = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(knotvector, ctrlpoints)
                C.degree_increase()
                assert C.degree == (degree + 1)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.skip(reason="Needs implementation")
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
                knotvector = self.create_random_knotvector(degree, npts)
                ctrlpoints = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(knotvector, ctrlpoints)
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
        knotvector = GeneratorKnotVector.random(degree, npts=npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        C = SplineCurve(knotvector, ctrlpoints)
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
