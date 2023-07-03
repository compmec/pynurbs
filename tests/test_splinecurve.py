import math

import numpy as np
import pytest

from compmec.nurbs.curves import RationalCurve, SplineCurve
from compmec.nurbs.knotspace import GeneratorKnotVector, KnotVector


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_functions.py::test_end",
        "tests/test_beziercurve.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestInitSplineCurve:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestInitSplineCurve::test_begin"])
    def test_build_scalar(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        SplineCurve(knotvector, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestInitSplineCurve::test_begin"])
    def test_build_vectorial(self):
        degree, npts = 3, 9
        ndim = 3
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
        SplineCurve(knotvector, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestInitSplineCurve::test_build_scalar"])
    def test_failbuild(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.uniform(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts + 1)
        with pytest.raises(ValueError):
            SplineCurve(knotvector, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestInitSplineCurve::test_build_scalar"])
    def test_attributes(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.uniform(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = SplineCurve(knotvector, ctrlpoints)
        curve = SplineCurve(knotvector, ctrlpoints)
        # Attributes
        assert hasattr(curve, "degree")
        assert hasattr(curve, "npts")
        assert hasattr(curve, "ctrlpoints")
        assert hasattr(curve, "knotvector")
        assert hasattr(curve, "knots")

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestInitSplineCurve::test_build_scalar"])
    def test_functions(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.uniform(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = SplineCurve(knotvector, ctrlpoints)
        curve = SplineCurve(knotvector, ctrlpoints)
        # Functions
        assert hasattr(curve, "deepcopy")
        assert hasattr(curve, "split")
        assert hasattr(curve, "degree_increase")
        assert hasattr(curve, "degree_decrease")
        assert hasattr(curve, "degree_clean")
        assert hasattr(curve, "knot_insert")
        assert hasattr(curve, "knot_remove")
        assert hasattr(curve, "knot_clean")
        assert callable(curve)

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestInitSplineCurve::test_attributes"])
    def test_atributesgood(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.uniform(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = SplineCurve(knotvector, ctrlpoints)
        assert curve.degree == degree
        assert curve.npts == npts
        assert curve.knotvector == knotvector
        np.testing.assert_allclose(curve.ctrlpoints, ctrlpoints)
        assert curve.knots == knotvector.knots

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestInitSplineCurve::test_build_vectorial"])
    def test_compare_two_curves(self):
        degree = np.random.randint(1, 5)
        npts = np.random.randint(degree + 2, degree + 11)
        ndim = np.random.randint(1, 4)
        knotvector = GeneratorKnotVector.random(degree, npts)
        P1 = np.random.uniform(-2, -1, (npts, ndim))
        P4 = np.random.uniform(1, 2, (npts, ndim))
        C1 = SplineCurve(knotvector, P1)
        C2 = SplineCurve(knotvector, P1)
        C3 = C1.deepcopy()
        C4 = SplineCurve(knotvector, P4)
        assert id(C1) != id(C2)
        assert C1 == C2
        assert id(C1) != id(C3)
        assert C1 == C3
        assert C1 != C4

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestInitSplineCurve::test_begin",
            "TestInitSplineCurve::test_build_scalar",
            "TestInitSplineCurve::test_build_vectorial",
            "TestInitSplineCurve::test_atributesgood",
            "TestInitSplineCurve::test_functions",
            "TestInitSplineCurve::test_compare_two_curves",
        ]
    )
    def test_end(self):
        pass


class TestCallShape:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestInitSplineCurve::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestCallShape::test_begin"])
    def test_build_scalar(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                knotvector = GeneratorKnotVector.random(degree, npts)
                ctrlpoints = np.random.uniform(-1, 1, npts)
                SplineCurve(knotvector, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestCallShape::test_begin"])
    def test_build_vectorial(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                for ndim in range(1, 5):
                    knotvector = GeneratorKnotVector.random(degree, npts)
                    ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                    SplineCurve(knotvector, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_build_scalar",
        ]
    )
    def test_callscal_scalpts(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                knotvector = GeneratorKnotVector.random(degree, npts)
                ctrlpoints = np.random.uniform(-1, 1, npts)
                curve = SplineCurve(knotvector, ctrlpoints)

                tparam = np.random.uniform(0, 1)
                curvevalues = curve(tparam)
                assert type(curvevalues) == type(ctrlpoints[0])

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_build_vectorial",
        ]
    )
    def test_callscal_vectpts(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                for ndim in range(1, 5):
                    knotvector = GeneratorKnotVector.random(degree, npts)
                    ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                    curve = SplineCurve(knotvector, ctrlpoints)

                    tparam = np.random.uniform(0, 1)
                    curvevalues = curve(tparam)
                    assert len(curvevalues) == ndim
                    assert type(curvevalues) == type(ctrlpoints[0])
                    assert type(curvevalues[0]) == type(ctrlpoints[0][0])

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_build_scalar",
            "TestCallShape::test_callscal_scalpts",
        ]
    )
    def test_callvect_scalpts(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                knotvector = GeneratorKnotVector.random(degree, npts)
                ctrlpoints = np.random.uniform(-1, 1, npts)
                curve = SplineCurve(knotvector, ctrlpoints)

                lower = npts + degree + 2
                upper = npts + degree + 11
                nsample = np.random.randint(lower, upper)
                tparam = np.linspace(0, 1, nsample)
                Cval = curve(tparam)
                assert len(Cval) == nsample
                assert type(Cval[0]) == type(ctrlpoints[0])

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_build_vectorial",
            "TestCallShape::test_callscal_vectpts",
        ]
    )
    def test_callvect_vectpts(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                for ndim in range(1, 5):
                    knotvector = GeneratorKnotVector.random(degree, npts)
                    ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                    curve = SplineCurve(knotvector, ctrlpoints)

                    lower = npts + degree + 2
                    upper = npts + degree + 11
                    nsample = np.random.randint(lower, upper)
                    tparam = np.linspace(0, 1, nsample)
                    curvevalues = curve(tparam)
                    assert len(curvevalues) == nsample
                    assert type(curvevalues[0]) == type(ctrlpoints[0])
                    assert np.array(curvevalues).shape == (nsample, ndim)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_callscal_scalpts",
            "TestCallShape::test_callscal_vectpts",
            "TestCallShape::test_callvect_scalpts",
            "TestCallShape::test_callvect_vectpts",
        ]
    )
    def test_end(self):
        pass


class TestSumSubtract:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestCallShape::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestSumSubtract::test_begin"])
    def test_sumsub_failknotvector(self):
        """
        If the knotvectors are different, it's not possible to sum
        It's expected a ValueError
        """
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                U1 = GeneratorKnotVector.random(degree, npts)
                U2 = GeneratorKnotVector.random(degree, npts)
                P1 = np.random.uniform(-1, 1, npts)
                P2 = np.random.uniform(-1, 1, npts)
                C1 = SplineCurve(U1, P1)
                C2 = SplineCurve(U2, P2)
                with pytest.raises(ValueError):
                    C1 + C2
                with pytest.raises(ValueError):
                    C1 - C2

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestSumSubtract::test_begin"])
    def test_sumsub_failcontrolpoints(self):
        """
        Not possible sum if shape of control points are different,
        It's expected a ValueError
        """
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                U1 = GeneratorKnotVector.random(degree, npts)
                U2 = U1.deepcopy()
                P1 = np.random.uniform(-1, 1, npts)
                P2 = np.random.uniform(-1, 1, (npts, 2))
                C1 = SplineCurve(U1, P1)
                C2 = SplineCurve(U2, P2)
                with pytest.raises(ValueError):
                    C1 + C2
                with pytest.raises(ValueError):
                    C1 - C2

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSumSubtract::test_begin",
            "TestSumSubtract::test_sumsub_failknotvector",
        ]
    )
    def test_sumsub_scalar(self):
        """
        Tests if the sum of two curves is equal to the new
        curve obtained by summing the control points
        """
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                knotvector = GeneratorKnotVector.random(degree, npts)
                P1 = np.random.uniform(-1, 1, npts)
                P2 = np.random.uniform(-1, 1, npts)
                C1 = SplineCurve(knotvector, P1)
                C2 = SplineCurve(knotvector, P2)
                Cs = SplineCurve(knotvector, P1 + P2)
                Cd = SplineCurve(knotvector, P1 - P2)
                assert (C1 + C2) == Cs
                assert (C1 - C2) == Cd

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSumSubtract::test_begin",
            "TestSumSubtract::test_sumsub_failknotvector",
            "TestSumSubtract::test_sumsub_scalar",
        ]
    )
    def test_sumsub_vector(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                ndim = np.random.randint(1, 4)
                knotvector = GeneratorKnotVector.random(degree, npts)
                P1 = np.random.uniform(-1, 1, (npts, ndim))
                P2 = np.random.uniform(-1, 1, (npts, ndim))
                C1 = SplineCurve(knotvector, P1)
                C2 = SplineCurve(knotvector, P2)
                Cs = SplineCurve(knotvector, P1 + P2)
                Cd = SplineCurve(knotvector, P1 - P2)
                assert (C1 + C2) == Cs
                assert (C1 - C2) == Cd

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestSumSubtract::test_begin"])
    def test_end(self):
        pass


class TestKnotOperations:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestCallShape::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_begin",
        ]
    )
    def test_insert_known_case(self):
        """
        Example 5.1 of nurbs book
        """
        degree, npts, ndim = 3, 8, 3
        knot = 2.5
        Uorig = KnotVector([0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5])
        assert Uorig.degree == degree
        assert Uorig.npts == npts
        assert Uorig.span(knot) == 5
        assert Uorig.mult(knot) == 0

        P = np.random.uniform(-1, 1, (npts, ndim))
        Corig = SplineCurve(Uorig, P)
        assert Corig.degree == degree
        assert Corig.npts == npts
        Corig.knot_insert(knot)
        assert Corig.degree == degree
        assert Corig.npts == npts + 1

        Q = np.zeros((npts + 1, ndim), dtype="float64")
        Q[:3] = P[:3]
        Q[3] = (1 / 6) * P[2] + (5 / 6) * P[3]
        Q[4] = (1 / 2) * P[3] + (1 / 2) * P[4]
        Q[5] = (5 / 6) * P[4] + (1 / 6) * P[5]
        Q[6:] = P[5:]

        Uinse = [0, 0, 0, 0, 1, 2, 2.5, 3, 4, 5, 5, 5, 5]
        Cinse = SplineCurve(Uinse, Q)
        assert Corig == Cinse

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_begin",
            "TestKnotOperations::test_insert_known_case",
        ]
    )
    def test_remove_known_case(self):
        degree, npts, ndim = 3, 9, 3
        knot = 2.5
        Uorig = KnotVector([0, 0, 0, 0, 1, 2, 2.5, 3, 4, 5, 5, 5, 5])

        Q = np.random.uniform(-1, 1, (npts - 1, ndim))
        P = np.zeros((npts, ndim), dtype="float64")
        P[:3] = Q[:3]
        P[3] = (1 / 6) * Q[2] + (5 / 6) * Q[3]
        P[4] = (1 / 2) * Q[3] + (1 / 2) * Q[4]
        P[5] = (5 / 6) * Q[4] + (1 / 6) * Q[5]
        P[6:] = Q[5:]

        Corig = SplineCurve(Uorig, P)
        assert Corig.degree == degree
        assert Corig.npts == npts
        Corig.knot_remove(knot)
        assert Corig.degree == degree
        assert Corig.npts == npts - 1

        Uinse = [0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5]
        Cinse = SplineCurve(Uinse, Q)
        assert Corig == Cinse

    @pytest.mark.order(3)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_begin",
            "TestKnotOperations::test_insert_known_case",
            "TestKnotOperations::test_remove_known_case",
        ]
    )
    def test_insert_remove_once_random(self):
        for degree in range(1, 6):
            npts = np.random.randint(degree + 2, degree + 11)
            ndim = np.random.randint(1, 5)
            knotvector = GeneratorKnotVector.random(degree, npts)
            ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
            curve = SplineCurve(knotvector, ctrlpoints)

            knot = np.random.uniform(0, 1)
            curve.knot_insert(knot)
            curve.knot_remove(knot)

            assert curve == SplineCurve(knotvector, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_begin",
            "TestKnotOperations::test_insert_remove_once_random",
        ]
    )
    def test_knotclean(self):
        U = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
        assert U.degree == 2
        assert U.npts == 4
        P = np.random.uniform(-1, 1, (4, 2))
        curve = SplineCurve(U, P)
        curve.knot_insert(0.5)
        curve.knot_clean()
        assert curve.knotvector == U

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_begin",
            "TestKnotOperations::test_knotclean",
        ]
    )
    def test_knotclean_random(self):
        for degree in range(1, 5):
            npts = np.random.randint(degree + 2, degree + 11)
            ndim = np.random.randint(1, 5)
            U = GeneratorKnotVector.random(degree, npts)
            P = np.random.uniform(-1, 1, (npts, ndim))
            curve = SplineCurve(U, P)
            knot = np.random.rand()
            curve.knot_insert(knot)
            curve.knot_clean()
            assert curve.knotvector == U

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_begin",
            "TestKnotOperations::test_insert_known_case",
            "TestKnotOperations::test_remove_known_case",
            "TestKnotOperations::test_insert_remove_once_random",
            "TestKnotOperations::test_knotclean",
            "TestKnotOperations::test_knotclean_random",
        ]
    )
    def test_somefails(self):
        degree = np.random.randint(1, 5)
        npts = np.random.randint(degree + 2, degree + 11)
        knotvector = GeneratorKnotVector.random(degree, npts=npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        C = SplineCurve(knotvector, ctrlpoints)
        with pytest.raises(TypeError):
            C.knot_insert(["asd", 3, None])
        with pytest.raises(ValueError):
            C.knot_insert([[0.9, 0.1], [0.5, 0.3]])
        with pytest.raises(ValueError):
            C.knot_insert([-0.1, 0.1])
        with pytest.raises(ValueError):
            C.knot_insert([1.1, 0.1])
        with pytest.raises(ValueError):
            C.knot_remove(0.5)

        U = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]  # deg=3, npt=7
        P = np.random.uniform(-1, 1, 7)
        C = SplineCurve(U, P)
        with pytest.raises(ValueError):
            C.knot_remove([0.5, 0.5, 0.5, 0.5])

    @pytest.mark.order(3)
    # @pytest.mark.skip(reason="Pause")
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_begin",
            "TestKnotOperations::test_insert_known_case",
            "TestKnotOperations::test_remove_known_case",
            "TestKnotOperations::test_insert_remove_once_random",
            "TestKnotOperations::test_knotclean",
            "TestKnotOperations::test_knotclean_random",
            "TestKnotOperations::test_somefails",
        ]
    )
    def test_end(self):
        pass


class TestSplitUnite:
    @pytest.mark.order(3)
    @pytest.mark.skip(reason="Needs correction")
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
        ]
    )
    def test_split_number_curves(self):
        U = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
        P = np.random.uniform(-1, 1, U.npts)
        C = SplineCurve(U, P)
        assert len(C.split()) == 2
        assert len(C.split(0.5)) == 2
        assert len(C.split(0.25)) == 2
        assert len(C.split([0.25, 0.75])) == 3
        assert C == SplineCurve(U, P)

        U = KnotVector([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
        P = np.random.uniform(-1, 1, U.npts)
        C = SplineCurve(U, P)
        assert len(C.split()) == 4
        assert len(C.split(0.5)) == 2
        assert len(C.split(0.25)) == 2
        assert len(C.split([0.25, 0.75])) == 3
        assert C == SplineCurve(U, P)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_split_number_curves",
        ]
    )
    def test_split_matchboundary(self):
        U = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
        P = np.random.uniform(-1, 1, U.npts)
        C = SplineCurve(U, P)
        curves = C.split()
        assert np.all(curves[0](0.5) == curves[1](0.5))
        assert min(curves[0].knotvector) == 0
        assert max(curves[0].knotvector) == 0.5
        assert min(curves[1].knotvector) == 0.5
        assert max(curves[1].knotvector) == 1

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_split_matchboundary",
        ]
    )
    def test_splitrand_matchboundary(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 11):
                U = GeneratorKnotVector.random(degree, npts)
                P = np.random.uniform(-1, 1, npts)
                C = SplineCurve(U, P)
                knots = C.knots
                curves = C.split()
                assert len(curves) == len(knots) - 1
                for i, seg in enumerate(curves):
                    assert seg.degree == degree
                    assert seg.npts == degree + 1
                    assert min(seg.knotvector) == knots[i]
                    assert max(seg.knotvector) == knots[i + 1]
                for i, knot in enumerate(knots[:-1]):
                    np.all(curves[i](knot) == curves[i + 1](knot))

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_splitrand_matchboundary",
        ]
    )
    def test_split_knowncase1(self):
        """
        Split bezier -> one bezier
        """
        degree, npts = 2, 3
        U = KnotVector([0, 0, 0, 1, 1, 1])
        assert U.degree == degree
        assert U.npts == npts
        P = np.random.uniform(-1, 1, npts)
        curve_original = SplineCurve(U, P)
        curves = curve_original.split()
        assert len(curves) == 1
        assert curves[0] == curve_original

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_splitrand_matchboundary",
            "TestSplitUnite::test_split_knowncase2",
        ]
    )
    def test_split_knowncase2(self):
        degree, npts, ndim = 2, 4, 2
        U = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
        assert U.degree == degree
        assert U.npts == npts
        P = np.random.uniform(-1, 1, (npts, ndim))
        curve_original = SplineCurve(U, P)
        curves = curve_original.split()
        assert len(curves) == 2
        assert curves[0].knotvector == [0, 0, 0, 0.5, 0.5, 0.5]
        assert curves[1].knotvector == [0.5, 0.5, 0.5, 1, 1, 1]

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_split_knowncase1",
            "TestSplitUnite::test_split_knowncase2",
        ]
    )
    def test_unite_knowncase(self):
        U0 = KnotVector([0, 0, 0, 0.5, 0.5, 0.5])
        U1 = KnotVector([0.5, 0.5, 0.5, 1, 1, 1])
        assert U0.degree == 2
        assert U1.degree == 2
        assert U0.npts == 3
        assert U1.npts == 3
        P0 = np.random.uniform(-1, 1, (3, 2))
        P1 = np.random.uniform(-1, 1, (3, 2))
        P1[0] = P0[-1]
        curve0 = SplineCurve(U0, P0)
        curve1 = SplineCurve(U1, P1)

        newcurve = curve0 + curve1  # Concatenate
        curves = newcurve.split(0.5)
        assert curves[0] == curve0
        assert curves[0] == curve1

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_split_knowncase1",
            "TestSplitUnite::test_split_knowncase2",
        ]
    )
    def test_somefails(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_split_number_curves",
            "TestSplitUnite::test_split_matchboundary",
            "TestSplitUnite::test_splitrand_matchboundary",
            "TestSplitUnite::test_split_knowncase1",
            "TestSplitUnite::test_split_knowncase2",
            "TestSplitUnite::test_unite_knowncase",
            "TestSplitUnite::test_somefails",
        ]
    )
    def test_somefails(self):
        degree = np.random.randint(1, 5)
        npts = np.random.randint(degree + 2, degree + 11)
        knotvector = GeneratorKnotVector.random(degree, npts=npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        C = SplineCurve(knotvector, ctrlpoints)
        with pytest.raises(TypeError):
            C.knot_insert(["asd", 3, None])
        with pytest.raises(ValueError):
            C.knot_insert([[0.9, 0.1], [0.5, 0.3]])
        with pytest.raises(ValueError):
            C.knot_insert([-0.1, 0.1])
        with pytest.raises(ValueError):
            C.knot_insert([1.1, 0.1])
        with pytest.raises(ValueError):
            C.knot_remove(0.5)
        U = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]  # deg=3, npt=7
        P = np.random.uniform(-1, 1, 7)
        C = SplineCurve(U, P)
        with pytest.raises(ValueError):
            C.knot_remove([0.5, 0.5, 0.5, 0.5])

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_split_number_curves",
            "TestSplitUnite::test_split_matchboundary",
            "TestSplitUnite::test_splitrand_matchboundary",
            "TestSplitUnite::test_split_known_case1",
            "TestSplitUnite::test_split_known_case2",
            "TestSplitUnite::test_unite_known_case",
            "TestSplitUnite::test_somefails",
        ]
    )
    def test_end(self):
        pass


class TestDegreeOperations:
    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.skip(reason="Needs correction")
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_end",
            "TestSplitUnite::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
        ]
    )
    def test_degree_increase_bezier_degree_1(self):
        degree = 1
        npts = degree + 1
        knotvector = [0] * npts + [1] * npts
        ctrlpoints = np.random.uniform(-1, 1, (npts, 2))

        curve = SplineCurve(knotvector, ctrlpoints)
        curve.degree += 1
        assert curve.degree == (degree + 2)
        correctctrlpoints = [
            ctrlpoints[0],
            0.5 * (ctrlpoints[0] + ctrlpoints[1]),
            ctrlpoints[1],
        ]
        np.testing.assert_allclose(curve.ctrlpoints, correctctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_bezier_degree_1",
        ]
    )
    def test_degree_increase_bezier_degree_2(self):
        degree = 2
        npts = degree + 1
        knotvector = [0] * npts + [1] * npts
        ctrlpoints = np.random.uniform(-1, 1, (npts, 2))

        curve = SplineCurve(knotvector, ctrlpoints)
        curve.degree += 1
        assert curve.degree == (degree + 2)
        correctctrlpoints = [
            ctrlpoints[0],
            (ctrlpoints[0] + 2 * ctrlpoints[1]) / 3,
            (2 * ctrlpoints[1] + ctrlpoints[2]) / 3,
            ctrlpoints[2],
        ]
        np.testing.assert_allclose(curve.ctrlpoints, correctctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_bezier_degree_1",
        ]
    )
    def test_degree_increase_bezier_degree_3(self):
        degree = 3
        npts = degree + 1
        knotvector = [0] * npts + [1] * npts
        ctrlpoints = np.random.uniform(-1, 1, (npts, 2))

        curve = SplineCurve(knotvector, ctrlpoints)
        curve.degree += 1
        assert curve.degree == (degree + 2)
        Pgood = [
            ctrlpoints[0],
            (ctrlpoints[0] + 3 * ctrlpoints[1]) / 4,
            (ctrlpoints[1] + ctrlpoints[2]) / 2,
            (3 * ctrlpoints[2] + ctrlpoints[3]) / 4,
            ctrlpoints[3],
        ]
        np.testing.assert_allclose(curve.ctrlpoints, Pgood)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_bezier_degree_2",
        ]
    )
    def test_degree_increase_bezier_degree_2_times4(self):
        """Example at page 205 of nurbs book"""
        degree = 2
        npts = degree + 1
        knotvector = [0] * npts + [1] * npts
        ctrlpoints = np.random.uniform(-1, 1, (npts, 2))

        curve = SplineCurve(knotvector, ctrlpoints)
        curve.degree += 4
        assert curve.degree == (degree + 4)
        correctctrlpoints = [
            ctrlpoints[0],
            (4 * ctrlpoints[0] + 2 * ctrlpoints[1]) / 6,
            (6 * ctrlpoints[0] + 8 * ctrlpoints[1] + ctrlpoints[2]) / 15,
            (4 * ctrlpoints[0] + 12 * ctrlpoints[1] + 4 * ctrlpoints[2]) / 20,
            (ctrlpoints[0] + 8 * ctrlpoints[1] + 6 * ctrlpoints[2]) / 15,
            (2 * ctrlpoints[1] + 4 * ctrlpoints[2]) / 6,
            ctrlpoints[2],
        ]
        np.testing.assert_allclose(curve.ctrlpoints, correctctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_bezier_degree_2_times4",
        ]
    )
    def test_degree_increase_bezier_random(self):
        for degree in range(1, 6):
            npts = degree + 1
            ndim = np.random.randint(1, 4)
            times = np.random.randint(1, 5)
            knotvector = self.create_random_knotvector(degree, npts)
            ctrlpoints = self.create_random_controlpoints(npts, ndim)
            curve = SplineCurve(knotvector, ctrlpoints)
            matrix = self.matrix_increase_degree(degree, times)
            matrix = np.array(matrix)
            Pgood = matrix @ ctrlpoints
            curve.degree += times
            assert curve.degree == (degree + times)
            np.testing.assert_allclose(curve.ctrlpoints[0], ctrlpoints[0])
            np.testing.assert_allclose(curve.ctrlpoints[-1], ctrlpoints[-1])
            np.testing.assert_allclose(curve.ctrlpoints, Pgood)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_bezier_random",
        ]
    )
    def test_degree_decrease_bezier_degree1(self):
        degree = 1
        npts = degree + 1
        ndim = np.random.randint(0, 5)
        Ugood = [0] * npts + [1] * npts
        Pgood = self.create_random_controlpoints(npts, ndim)

        Uinc = [0] * (npts + 1) + [1] * (npts + 1)
        Pinc = [Pgood[0], 0.5 * (Pgood[0] + Pgood[1]), Pgood[1]]
        curve = SplineCurve(Uinc, Pinc)
        curve.degree -= 1

        assert curve.knotvector == Ugood
        np.testing.assert_allclose(curve.ctrlpoints, Pgood)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_decrease_bezier_degree1",
        ]
    )
    def test_degree_decrease_bezier_degree2(self):
        degree = 2
        npts = degree + 1
        ndim = np.random.randint(0, 5)
        ndim = 0
        Ugood = [0] * npts + [1] * npts
        Pgood = self.create_random_controlpoints(npts, ndim)

        Uinc = [0] * (npts + 1) + [1] * (npts + 1)
        matrix = self.matrix_increase_degree(degree, 1)
        Pinc = matrix @ Pgood
        curve = SplineCurve(Uinc, Pinc)
        curve.degree -= 1

        assert curve.knotvector == Ugood
        np.testing.assert_allclose(curve.ctrlpoints, Pgood)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_decrease_bezier_degree1",
            "TestDegreeOperations::test_degree_decrease_bezier_degree2",
        ]
    )
    def test_degree_decrease_bezier_degree3(self):
        degree = 3
        npts = degree + 1
        ndim = np.random.randint(0, 5)
        ndim = 0
        Ugood = [0] * npts + [1] * npts
        Pgood = self.create_random_controlpoints(npts, ndim)

        Uinc = [0] * (npts + 1) + [1] * (npts + 1)
        Mat = self.matrix_increase_degree(degree, 1)
        Pinc = [
            Pgood[0],
            (Pgood[0] + 3 * Pgood[1]) / 4,
            (Pgood[1] + Pgood[2]) / 2,
            (3 * Pgood[2] + Pgood[3]) / 4,
            Pgood[3],
        ]
        curve = SplineCurve(Uinc, Pinc)
        curve.degree -= 1

        assert curve.knotvector == Ugood
        np.testing.assert_allclose(curve.ctrlpoints, Pgood)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_decrease_bezier_degree1",
            "TestDegreeOperations::test_degree_decrease_bezier_degree2",
            "TestDegreeOperations::test_degree_decrease_bezier_degree3",
        ]
    )
    def test_degree_decrease_bezier_random_degree(self):
        for degree in range(1, 6):
            npts = degree + 1
            ndim = np.random.randint(0, 5)
            Ugood = [0] * npts + [1] * npts
            Pgood = self.create_random_controlpoints(npts, ndim)

            Uinc = [0] * (npts + 1) + [1] * (npts + 1)
            matrix = self.matrix_increase_degree(degree, 1)
            Pinc = matrix @ Pgood
            curve = SplineCurve(Uinc, Pinc)
            curve.degree -= 1

            assert curve.knotvector == Ugood
            np.testing.assert_allclose(curve.ctrlpoints, Pgood)

    @pytest.mark.order(3)
    @pytest.mark.timeout(30)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_bezier_random",
            "TestDegreeOperations::test_degree_decrease_bezier_random_degree",
        ]
    )
    def test_degree_increase_decrease_random_bezier(self, ntests=1):
        for degree in range(1, 6):
            for i in range(ntests):
                npts = degree + 1
                ndim = np.random.randint(0, 5)
                times = np.random.randint(1, 5)
                times = 1
                knotvector = self.create_random_knotvector(degree, npts)
                ctrlpoints = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(knotvector, ctrlpoints)
                C.degree += 1
                assert C.degree == degree + times
                C.degree -= 1
                assert C.degree == degree
                np.testing.assert_allclose(C.ctrlpoints[0], ctrlpoints[0])
                np.testing.assert_allclose(C.ctrlpoints[degree], ctrlpoints[degree])
                np.testing.assert_allclose(C.ctrlpoints, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(60)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_decrease_random_bezier",
        ]
    )
    def test_degree_increase_decrease_random(self, ntests=1):
        for degree in range(1, 3):
            for i in range(ntests):
                npts = np.random.randint(degree + 2, degree + 4)
                ndim = np.random.randint(0, 5)
                times = np.random.randint(1, 3)
                knotvector = self.create_random_knotvector(degree, npts)
                ctrlpoints = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(knotvector, ctrlpoints)
                C.degree += times
                C.degree -= times
                assert C.degree == degree
                np.testing.assert_allclose(C.ctrlpoints[0], ctrlpoints[0])
                np.testing.assert_allclose(C.ctrlpoints[degree], ctrlpoints[degree])
                np.testing.assert_allclose(C.ctrlpoints, ctrlpoints)

    @pytest.mark.order(3)
    @pytest.mark.timeout(60)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_decrease_random",
        ]
    )
    def test_degree_clean(self, ntests=1):
        for degree in range(1, 3):
            for i in range(ntests):
                npts = np.random.randint(degree + 2, degree + 4)
                ndim = np.random.randint(0, 5)
                times = np.random.randint(1, 3)
                knotvector = self.create_random_knotvector(degree, npts)
                ctrlpoints = self.create_random_controlpoints(npts, ndim)
                C = SplineCurve(knotvector, ctrlpoints)
                C.degree += times
                C.degree_clean()
                assert C.degree == degree

    def test_fails(self):
        U = KnotVector([0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1])
        assert U.degree == 3
        assert U.npts == 7
        P = np.random.uniform(-1, 1, 7)
        C = SplineCurve(U, P)
        with pytest.raises(ValueError):
            C.degree -= 1
        with pytest.raises(ValueError):
            C.degree -= 4
        with pytest.raises(ValueError):
            C.degree_decrease(5)

    @pytest.mark.order(3)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_degree_increase_decrease_random",
            "TestDegreeOperations::test_degree_clean",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestInitSplineCurve::test_end",
        "TestCallShape::test_end",
        "TesKnotOperations::test_end",
        "TestSplitUnite::test_end",
        "TestDegreeOperations::test_end",
        "TestSumSubtract::test_end",
    ]
)
def test_end():
    pass
