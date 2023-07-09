import numpy as np
import pytest

from compmec.nurbs.curves import Curve
from compmec.nurbs.knotspace import GeneratorKnotVector, KnotVector


@pytest.mark.order(5)
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


class TestInitCurve:
    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInitCurve::test_begin"])
    def test_build_scalar(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        Curve(knotvector, ctrlpoints)

    @pytest.mark.order(5)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInitCurve::test_begin"])
    def test_build_vectorial(self):
        degree, npts = 3, 9
        ndim = 3
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
        Curve(knotvector, ctrlpoints)

    @pytest.mark.order(5)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_scalar"])
    def test_failbuild(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts + 1)
        with pytest.raises(ValueError):
            Curve(knotvector, ctrlpoints)
        with pytest.raises(ValueError):
            Curve(knotvector, "asd")
        with pytest.raises(ValueError):
            Curve(knotvector, "asdefghjk")
        with pytest.raises(ValueError):
            Curve(knotvector, 1)

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_scalar"])
    def test_attributes(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = Curve(knotvector, ctrlpoints)
        curve = Curve(knotvector, ctrlpoints)
        # Attributes
        assert hasattr(curve, "degree")
        assert hasattr(curve, "npts")
        assert hasattr(curve, "ctrlpoints")
        assert hasattr(curve, "knotvector")
        assert hasattr(curve, "knots")

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_scalar"])
    def test_functions(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = Curve(knotvector, ctrlpoints)
        curve = Curve(knotvector, ctrlpoints)
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

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestInitCurve::test_attributes"])
    def test_atributesgood(self):
        degree, npts = 3, 9
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = Curve(knotvector, ctrlpoints)
        assert curve.degree == degree
        assert curve.npts == npts
        assert curve.knotvector == knotvector
        np.testing.assert_allclose(curve.ctrlpoints, ctrlpoints)
        assert curve.knots == knotvector.knots

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_vectorial"])
    def test_compare_two_curves(self):
        degree = np.random.randint(1, 5)
        npts = np.random.randint(degree + 2, degree + 9)
        ndim = np.random.randint(1, 4)
        knotvector = GeneratorKnotVector.random(degree, npts)
        P1 = np.random.uniform(-2, -1, (npts, ndim))
        P4 = np.random.uniform(1, 2, (npts, ndim))
        C1 = Curve(knotvector, P1)
        C2 = Curve(knotvector, P1)
        C3 = C1.deepcopy()
        C4 = Curve(knotvector, P4)
        assert id(C1) != id(C2)
        assert C1 == C2
        assert id(C1) != id(C3)
        assert C1 == C3
        assert C1 != C4

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_scalar"])
    def test_print(self):
        knotvector = GeneratorKnotVector.uniform(2, 4)
        spline = Curve(knotvector)
        str(spline)
        spline.ctrlpoints = [2, 4, 3, 1]
        str(spline)

    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=[
            "TestInitCurve::test_begin",
            "TestInitCurve::test_build_scalar",
            "TestInitCurve::test_build_vectorial",
            "TestInitCurve::test_atributesgood",
            "TestInitCurve::test_functions",
            "TestInitCurve::test_compare_two_curves",
            "TestInitCurve::test_print",
        ]
    )
    def test_end(self):
        pass


class TestCompare:
    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestInitCurve::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestCompare::test_begin"])
    def test_knotvector(self):
        degree, npts = 3, 7
        knotvector0 = GeneratorKnotVector.uniform(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve0 = Curve(knotvector0, ctrlpoints)

        knotvector1 = KnotVector(np.array(knotvector0) + 0.5)
        curve1 = Curve(knotvector1, ctrlpoints)
        assert curve0 != curve1

        knotvector1 = KnotVector(0.5 * np.array(knotvector0))
        curve1 = Curve(knotvector1, ctrlpoints)
        assert curve0 != curve1

        knotvector1 = KnotVector(0.5 * np.array(knotvector0) + 0.5)
        curve1 = Curve(knotvector1, ctrlpoints)
        assert curve0 != curve1

        knotvector1 = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve1 = Curve(knotvector1, ctrlpoints)
        assert curve0 != curve1

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestCompare::test_begin"])
    def test_controlpoints(self):
        npts = 7
        knotvector0 = GeneratorKnotVector.uniform(3, npts)
        knotvector1 = GeneratorKnotVector.uniform(3, npts)
        ctrlpoints0 = np.random.uniform(-1, 1, npts)
        ctrlpoints1 = np.random.uniform(-1, 1, npts)
        curve0 = Curve(knotvector0, ctrlpoints0)
        curve1 = Curve(knotvector1, ctrlpoints1)
        assert curve0 != curve1

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestCompare::test_begin"])
    def test_userentry(self):
        npts = 7
        knotvector = GeneratorKnotVector.uniform(3, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = Curve(knotvector, ctrlpoints)
        assert curve != 0
        assert curve != "asd"
        assert curve != knotvector
        assert curve != ctrlpoints
        assert curve != []

    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=[
            "TestCompare::test_begin",
            "TestCompare::test_knotvector",
            "TestCompare::test_controlpoints",
        ]
    )
    def test_end(self):
        pass


class TestCallShape:
    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=["TestCompare::test_end", "TestInitCurve::test_end"]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestCallShape::test_begin"])
    def test_build_scalar(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 9):
                knotvector = GeneratorKnotVector.random(degree, npts)
                ctrlpoints = np.random.uniform(-1, 1, npts)
                Curve(knotvector, ctrlpoints)

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestCallShape::test_begin"])
    def test_build_vectorial(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 9):
                for ndim in range(1, 5):
                    knotvector = GeneratorKnotVector.random(degree, npts)
                    ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                    Curve(knotvector, ctrlpoints)

    @pytest.mark.order(5)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_build_scalar",
        ]
    )
    def test_callscal_scalpts(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 9):
                knotvector = GeneratorKnotVector.random(degree, npts)
                ctrlpoints = np.random.uniform(-1, 1, npts)
                curve = Curve(knotvector, ctrlpoints)

                tparam = np.random.uniform(knotvector[0], knotvector[-1])
                curvevalues = curve(tparam)
                assert type(curvevalues) == type(ctrlpoints[0])

    @pytest.mark.order(5)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_build_vectorial",
        ]
    )
    def test_callscal_vectpts(self, ntests=1):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 9):
                for ndim in range(1, 5):
                    knotvector = GeneratorKnotVector.random(degree, npts)
                    ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                    curve = Curve(knotvector, ctrlpoints)

                    tparam = np.random.uniform(knotvector[0], knotvector[-1])
                    curvevalues = curve(tparam)
                    assert len(curvevalues) == ndim
                    assert type(curvevalues) == type(ctrlpoints[0])
                    assert type(curvevalues[0]) == type(ctrlpoints[0][0])

    @pytest.mark.order(5)
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
            for npts in range(degree + 2, degree + 9):
                knotvector = GeneratorKnotVector.random(degree, npts)
                ctrlpoints = np.random.uniform(-1, 1, npts)
                curve = Curve(knotvector, ctrlpoints)

                lower = npts + degree + 2
                upper = npts + degree + 9
                nsample = np.random.randint(lower, upper)
                tparam = np.linspace(knotvector[0], knotvector[-1], nsample)
                Cval = curve(tparam)
                assert len(Cval) == nsample
                assert type(Cval[0]) == type(ctrlpoints[0])

    @pytest.mark.order(5)
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
            for npts in range(degree + 2, degree + 9):
                for ndim in range(1, 5):
                    knotvector = GeneratorKnotVector.random(degree, npts)
                    ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                    curve = Curve(knotvector, ctrlpoints)

                    lower = npts + degree + 2
                    upper = npts + degree + 9
                    nsample = np.random.randint(lower, upper)
                    tparam = np.linspace(knotvector[0], knotvector[-1], nsample)
                    curvevalues = curve(tparam)
                    assert len(curvevalues) == nsample
                    assert type(curvevalues[0]) == type(ctrlpoints[0])
                    assert np.array(curvevalues).shape == (nsample, ndim)

    @pytest.mark.order(5)
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
    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=["TestCompare::test_end", "TestCallShape::test_end"]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestSumSubtract::test_begin"])
    def test_sumsub_failknotvector(self):
        """
        If the knotvectors are different, it's not possible to sum
        It's expected a ValueError
        """
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 9):
                U1 = GeneratorKnotVector.random(degree, npts)
                U2 = GeneratorKnotVector.random(degree, npts)
                P1 = np.random.uniform(-1, 1, npts)
                P2 = np.random.uniform(-1, 1, npts)
                C1 = Curve(U1, P1)
                C2 = Curve(U2, P2)
                with pytest.raises(ValueError):
                    C1 + C2
                with pytest.raises(ValueError):
                    C1 - C2

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestSumSubtract::test_begin"])
    def test_sumsub_scalar(self):
        """
        Tests if the sum of two curves is equal to the new
        curve obtained by summing the control points
        """
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 9):
                knotvector = GeneratorKnotVector.random(degree, npts)
                P1 = np.random.uniform(-1, 1, npts)
                P2 = np.random.uniform(-1, 1, npts)
                C1 = Curve(knotvector, P1)
                C2 = Curve(knotvector, P2)
                Cadd = Curve(knotvector, P1 + P2)
                Csub = Curve(knotvector, P1 - P2)
                assert (C1 + C2) == Cadd
                assert (C1 - C2) == Csub

    @pytest.mark.order(5)
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
            for npts in range(degree + 2, degree + 9):
                ndim = np.random.randint(1, 4)
                knotvector = GeneratorKnotVector.random(degree, npts)
                P1 = np.random.uniform(-1, 1, (npts, ndim))
                P2 = np.random.uniform(-1, 1, (npts, ndim))
                C1 = Curve(knotvector, P1)
                C2 = Curve(knotvector, P2)
                Cs = Curve(knotvector, P1 + P2)
                Cd = Curve(knotvector, P1 - P2)
                assert (C1 + C2) == Cs
                assert (C1 - C2) == Cd

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestSumSubtract::test_begin"])
    def test_somefails(self):
        knotvector = GeneratorKnotVector.random(3, 7)
        ctrlpoints = np.random.uniform(-1, 1, 7)
        curve = Curve(knotvector, ctrlpoints)
        with pytest.raises(TypeError):
            curve + 1
        with pytest.raises(TypeError):
            curve + "asd"

    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=[
            "TestSumSubtract::test_begin",
            "TestSumSubtract::test_sumsub_scalar",
            "TestSumSubtract::test_sumsub_vector",
            "TestSumSubtract::test_somefails",
        ]
    )
    def test_end(self):
        pass


class TestKnotOperations:
    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=["TestCompare::test_end", "TestCallShape::test_end"]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(5)
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
        Corig = Curve(Uorig, P)
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
        Cinse = Curve(Uinse, Q)
        assert Corig == Cinse

    @pytest.mark.order(5)
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

        Corig = Curve(Uorig, P)
        assert Corig.degree == degree
        assert Corig.npts == npts
        Corig.knot_remove(knot)
        assert Corig.degree == degree
        assert Corig.npts == npts - 1

        Uinse = [0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5]
        Cinse = Curve(Uinse, Q)
        assert Corig == Cinse

    @pytest.mark.order(5)
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
            npts = np.random.randint(degree + 2, degree + 9)
            ndim = np.random.randint(1, 5)
            knotvector = GeneratorKnotVector.random(degree, npts)
            ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
            curve = Curve(knotvector, ctrlpoints)

            umin, umax = knotvector[0], knotvector[-1]
            knot = np.random.uniform(umin, umax)
            curve.knot_insert(knot)
            curve.knot_remove(knot)

            assert curve == Curve(knotvector, ctrlpoints)

    @pytest.mark.order(5)
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
        curve = Curve(U, P)
        curve.knot_insert(0.5)
        curve.knot_clean()
        assert curve.knotvector == U

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_begin",
            "TestKnotOperations::test_knotclean",
        ]
    )
    def test_knotclean_random(self):
        for degree in range(1, 5):
            npts = np.random.randint(degree + 2, degree + 9)
            ndim = np.random.randint(1, 5)
            knotvector = GeneratorKnotVector.random(degree, npts)
            P = np.random.uniform(-1, 1, (npts, ndim))
            curve = Curve(knotvector, P)
            umin, umax = knotvector[0], knotvector[-1]
            knot = np.random.uniform(umin, umax)
            curve.knot_insert(knot)
            curve.knot_clean()
            assert curve.knotvector == knotvector

    @pytest.mark.order(5)
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
        npts = np.random.randint(degree + 2, degree + 9)
        knotvector = GeneratorKnotVector.random(degree, npts)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        C = Curve(knotvector, ctrlpoints)
        with pytest.raises(ValueError):
            C.knot_insert(["asd", 3, None])
        with pytest.raises(ValueError):
            C.knot_remove(0.5)

        U = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]  # deg=3, npt=7
        P = np.random.uniform(-1, 1, 7)
        C = Curve(U, P)
        with pytest.raises(ValueError):
            C.knot_remove([0.5, 0.5, 0.5, 0.5])

    @pytest.mark.order(5)
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
    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=[
            "TestCompare::test_end",
            "TestKnotOperations::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
        ]
    )
    def test_split_number_curves(self):
        U = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
        P = np.random.uniform(-1, 1, U.npts)
        C = Curve(U, P)
        assert len(C.split()) == 2
        assert len(C.split(0.5)) == 2
        assert len(C.split(0.25)) == 2
        assert len(C.split([0.25, 0.75])) == 3
        assert C == Curve(U, P)

        U = KnotVector([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
        P = np.random.uniform(-1, 1, U.npts)
        C = Curve(U, P)
        assert len(C.split()) == 4
        assert len(C.split(0.5)) == 2
        assert len(C.split(0.25)) == 2
        assert len(C.split([0.25, 0.75])) == 3
        assert C == Curve(U, P)

    @pytest.mark.order(5)
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
        C = Curve(U, P)
        curves = C.split()
        assert np.all(curves[0](0.5) == curves[1](0.5))
        assert min(curves[0].knotvector) == 0
        assert max(curves[0].knotvector) == 0.5
        assert min(curves[1].knotvector) == 0.5
        assert max(curves[1].knotvector) == 1

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_split_matchboundary",
        ]
    )
    def test_splitrand_matchboundary(self):
        for degree in range(1, 5):
            for npts in range(degree + 2, degree + 9):
                U = GeneratorKnotVector.random(degree, npts)
                P = np.random.uniform(-1, 1, npts)
                C = Curve(U, P)
                knots = C.knots
                curves = C.split()
                assert len(curves) == len(knots) - 1
                for i, seg in enumerate(curves):
                    assert seg.degree == degree
                    assert seg.npts == degree + 1
                    assert min(seg.knotvector) == knots[i]
                    assert max(seg.knotvector) == knots[i + 1]
                for i, knot in enumerate(knots[1:-1]):
                    np.all(curves[i](knot) == curves[i + 1](knot))

    @pytest.mark.order(5)
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
        curve_original = Curve(U, P)
        curves = curve_original.split()
        assert len(curves) == 1
        assert curves[0] == curve_original

    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_splitrand_matchboundary",
            "TestSplitUnite::test_split_knowncase1",
        ]
    )
    def test_split_knowncase2(self):
        degree, npts, ndim = 2, 4, 2
        U = KnotVector([0, 0, 0, 0.5, 1, 1, 1])
        assert U.degree == degree
        assert U.npts == npts
        P = np.random.uniform(-1, 1, (npts, ndim))
        curve_original = Curve(U, P)
        curves = curve_original.split()
        assert len(curves) == 2
        assert curves[0].knotvector == [0, 0, 0, 0.5, 0.5, 0.5]
        assert curves[1].knotvector == [0.5, 0.5, 0.5, 1, 1, 1]

    @pytest.mark.order(5)
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
        curve0 = Curve(U0, P0)
        curve1 = Curve(U1, P1)

        newcurve = curve0 | curve1  # Concatenate
        curves = newcurve.split(0.5)
        assert curves[0] == curve0
        assert curves[1] == curve1

    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=[
            "TestSplitUnite::test_begin",
            "TestSplitUnite::test_split_number_curves",
            "TestSplitUnite::test_split_matchboundary",
            "TestSplitUnite::test_splitrand_matchboundary",
            "TestSplitUnite::test_split_knowncase1",
            "TestSplitUnite::test_split_knowncase2",
            "TestSplitUnite::test_unite_knowncase",
        ]
    )
    def test_somefails(self):
        knotvector0 = [0, 0, 0.25, 0.5, 0.75, 1, 1]
        ctrlpoints = np.random.uniform(-1, 1, 5)
        curve0 = Curve(knotvector0, ctrlpoints)
        knotvector1 = [1, 1, 1.25, 1.5, 1.75, 2, 2]
        curve1 = Curve(knotvector1, ctrlpoints)
        with pytest.raises(ValueError):
            curve0 | curve0
        with pytest.raises(ValueError):
            curve0 | curve1

    @pytest.mark.order(5)
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
    def test_end(self):
        pass


class TestDegreeOperations:
    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestKnotOperations::test_end",
            "TestSplitUnite::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(depends=["TestDegreeOperations::test_begin"])
    def test_increase_decrease_random(self):
        for degree in range(1, 4):
            times = np.random.randint(1, 3)
            npts = np.random.randint(degree + 2, degree + 9)
            knotvector = GeneratorKnotVector.random(degree, npts)
            original_ctrlpoints = np.random.uniform(-1, 1, npts)
            curve = Curve(knotvector, original_ctrlpoints)
            curve.degree += times
            assert curve.degree == degree + times
            curve.degree -= times
            assert curve.degree == degree
            np.testing.assert_allclose(curve.ctrlpoints, curve.ctrlpoints)

    @pytest.mark.order(5)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_increase_decrease_random",
        ]
    )
    def test_clean(self):
        for degree in range(1, 4):
            times = np.random.randint(1, 4)
            npts = np.random.randint(degree + 2, degree + 9)
            knotvector = GeneratorKnotVector.random(degree, npts)
            ctrlpoints = np.random.uniform(-1, 1, npts)
            curve = Curve(knotvector, ctrlpoints)
            curve.degree += times
            curve.degree_clean()
            assert curve.degree == degree
            np.testing.assert_allclose(curve.ctrlpoints, ctrlpoints)

    @pytest.mark.order(5)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=["TestDegreeOperations::test_begin", "TestDegreeOperations::test_clean"]
    )
    def test_fails(self):
        U = KnotVector([0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1])
        assert U.degree == 3
        assert U.npts == 7
        P = np.random.uniform(-1, 1, 7)
        curve = Curve(U, P)
        with pytest.raises(ValueError):
            # tolerance error
            curve.degree -= 1
        with pytest.raises(ValueError):
            # max degree = 3
            curve.degree -= 4
        with pytest.raises(ValueError):
            curve.degree = "asd"

    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_clean",
            "TestDegreeOperations::test_fails",
        ]
    )
    def test_end(self):
        pass


class TestOthers:
    @pytest.mark.order(5)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestInitCurve::test_end",
            "TestCallShape::test_end",
            "TestKnotOperations::test_end",
            "TestSplitUnite::test_end",
            "TestDegreeOperations::test_end",
            "TestSumSubtract::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(depends=["TestOthers::test_begin"])
    def test_others(self):
        knotvector = KnotVector([1, 1, 2, 2])
        curve = Curve(knotvector)
        assert curve.knotvector == knotvector
        assert id(curve.knotvector) != id(knotvector)

        newvector = knotvector.deepcopy()
        newvector += [1.5]
        curve.knotvector = newvector
        with pytest.raises(ValueError):
            curve(1.5)
        curve.ctrlpoints = np.random.uniform(-1, 1, curve.npts)
        curve.degree_increase(1)
        curve.degree_decrease(1)
        curve.knot_insert(1.5)
        curve.knot_remove(1.5)
        with pytest.raises(ValueError):
            curve.knot_remove(1.5)
        curve.knot_remove(1.5, None)  # No tolerance

    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=["TestOthers::test_begin", "TestOthers::test_others"]
    )
    def test_end(self):
        pass


@pytest.mark.order(5)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestInitCurve::test_end",
        "TestCallShape::test_end",
        "TestKnotOperations::test_end",
        "TestSplitUnite::test_end",
        "TestDegreeOperations::test_end",
        "TestSumSubtract::test_end",
    ]
)
def test_end():
    pass
