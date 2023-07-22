from __future__ import annotations

from fractions import Fraction as frac

import numpy as np
import pytest

from compmec.nurbs.curves import Curve
from compmec.nurbs.knotspace import GeneratorKnotVector, KnotVector


@pytest.mark.order(4)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_functions.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestInitCurve:
    @pytest.mark.order(4)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(4)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInitCurve::test_begin"])
    def test_build_scalar(self):
        degree, npts = 3, 4
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        Curve(knotvector, ctrlpoints)

    @pytest.mark.order(4)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInitCurve::test_begin"])
    def test_build_vectorial(self):
        degree, npts = 3, 4
        ndim = 3
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
        Curve(knotvector, ctrlpoints)

    @pytest.mark.order(4)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_scalar"])
    def test_failbuild(self):
        degree, npts = 3, 4
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts + 1)
        with pytest.raises(ValueError):
            Curve(knotvector, ctrlpoints)
        with pytest.raises(TypeError):
            Curve(knotvector, "asd")
        with pytest.raises(TypeError):
            Curve(knotvector, "asdefghjk")
        with pytest.raises(TypeError):
            Curve(knotvector, 1)

    @pytest.mark.order(4)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_scalar"])
    def test_attributes(self):
        degree, npts = 3, 4
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = Curve(knotvector, ctrlpoints)
        curve = Curve(knotvector, ctrlpoints)
        # Attributes
        assert hasattr(curve, "degree")
        assert hasattr(curve, "npts")
        assert hasattr(curve, "ctrlpoints")
        assert hasattr(curve, "knotvector")
        assert hasattr(curve, "knots")

    @pytest.mark.order(4)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_scalar"])
    def test_functions(self):
        degree, npts = 3, 4
        knotvector = GeneratorKnotVector.bezier(degree)
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
        assert hasattr(curve, "__str__")
        assert callable(curve)

    @pytest.mark.order(4)
    @pytest.mark.dependency(depends=["TestInitCurve::test_attributes"])
    def test_atributesgood(self):
        degree, npts = 3, 4
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = Curve(knotvector, ctrlpoints)
        assert curve.degree == degree
        assert curve.npts == npts
        assert curve.knotvector == knotvector
        np.testing.assert_allclose(curve.ctrlpoints, ctrlpoints)
        assert curve.knots == knotvector.knots

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_vectorial"])
    def test_compare_two_curves(self):
        degree = np.random.randint(1, 5)
        npts = degree + 1
        ndim = np.random.randint(1, 4)
        knotvector = GeneratorKnotVector.bezier(degree)
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

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestInitCurve::test_build_scalar"])
    def test_print(self):
        knotvector = GeneratorKnotVector.bezier(3)
        bezier = Curve(knotvector)
        str(bezier)
        bezier.ctrlpoints = [2, 4, 3, 1]
        str(bezier)

    @pytest.mark.order(4)
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
    @pytest.mark.order(4)
    @pytest.mark.dependency(depends=["TestInitCurve::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(4)
    @pytest.mark.dependency(depends=["TestCompare::test_begin"])
    def test_knotvector(self):
        degree, npts = 3, 4
        knotvector0 = GeneratorKnotVector.bezier(degree)
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

        knotvector1 = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve1 = Curve(knotvector1, ctrlpoints)
        assert curve0 != curve1

    @pytest.mark.order(4)
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

    @pytest.mark.order(4)
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

    @pytest.mark.order(4)
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
    @pytest.mark.order(4)
    @pytest.mark.dependency(
        depends=["TestCompare::test_end", "TestInitCurve::test_end"]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestCallShape::test_begin"])
    def test_build_scalar(self):
        for degree in range(1, 5):
            npts = degree + 1
            knotvector = GeneratorKnotVector.bezier(degree)
            ctrlpoints = np.random.uniform(-1, 1, npts)
            Curve(knotvector, ctrlpoints)

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestCallShape::test_begin"])
    def test_build_vectorial(self):
        for degree in range(1, 5):
            npts = degree + 1
            knotvector = GeneratorKnotVector.bezier(degree)
            for ndim in range(1, 5):
                ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                Curve(knotvector, ctrlpoints)

    @pytest.mark.order(4)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_build_scalar",
        ]
    )
    def test_callscal_scalpts(self):
        for degree in range(1, 5):
            npts = degree + 1
            knotvector = GeneratorKnotVector.bezier(degree)
            ctrlpoints = np.random.uniform(-1, 1, npts)
            curve = Curve(knotvector, ctrlpoints)

            umin, umax = knotvector[0], knotvector[-1]
            tparam = np.random.uniform(umin, umax)
            curvevalues = curve(tparam)
            assert type(curvevalues) == type(ctrlpoints[0])

    @pytest.mark.order(4)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestCallShape::test_begin",
            "TestCallShape::test_build_vectorial",
        ]
    )
    def test_callscal_vectpts(self, ntests=1):
        for degree in range(1, 5):
            npts = degree + 1
            for ndim in range(1, 5):
                knotvector = GeneratorKnotVector.bezier(degree)
                ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                curve = Curve(knotvector, ctrlpoints)
                umin, umax = knotvector[0], knotvector[-1]
                tparam = np.random.uniform(umin, umax)
                curvevalues = curve(tparam)
                assert len(curvevalues) == ndim
                assert type(curvevalues) == type(ctrlpoints[0])
                assert type(curvevalues[0]) == type(ctrlpoints[0][0])

    @pytest.mark.order(4)
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
            npts = degree + 1
            knotvector = GeneratorKnotVector.bezier(degree)
            ctrlpoints = np.random.uniform(-1, 1, npts)
            curve = Curve(knotvector, ctrlpoints)

            lower = npts + degree + 2
            upper = npts + degree + 11
            nsample = np.random.randint(lower, upper)
            umin, umax = knotvector[0], knotvector[-1]
            tparam = np.linspace(umin, umax, nsample)
            Cval = curve(tparam)
            assert len(Cval) == nsample
            assert type(Cval[0]) == type(ctrlpoints[0])

    @pytest.mark.order(4)
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
            npts = degree + 1
            for ndim in range(1, 5):
                knotvector = GeneratorKnotVector.bezier(degree)
                ctrlpoints = np.random.uniform(-1, 1, (npts, ndim))
                curve = Curve(knotvector, ctrlpoints)

                lower = npts + degree + 2
                upper = npts + degree + 11
                nsample = np.random.randint(lower, upper)
                umin, umax = knotvector[0], knotvector[-1]
                tparam = np.linspace(umin, umax, nsample)
                curvevalues = curve(tparam)
                assert len(curvevalues) == nsample
                assert type(curvevalues[0]) == type(ctrlpoints[0])
                assert np.array(curvevalues).shape == (nsample, ndim)

    @pytest.mark.order(4)
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
    @pytest.mark.order(4)
    @pytest.mark.dependency(
        depends=["TestCompare::test_end", "TestCallShape::test_end"]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(4)
    @pytest.mark.dependency(depends=["TestSumSubtract::test_begin"])
    def test_sumsub_failknotvector(self):
        """
        If the knotvectors are different, it's not possible to sum
        It's expected a ValueError
        """
        for degree in range(1, 5):
            npts = degree + 1
            U1 = GeneratorKnotVector.bezier(degree)
            U2 = KnotVector(np.array(U1) + 0.5)
            P1 = np.random.uniform(-1, 1, npts)
            P2 = np.random.uniform(-1, 1, npts)
            C1 = Curve(U1, P1)
            C2 = Curve(U2, P2)
            with pytest.raises(ValueError):
                C1 + C2
            with pytest.raises(ValueError):
                C1 - C2

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestSumSubtract::test_begin"])
    def test_sumsub_scalar(self):
        """
        Tests if the sum of two curves is equal to the new
        curve obtained by summing the control points
        """
        for degree in range(1, 5):
            npts = degree + 1
            knotvector = GeneratorKnotVector.bezier(degree)
            P1 = np.random.uniform(-1, 1, npts)
            P2 = np.random.uniform(-1, 1, npts)
            C1 = Curve(knotvector, P1)
            C2 = Curve(knotvector, P2)
            Cadd = Curve(knotvector, P1 + P2)
            Csub = Curve(knotvector, P1 - P2)
            assert (C1 + C2) == Cadd
            assert (C1 - C2) == Csub

    @pytest.mark.order(4)
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
            npts = degree + 1
            ndim = np.random.randint(1, 4)
            knotvector = GeneratorKnotVector.bezier(degree)
            P1 = np.random.uniform(-1, 1, (npts, ndim))
            P2 = np.random.uniform(-1, 1, (npts, ndim))
            C1 = Curve(knotvector, P1)
            C2 = Curve(knotvector, P2)
            Cadd = Curve(knotvector, P1 + P2)
            Csub = Curve(knotvector, P1 - P2)
            assert (C1 + C2) == Cadd
            assert (C1 - C2) == Csub

    @pytest.mark.order(4)
    @pytest.mark.dependency(
        depends=["TestSumSubtract::test_begin", "TestSumSubtract::test_sumsub_vector"]
    )
    def test_somefails(self):
        degree, npts = 3, 4
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts)
        curve = Curve(knotvector, ctrlpoints)
        with pytest.raises(TypeError):
            curve + 1
        with pytest.raises(TypeError):
            curve + "asd"

    @pytest.mark.order(4)
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


class TestDegreeOperations:
    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestCompare::test_end",
            "TestCallShape::test_end",
            "TestSumSubtract::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
        ]
    )
    def test_increase_once_degree1(self):
        degree, npts = 1, 2
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts)

        curve = Curve(knotvector, ctrlpoints)
        curve.degree += 1
        assert curve.degree == (degree + 1)
        matrix = [[1, 0], [1 / 2, 1 / 2], [0, 1]]
        correctctrlpoints = matrix @ ctrlpoints
        np.testing.assert_allclose(curve.ctrlpoints, correctctrlpoints)

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_increase_once_degree1",
        ]
    )
    def test_increase_once_degree2(self):
        degree, npts = 2, 3
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts)

        curve = Curve(knotvector, ctrlpoints)
        curve.degree += 1
        assert curve.degree == (degree + 1)
        matrix = [[1, 0, 0], [1 / 3, 2 / 3, 0], [0, 2 / 3, 1 / 3], [0, 0, 1]]
        correctctrlpoints = matrix @ ctrlpoints
        np.testing.assert_allclose(curve.ctrlpoints, correctctrlpoints)

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_increase_once_degree1",
            "TestDegreeOperations::test_increase_once_degree2",
        ]
    )
    def test_increase_once_degree3(self):
        degree, npts = 3, 4
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, (npts, 2))

        curve = Curve(knotvector, ctrlpoints)
        curve.degree += 1
        assert curve.degree == (degree + 1)
        matrix = [
            [1, 0, 0, 0],
            [1 / 4, 3 / 4, 0, 0],
            [0, 1 / 2, 1 / 2, 0],
            [0, 0, 3 / 4, 1 / 4],
            [0, 0, 0, 1],
        ]
        Pgood = matrix @ ctrlpoints
        np.testing.assert_allclose(curve.ctrlpoints, Pgood)

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_increase_once_degree1",
            "TestDegreeOperations::test_increase_once_degree2",
            "TestDegreeOperations::test_increase_once_degree3",
        ]
    )
    def test_increase_4times_degree2(self):
        """Example at page 205 of nurbs book"""
        degree, npts = 2, 3
        knotvector = GeneratorKnotVector.bezier(degree)
        ctrlpoints = np.random.uniform(-1, 1, npts)

        curve = Curve(knotvector, ctrlpoints)
        curve.degree += 4
        assert curve.degree == (degree + 4)
        matrix = [
            [1, 0, 0],
            [4 / 6, 2 / 6, 0],
            [6 / 15, 8 / 15, 1 / 15],
            [4 / 20, 12 / 20, 4 / 20],
            [1 / 15, 8 / 15, 6 / 15],
            [0, 2 / 6, 4 / 6],
            [0, 0, 1],
        ]
        correctctrlpoints = matrix @ ctrlpoints
        np.testing.assert_allclose(curve.ctrlpoints, correctctrlpoints)

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_increase_once_degree1",
            "TestDegreeOperations::test_increase_once_degree2",
            "TestDegreeOperations::test_increase_once_degree3",
            "TestDegreeOperations::test_increase_4times_degree2",
        ]
    )
    def test_increase_random(self):
        for degree in range(1, 6):
            for times in range(1, 6):
                npts = degree + 1
                knotvector = GeneratorKnotVector.bezier(degree)
                ctrlpoints = np.random.uniform(-1, 1, npts)
                curve = Curve(knotvector, ctrlpoints)
                curve.degree += times
                assert curve.degree == (degree + times)
                np.testing.assert_allclose(curve.ctrlpoints[0], ctrlpoints[0])
                np.testing.assert_allclose(curve.ctrlpoints[-1], ctrlpoints[-1])

    @pytest.mark.order(4)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_increase_random",
        ]
    )
    def test_decrease_random(self):
        for degree in range(1, 6):
            for times in range(1, 6):
                npts = degree + 1
                knotvector = GeneratorKnotVector.bezier(degree)
                ctrlpoints = np.random.uniform(-1, 1, npts)
                curve = Curve(knotvector, ctrlpoints)
                curve.degree += times
                assert curve.degree == (degree + times)
                np.testing.assert_allclose(curve.ctrlpoints[0], ctrlpoints[0])
                np.testing.assert_allclose(curve.ctrlpoints[-1], ctrlpoints[-1])

    @pytest.mark.order(4)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_increase_random",
            "TestDegreeOperations::test_decrease_random",
        ]
    )
    def test_increase_decrease_random(self):
        for degree in range(1, 4):
            times = np.random.randint(1, 6)
            npts = degree + 1
            knotvector = GeneratorKnotVector.bezier(degree)
            original_ctrlpoints = np.random.uniform(-1, 1, npts)
            curve = Curve(knotvector, original_ctrlpoints)
            curve.degree += times
            assert curve.degree == degree + times
            curve.degree -= times
            assert curve.degree == degree
            np.testing.assert_allclose(curve.ctrlpoints, curve.ctrlpoints)

    @pytest.mark.order(4)
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
            npts = degree + 1
            knotvector = GeneratorKnotVector.bezier(degree)
            ctrlpoints = np.random.uniform(-1, 1, npts)
            curve = Curve(knotvector, ctrlpoints)
            curve.degree += times
            curve.degree_clean()
            assert curve.degree == degree

    @pytest.mark.order(4)
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

    @pytest.mark.order(4)
    @pytest.mark.dependency(
        depends=[
            "TestDegreeOperations::test_begin",
            "TestDegreeOperations::test_clean",
            "TestDegreeOperations::test_fails",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(4)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestInitCurve::test_end",
        "TestCallShape::test_end",
        "TestSumSubtract::test_end",
        "TestDegreeOperations::test_end",
    ]
)
def test_end():
    pass
