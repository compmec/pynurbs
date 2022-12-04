import numpy as np
import pytest

from compmec.nurbs.algorithms import *


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
    pass


class TestChapter1Algorithms:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestChapter1Algorithms::test_begin"])
    def test_end(self):
        pass


class TestChapter2Algorithms:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestChapter2Algorithms::test_begin"])
    def test_FindSpan(self):
        degree, npts = 2, 3
        U = [0, 0, 0, 1, 1, 1]
        assert Chapter2.FindSpan(npts, degree, 0, U) == degree
        assert Chapter2.FindSpan(npts, degree, 0.2, U) == 2
        assert Chapter2.FindSpan(npts, degree, 0.5, U) == 2
        assert Chapter2.FindSpan(npts, degree, 0.8, U) == 2
        assert Chapter2.FindSpan(npts, degree, 1, U) == npts

        degree, npts = 2, 4
        U = [0, 0, 0, 0.5, 1, 1, 1]
        assert Chapter2.FindSpan(npts, degree, 0, U) == degree
        assert Chapter2.FindSpan(npts, degree, 0.2, U) == 2
        assert Chapter2.FindSpan(npts, degree, 0.5, U) == 3
        assert Chapter2.FindSpan(npts, degree, 0.7, U) == 3
        assert Chapter2.FindSpan(npts, degree, 1, U) == npts

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestChapter2Algorithms::test_begin",
            "TestChapter2Algorithms::test_FindSpan",
        ]
    )
    def test_FindSpanMult(self):
        degree, n = 2, 3
        U = [0, 0, 0, 1, 1, 1]
        assert Chapter2.FindSpanMult(n, degree, 0, U) == (degree, 3)
        assert Chapter2.FindSpanMult(n, degree, 0.2, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, degree, 0.5, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, degree, 0.8, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, degree, 1, U) == (n, 3)

        degree, n = 2, 4
        U = [0, 0, 0, 0.5, 1, 1, 1]
        assert Chapter2.FindSpanMult(n, degree, 0, U) == (degree, 3)
        assert Chapter2.FindSpanMult(n, degree, 0.2, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, degree, 0.5, U) == (3, 1)
        assert Chapter2.FindSpanMult(n, degree, 0.7, U) == (3, 0)
        assert Chapter2.FindSpanMult(n, degree, 1, U) == (n, 3)

        degree, n = 1, 5
        U = [0, 0, 1 / 3, 0.5, 2 / 3, 1, 1]
        assert Chapter2.FindSpanMult(n, degree, 0, U) == (degree, 2)
        assert Chapter2.FindSpanMult(n, degree, 0.2, U) == (1, 0)
        assert Chapter2.FindSpanMult(n, degree, 0.4, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, degree, 0.5, U) == (3, 1)
        assert Chapter2.FindSpanMult(n, degree, 0.7, U) == (4, 0)
        assert Chapter2.FindSpanMult(n, degree, 1, U) == (n, 2)

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestChapter2Algorithms::test_begin",
            "TestChapter2Algorithms::test_FindSpan",
            "TestChapter2Algorithms::test_FindSpanMult",
        ]
    )
    def test_end(self):
        pass


class TestChapter3Algorithms:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestChapter3Algorithms::test_begin"])
    def test_end(self):
        pass


class TestChapter4Algorithms:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["TestChapter4Algorithms::test_begin"])
    def test_end(self):
        pass


class TestChapter5Algorithms:
    @pytest.mark.order(1)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestChapter5Algorithms::test_begin"])
    def test_Distance4D(self):
        assert Chapter5.Distance4D([0, 0, 0, 0], [1, 1, 1, 1]) == 2
        assert Chapter5.Distance4D([0, 0, 0, 0], [4, 3, 0, 0]) == 5
        assert Chapter5.Distance4D([0, 0, 0, 0], [0, 5, 0, 12]) == 13

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestChapter5Algorithms::test_begin"])
    def test_CurveKnotIns(self):
        U = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]  # degree = 3, npts = 6
        knot, times = 0.3, 1
        P = [1, 1, 1, 1, 1, 1]
        Uq, Qw = Chapter5.CurveKnotIns(U, P, knot, times)
        assert len(U) + times == len(Uq)
        assert np.all(np.array(Qw) == 1)
        assert knot in Uq
        for knot in U:
            assert knot in Uq

        U = [0, 0, 1 / 3, 2 / 3, 1, 1]  # degree = 1, npts = 4
        knot, times = 0.5, 1
        P = [1, 2, 3, 4]
        Uq, Qw = Chapter5.CurveKnotIns(U, P, knot, times)
        assert Uq == [0, 0, 1 / 3, 0.5, 2 / 3, 1, 1]
        assert Qw == [1, 2, 2.5, 3, 4]

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestChapter5Algorithms::test_begin",
            "TestChapter2Algorithms::test_FindSpanMult",
        ]
    )
    def test_CurvePntByCornerCut(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestChapter5Algorithms::test_begin",
            "TestChapter2Algorithms::test_FindSpan",
        ]
    )
    def test_RefineKnotVectCurve(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestChapter5Algorithms::test_begin",
            "TestChapter5Algorithms::test_Distance4D",
        ]
    )
    def test_RemoveCurveKnot(self):
        pass

    @pytest.mark.order(1)
    @pytest.mark.timeout(3)
    @pytest.mark.dependency(
        depends=[
            "TestChapter5Algorithms::test_begin",
            "TestChapter5Algorithms::test_CurveKnotIns",
            "TestChapter5Algorithms::test_RemoveCurveKnot",
            "TestChapter2Algorithms::test_FindSpanMult",
        ]
    )
    def test_InsertAndRemoveCurveKnot_known(self):
        degree, npts = 1, 4
        U = [0, 0, 1 / 3, 2 / 3, 1, 1]
        knot, times = 0.5, 1
        Pw = [1, 2, 3, 4]
        Uq, Qw = Chapter5.CurveKnotIns(U, Pw, knot, times)
        assert Uq == [0, 0, 1 / 3, 0.5, 2 / 3, 1, 1]
        assert Qw == [1, 2, 2.5, 3, 4]

        t, Uo, Ow = Chapter5.RemoveCurveKnot(Uq, Qw, knot, times)
        assert Uo == U
        assert Ow == Pw

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestChapter5Algorithms::test_begin",
            "TestChapter5Algorithms::test_InsertAndRemoveCurveKnot_known",
        ]
    )
    def test_InsertAndRemoveCurveKnot_random(self):
        ntests = 100
        for i in range(ntests):
            degree = np.random.randint(1, 5)
            npts = np.random.randint(degree + 1, degree + 11)
            U = [0] * degree + list(np.linspace(0, 1, npts - degree + 1)) + [1] * degree

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestChapter5Algorithms::test_begin",
            "TestChapter5Algorithms::test_InsertAndRemoveCurveKnot_random",
        ]
    )
    def test_end(self):
        pass


class TestCustom:
    @pytest.mark.order(1)
    @pytest.mark.dependency()
    def test_BezDegreeReduce(self, ntests=100):
        for degree in range(2, 5):
            ndim = np.random.randint(1, 2)
            npts = degree + 1
            U = [0] * npts + [1] * npts
            P = np.random.uniform(-1, 1, (npts, ndim))
            newP, error = Custom.BezDegreeReduce(P)
            assert len(newP) == npts - 1


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestChapter1Algorithms::test_end",
        "TestChapter2Algorithms::test_end",
        "TestChapter3Algorithms::test_end",
        "TestChapter4Algorithms::test_end",
        "TestChapter5Algorithms::test_end",
    ]
)
def test_end():
    pass
