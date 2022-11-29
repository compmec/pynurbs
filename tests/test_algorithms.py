import numpy as np
import pytest

from compmec.nurbs.algorithms import *


@pytest.mark.order(1)
@pytest.mark.skip()
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
        p, n = 2, 3
        U = [0, 0, 0, 1, 1, 1]
        assert Chapter2.FindSpan(n, p, 0, U) == p
        assert Chapter2.FindSpan(n, p, 0.2, U) == 2
        assert Chapter2.FindSpan(n, p, 0.5, U) == 2
        assert Chapter2.FindSpan(n, p, 0.8, U) == 2
        assert Chapter2.FindSpan(n, p, 1, U) == n

        p, n = 2, 4
        U = [0, 0, 0, 0.5, 1, 1, 1]
        assert Chapter2.FindSpan(n, p, 0, U) == p
        assert Chapter2.FindSpan(n, p, 0.2, U) == 2
        assert Chapter2.FindSpan(n, p, 0.5, U) == 3
        assert Chapter2.FindSpan(n, p, 0.7, U) == 3
        assert Chapter2.FindSpan(n, p, 1, U) == n

    @pytest.mark.order(1)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestChapter2Algorithms::test_begin",
            "TestChapter2Algorithms::test_FindSpan",
        ]
    )
    def test_FindSpanMult(self):
        p, n = 2, 3
        U = [0, 0, 0, 1, 1, 1]
        assert Chapter2.FindSpanMult(n, p, 0, U) == (p, 3)
        assert Chapter2.FindSpanMult(n, p, 0.2, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, p, 0.5, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, p, 0.8, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, p, 1, U) == (n, 3)

        p, n = 2, 4
        U = [0, 0, 0, 0.5, 1, 1, 1]
        assert Chapter2.FindSpanMult(n, p, 0, U) == (p, 3)
        assert Chapter2.FindSpanMult(n, p, 0.2, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, p, 0.5, U) == (3, 1)
        assert Chapter2.FindSpanMult(n, p, 0.7, U) == (3, 0)
        assert Chapter2.FindSpanMult(n, p, 1, U) == (n, 3)

        p, n = 1, 5
        U = [0, 0, 1 / 3, 0.5, 2 / 3, 1, 1]
        assert Chapter2.FindSpanMult(n, p, 0, U) == (p, 2)
        assert Chapter2.FindSpanMult(n, p, 0.2, U) == (1, 0)
        assert Chapter2.FindSpanMult(n, p, 0.4, U) == (2, 0)
        assert Chapter2.FindSpanMult(n, p, 0.5, U) == (3, 1)
        assert Chapter2.FindSpanMult(n, p, 0.7, U) == (4, 0)
        assert Chapter2.FindSpanMult(n, p, 1, U) == (n, 2)

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
        p, n = 3, 6
        U = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]
        u, k, s, r = 0.3, 4, 0, 1
        P = [1, 1, 1, 1, 1, 1]
        nq, Uq, Qw = Chapter5.CurveKnotIns(n, p, U, P, u, k, s, r)
        assert nq == n + r
        assert len(U) + r == len(Uq)
        assert np.all(np.array(Qw) == 1)
        assert u in Uq
        for u in U:
            assert u in Uq

        p, n = 1, 4
        U = [0, 0, 1 / 3, 2 / 3, 1, 1]
        u, k, s, r = 0.5, 2, 0, 1
        P = [1, 2, 3, 4]
        nq, Uq, Qw = Chapter5.CurveKnotIns(n, p, U, P, u, k, s, r)
        assert nq == n + r
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
        p, n = 1, 4
        U = [0, 0, 1 / 3, 2 / 3, 1, 1]
        u, k, s, r = 0.5, 2, 0, 1
        Pw = [1, 2, 3, 4]
        nq, Uq, Qw = Chapter5.CurveKnotIns(n, p, U, Pw, u, k, s, r)
        assert nq == n + r
        assert Uq == [0, 0, 1 / 3, 0.5, 2 / 3, 1, 1]
        assert Qw == [1, 2, 2.5, 3, 4]

        k, s = 3, 1
        t, Uo, Ow = Chapter5.RemoveCurveKnot(nq, p, Uq, Qw, u, k, s, r)
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
            p = np.random.randint(1, 5)
            n = np.random.randint(p + 1, p + 11)
            U = [0] * p + list(np.linspace(0, 1, n - p + 1)) + [1] * p

    @pytest.mark.order(1)
    @pytest.mark.dependency(
        depends=[
            "TestChapter5Algorithms::test_begin",
            "TestChapter5Algorithms::test_InsertAndRemoveCurveKnot_random",
        ]
    )
    def test_end(self):
        pass


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
