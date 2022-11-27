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
    @pytest.mark.dependency(depends=["TestChapter1Algorithms::test_begin"])
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
    @pytest.mark.dependency(
        depends=[
            "TestChapter2Algorithms::test_begin",
            "TestChapter2Algorithms::test_FindSpan",
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
    @pytest.mark.dependency(depends=["TestChapter5Algorithms::test_begin"])
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
