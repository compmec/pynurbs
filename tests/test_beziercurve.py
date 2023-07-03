import pytest


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_functions.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestInitBezierCurve:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestInitBezierCurve::test_begin"])
    def test_end(self):
        pass


@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "TestInitBezierCurve::test_end"])
def test_end():
    pass
