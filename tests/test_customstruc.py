"""
This file is responsible for testing the code for omissive custom objects.
Example 1:
    If we define a omissive CustomFloat which
        float + CustomFloat -> float
        float * CustomFloat -> float
        CustomFloat + float -> float
        CustomFloat * float -> float
        CustomFloat + CustomFloat -> CustomFloat
        CustomFloat * CustomFloat -> CustomFloat
    We expect all final computations returns CustomFloats
"""

import numpy as np
import pytest

from pynurbs import calculus
from pynurbs.curves import Curve
from pynurbs.functions import Function
from pynurbs.knotspace import GeneratorKnotVector, KnotVector


class CustomFloat:
    def __init__(self, number: float):
        self.internal = float(number)

    def __hash__(self):
        return id(self)

    def __str__(self) -> str:
        return str(self.internal)

    def __repr__(self) -> str:
        return f"CustomFloat({str(self)})"

    def __float__(self) -> float:
        return float(self.internal)

    def __neg__(self):
        return self.__class__(-self.internal)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return CustomFloat(self.internal + other.internal)
        return float(self) + other

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return CustomFloat(self.internal * other.internal)
        elif isinstance(other, int):
            return CustomFloat(other * self.internal)
        return float(self) * other

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return CustomFloat(self.internal / other.internal)
        if isinstance(other, int):
            return CustomFloat(self.internal / other)
        return float(self) / other

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            return CustomFloat(other.internal / self.internal)
        if isinstance(other, int):
            return CustomFloat(other / self.internal)
        return other / float(self)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __lt__(self, other):
        return float(self) < other

    def __eq__(self, other):
        return float(self) == other

    def __gt__(self, other):
        return float(self) > other

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __abs__(self):
        return self.__class__(self.internal if self.internal > 0 else -self.internal)


class CustomPoint:
    def __init__(self, value: float):
        self.internal = value

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        return self.__class__(self.internal + other.internal)

    def __rmul__(self, other: CustomFloat):
        return self.__class__(other * self.internal)


@pytest.mark.order(7)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_functions.py::test_end",
        "tests/test_beziercurve.py::test_end",
        "tests/test_splinecurve.py::test_end",
        "tests/test_rationalcurve.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(7)
@pytest.mark.dependency(depends=["test_begin"])
def test_custom_float():
    a = CustomFloat(1)
    b = CustomFloat(2)
    assert type(a + b) is CustomFloat
    assert type(a - b) is CustomFloat
    assert type(a * b) is CustomFloat
    assert type(a / b) is CustomFloat

    assert type(a + 1) is float
    assert type(1 + b) is float
    assert type(a - 1) is float
    assert type(1 - b) is float
    assert type(a + 1.0) is float
    assert type(1.0 + b) is float
    assert type(a - 1.0) is float
    assert type(1.0 - b) is float

    assert type(a * 1.0) is float
    assert type(1.0 * b) is float
    assert type(a * 1) is CustomFloat
    assert type(1 * b) is CustomFloat

    assert type(a / 1) is CustomFloat
    assert type(1 / b) is CustomFloat


class TestKnotVector:
    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin", "test_custom_float"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestKnotVector::test_begin"])
    def test_creation(self):
        a, b = CustomFloat(0), CustomFloat(1)
        vector = KnotVector([a, a, b, b])
        assert type(vector[0]) is CustomFloat
        assert type(vector[1]) is CustomFloat
        assert type(vector[2]) is CustomFloat
        assert type(vector[3]) is CustomFloat
        tuple(vector)

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestKnotVector::test_begin", "TestKnotVector::test_creation"]
    )
    def test_end(self):
        pass


class TestBasisFunctions:
    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin", "TestKnotVector::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.timeout(1)
    # @pytest.mark.skip(reason="Needs correction")
    @pytest.mark.dependency(depends=["TestBasisFunctions::test_begin"])
    def test_creation(self):
        a, b = CustomFloat(0), CustomFloat(1)
        vector = KnotVector([a, a, b, b])
        N = Function(vector)
        assert type(N[0](a)) is CustomFloat
        assert type(N[0](b)) is CustomFloat

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestBasisFunctions::test_begin", "TestBasisFunctions::test_creation"]
    )
    def test_end(self):
        pass


@pytest.mark.order(7)
@pytest.mark.dependency(
    depends=["test_begin", "TestKnotVector::test_end", "TestBasisFunctions::test_end"]
)
def test_end():
    pass
