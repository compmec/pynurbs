import numpy as np
import pytest

from compmec.nurbs.curves import Curve
from compmec.nurbs.knotspace import GeneratorKnotVector


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


class TestCurve:
    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestCurve::test_begin"])
    def test_constant(self):
        vector = GeneratorKnotVector.bezier(0)
        good_curve = Curve(vector)
        constval = np.random.uniform(-1, 1)
        good_curve.ctrlpoints = [constval]
        for degree in range(1, 6):
            vector = GeneratorKnotVector.bezier(degree)
            test_curve = Curve(vector)
            test_curve.fit(good_curve)
            assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestCurve::test_begin", "TestCurve::test_constant"]
    )
    def test_overdefined_spline(self):
        for degree_base in range(0, 4):
            vector = GeneratorKnotVector.bezier(degree_base)
            good_curve = Curve(vector)
            points = np.random.uniform(-1, 1, 1 + degree_base)
            good_curve.ctrlpoints = points
            for degree in range(degree_base, 7):
                vector = GeneratorKnotVector.bezier(degree)
                test_curve = Curve(vector)
                test_curve.fit(good_curve)
                assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestCurve::test_begin",
            "TestCurve::test_constant",
            "TestCurve::test_overdefined_spline",
        ]
    )
    def test_overdefined_rational(self):
        for degree_base in range(0, 4):
            vector = GeneratorKnotVector.bezier(degree_base)
            good_curve = Curve(vector)
            points = np.random.uniform(-1, 1, 1 + degree_base)
            good_curve.ctrlpoints = points
            for degree in range(degree_base, 7):
                vector = GeneratorKnotVector.bezier(degree)
                test_curve = Curve(vector)
                test_curve.weights = [1] * test_curve.npts
                test_curve.fit(good_curve)
                assert test_curve == good_curve

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestCurve::test_begin",
            "TestCurve::test_constant",
            "TestCurve::test_overdefined_spline",
            "TestCurve::test_overdefined_rational",
        ]
    )
    def test_end(self):
        pass


class TestFunction:
    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestFunction::test_begin"])
    def test_constant(self):
        constval = np.random.uniform(-1, 1)
        function = lambda u: constval
        for degree in range(0, 6):
            vector = GeneratorKnotVector.bezier(degree)
            test_curve = Curve(vector)
            test_curve.fit(function)
            for point in test_curve.ctrlpoints:
                assert np.abs(point - constval) < 1e-9

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestFunction::test_begin", "TestFunction::test_constant"]
    )
    def test_overdefined_spline(self):
        usample = np.linspace(0, 1, 33)
        for degree_base in range(0, 4):
            coefs = np.random.uniform(-1, 1, 1 + degree_base)
            function = lambda u: sum([cof * u**i for i, cof in enumerate(coefs)])
            for degree in range(degree_base, 7):
                vector = GeneratorKnotVector.bezier(degree)
                test_curve = Curve(vector)
                test_curve.fit(function)
                for ui in usample:
                    assert np.abs(test_curve(ui) - function(ui)) < 1e-9

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestFunction::test_begin",
            "TestFunction::test_constant",
            "TestFunction::test_overdefined_spline",
        ]
    )
    def test_overdefined_rational(self):
        usample = np.linspace(0, 1, 33)
        for degree_base in range(0, 4):
            coefs = np.random.uniform(-1, 1, 1 + degree_base)
            function = lambda u: sum([cof * u**i for i, cof in enumerate(coefs)])
            for degree in range(degree_base, 7):
                vector = GeneratorKnotVector.bezier(degree)
                test_curve = Curve(vector)
                test_curve.weights = [1] * test_curve.npts
                test_curve.fit(function)
                for ui in usample:
                    assert np.abs(test_curve(ui) - function(ui)) < 1e-9

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestFunction::test_begin",
            "TestFunction::test_constant",
            "TestFunction::test_overdefined_spline",
            "TestFunction::test_overdefined_rational",
        ]
    )
    def test_end(self):
        pass


class TestPoints:
    @pytest.mark.order(7)
    # @pytest.mark.skip(reason="Needs implementation")
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestPoints::test_begin"])
    def test_constant(self):
        usample = np.linspace(0, 1, 9)
        constval = np.random.uniform(-1, 1)
        points = [constval] * len(usample)
        for degree in range(0, 6):
            vector = GeneratorKnotVector.bezier(degree)
            test_curve = Curve(vector)
            test_curve.fit(points)
            for point in test_curve.ctrlpoints:
                assert np.abs(point - constval) < 1e-9

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestPoints::test_begin", "TestPoints::test_constant"]
    )
    def test_overdefined_spline(self):
        usample = np.linspace(0, 1, 33)
        for degree_base in range(0, 4):
            coefs = np.random.uniform(-1, 1, 1 + degree_base)
            values = np.zeros(len(usample), dtype="float64")
            for i, coefi in enumerate(coefs):
                values += coefi * usample**i
            for degree in range(degree_base, 7):
                vector = GeneratorKnotVector.bezier(degree)
                test_curve = Curve(vector)
                test_curve.fit(values)
                for ui, valui in zip(usample, values):
                    assert np.abs(test_curve(ui) - valui) < 1e-9

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestPoints::test_begin",
            "TestPoints::test_constant",
            "TestPoints::test_overdefined_spline",
        ]
    )
    def test_overdefined_rational(self):
        usample = np.linspace(0, 1, 33)
        for degree_base in range(0, 4):
            coefs = np.random.uniform(-1, 1, 1 + degree_base)
            values = np.zeros(len(usample), dtype="float64")
            for i, coefi in enumerate(coefs):
                values += coefi * usample**i
            for degree in range(degree_base, 7):
                vector = GeneratorKnotVector.bezier(degree)
                test_curve = Curve(vector)
                test_curve.weights = [1] * test_curve.npts
                test_curve.fit(values)
                for ui, valui in zip(usample, values):
                    assert np.abs(test_curve(ui) - valui) < 1e-9

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestPoints::test_begin",
            "TestPoints::test_constant",
            "TestPoints::test_overdefined_spline",
            "TestPoints::test_overdefined_rational",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(7)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestCurve::test_end",
        "TestFunction::test_end",
        "TestPoints::test_end",
    ]
)
def test_end():
    pass
