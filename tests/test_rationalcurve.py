from fractions import Fraction as frac

import numpy as np
import pytest

from compmec.nurbs.curves import Curve
from compmec.nurbs.knotspace import GeneratorKnotVector


@pytest.mark.order(6)
@pytest.mark.dependency(
    depends=[
        "tests/test_knotspace.py::test_end",
        "tests/test_functions.py::test_end",
        "tests/test_beziercurve.py::test_end",
        "tests/test_splinecurve.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestBuild:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestBuild::test_begin"])
    def test_failbuild(self):
        for degree in range(1, 6):
            npts = np.random.randint(degree + 1, degree + 3)
            knotvector = GeneratorKnotVector.random(degree, npts)
            curve = Curve(knotvector)
            with pytest.raises(ValueError):
                curve.weights = 1
            with pytest.raises(ValueError):
                curve.weights = "asd"

    @pytest.mark.order(6)
    @pytest.mark.timeout(15)
    @pytest.mark.dependency(depends=["TestBuild::test_failbuild"])
    def test_print(self):
        knotvector = GeneratorKnotVector.uniform(2, 4)
        rational = Curve(knotvector)
        str(rational)
        rational.ctrlpoints = [2, 4, 3, 1]
        str(rational)
        rational.weights = (2, 3, 1, 4)
        str(rational)

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestBuild::test_begin",
            "TestBuild::test_failbuild",
            "TestBuild::test_print",
        ]
    )
    def test_end(self):
        pass


class TestCircle:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestBuild::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestCircle::test_begin"])
    def test_quarter_circle_standard(self):
        knotvector = GeneratorKnotVector.bezier(2, frac)
        ctrlpoints = [(1, 0), (1, 1), (0, 1)]
        weights = [1, 1, 2]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = [frac(weight) for weight in weights]
        nsample = 128
        nodes_sample = [frac(i, nsample) for i in range(nsample + 1)]
        points = curve(nodes_sample)
        for point in points:
            dist2 = sum(point**2)
            assert abs(dist2 - 1) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestCircle::test_quarter_circle_standard"])
    def test_quarter_circle_symmetric(self):
        knotvector = [0, 0, 0, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 1), (0, 1)]
        weights = [2, np.sqrt(2), 2]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = weights
        nsample = 128
        nodes_sample = [frac(i, nsample) for i in range(nsample + 1)]
        points = curve(nodes_sample)
        for point in points:
            dist2 = sum(point**2)
            assert abs(dist2 - 1) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestCircle::test_quarter_circle_standard",
            "TestCircle::test_quarter_circle_symmetric",
        ]
    )
    def test_half_circle(self):
        knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 2), (-1, 2), (-1, 0)]
        weights = [3, 1, 1, 3]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = weights
        nsample = 128
        nodes_sample = [frac(i, nsample) for i in range(nsample + 1)]
        points = curve(nodes_sample)
        for point in points:
            dist2 = sum(point**2)
            assert abs(dist2 - 1) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestCircle::test_quarter_circle_standard",
            "TestCircle::test_quarter_circle_symmetric",
        ]
    )
    def test_full_circle(self):
        knotvector = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 2), (-1, 2), (-1, 0), (-1, -2), (1, -2), (1, 0)]
        weights = [3, 1, 1, 3, 1, 1, 3]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints)
        curve.weights = weights
        nsample = 128
        nodes_sample = [frac(i, nsample) for i in range(nsample + 1)]
        points = curve(nodes_sample)
        for point in points:
            dist2 = sum(point**2)
            assert abs(dist2 - 1) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestCircle::test_begin",
            "TestCircle::test_quarter_circle_standard",
            "TestCircle::test_quarter_circle_symmetric",
            "TestCircle::test_half_circle",
            "TestCircle::test_full_circle",
        ]
    )
    def test_end(self):
        pass


class TestRandomInsertKnot:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestCircle::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(depends=["TestRandomInsertKnot::test_begin"])
    def test_none_weights_fraction(self):
        denmax = 100
        for degree in range(1, 6):
            for npts in range(degree + 1, degree + 3):
                knotvector = GeneratorKnotVector.uniform(degree, npts, frac)
                randnums = [np.random.randint(denmax + 1) for i in range(npts)]
                ctrlpoints = [frac(num, denmax) for num in randnums]
                oldcurve = Curve(knotvector)
                oldcurve.ctrlpoints = ctrlpoints

                newcurve = oldcurve.deepcopy()
                while True:
                    newknot = frac(np.random.randint(denmax), denmax)
                    if oldcurve.knotvector.mult(newknot) == 0:
                        break
                newcurve.knot_insert([newknot])

                nodes_sample = [frac(i, denmax) for i in range(denmax + 1)]
                points_old = oldcurve(nodes_sample)
                points_new = newcurve(nodes_sample)
                for oldpt, newpt in zip(points_old, points_new):
                    diff = oldpt - newpt
                    assert float(diff**2) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestRandomInsertKnot::test_begin",
            "TestRandomInsertKnot::test_none_weights_fraction",
        ]
    )
    def test_unitary_weights_fraction(self):
        denmax = 100
        for degree in range(1, 6):
            for npts in range(degree + 1, degree + 3):
                knotvector = GeneratorKnotVector.uniform(degree, npts, frac)
                randnums = [np.random.randint(denmax + 1) for i in range(npts)]
                ctrlpoints = [frac(num, denmax) for num in randnums]
                weights = [frac(1)] * npts
                oldcurve = Curve(knotvector)
                oldcurve.ctrlpoints = ctrlpoints
                oldcurve.weights = weights

                newcurve = oldcurve.deepcopy()
                while True:
                    newknot = frac(np.random.randint(denmax), denmax)
                    if oldcurve.knotvector.mult(newknot) == 0:
                        break
                newcurve.knot_insert([newknot])

                nodes_sample = [frac(i, denmax) for i in range(denmax + 1)]
                points_old = oldcurve(nodes_sample)
                points_new = newcurve(nodes_sample)
                for oldpt, newpt in zip(points_old, points_new):
                    diff = oldpt - newpt
                    assert float(diff**2) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestRandomInsertKnot::test_begin",
            "TestRandomInsertKnot::test_none_weights_fraction",
            "TestRandomInsertKnot::test_unitary_weights_fraction",
        ]
    )
    def test_const_weights_fraction(self):
        denmax = 100
        for degree in range(1, 6):
            for npts in range(degree + 1, degree + 3):
                knotvector = GeneratorKnotVector.uniform(degree, npts, frac)
                randnums = [np.random.randint(denmax + 1) for i in range(npts)]
                ctrlpoints = [frac(num, denmax) for num in randnums]
                weights = [frac(np.random.randint(1, denmax + 1), denmax)] * npts
                oldcurve = Curve(knotvector)
                oldcurve.ctrlpoints = ctrlpoints
                oldcurve.weights = weights

                newcurve = oldcurve.deepcopy()
                while True:
                    newknot = frac(np.random.randint(1, denmax), denmax)
                    if oldcurve.knotvector.mult(newknot) == 0:
                        break
                newcurve.knot_insert([newknot])

                nodes_sample = [frac(i, denmax) for i in range(denmax + 1)]
                points_old = oldcurve(nodes_sample)
                points_new = newcurve(nodes_sample)
                for oldpt, newpt in zip(points_old, points_new):
                    diff = oldpt - newpt
                    assert float(diff**2) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=[
            "TestRandomInsertKnot::test_begin",
            "TestRandomInsertKnot::test_none_weights_fraction",
            "TestRandomInsertKnot::test_unitary_weights_fraction",
            "TestRandomInsertKnot::test_const_weights_fraction",
        ]
    )
    def test_random_weights_fraction(self):
        denmax = 20
        for degree in range(1, 6):
            for npts in range(degree + 1, degree + 3):
                knotvector = GeneratorKnotVector.uniform(degree, npts, frac)
                randnums = [np.random.randint(denmax + 1) for i in range(npts)]
                ctrlpoints = [frac(num, denmax) for num in randnums]
                randnums = [np.random.randint(1, denmax + 1) for i in range(npts)]
                weights = [frac(num, denmax) for num in randnums]
                oldcurve = Curve(knotvector)
                oldcurve.ctrlpoints = ctrlpoints
                oldcurve.weights = weights

                newcurve = oldcurve.deepcopy()
                while True:
                    newknot = frac(np.random.randint(1, denmax), denmax)
                    if oldcurve.knotvector.mult(newknot) == 0:
                        break
                newcurve.knot_insert([newknot])

                nodes_sample = [frac(i, denmax) for i in range(denmax + 1)]
                points_old = oldcurve(nodes_sample)
                points_new = newcurve(nodes_sample)
                for oldpt, newpt in zip(points_old, points_new):
                    diff = oldpt - newpt
                    assert float(diff**2) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestRandomInsertKnot::test_begin",
            "TestRandomInsertKnot::test_none_weights_fraction",
            "TestRandomInsertKnot::test_unitary_weights_fraction",
            "TestRandomInsertKnot::test_const_weights_fraction",
            "TestRandomInsertKnot::test_random_weights_fraction",
        ]
    )
    def test_end(self):
        pass


class TestInsKnotCircle:
    @pytest.mark.order(6)
    # @pytest.mark.skip(reason="Needs knot insertion correction")
    @pytest.mark.dependency(depends=["TestRandomInsertKnot::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInsKnotCircle::test_begin"])
    def test_quarter_circle_standard(self):
        zero, one = frac(0), frac(1)
        knotvector = GeneratorKnotVector.bezier(2, frac)
        ctrlpoints = [(one, zero), (one, one), (zero, one)]
        weights = [1, 1, 2]
        oldcurve = Curve(knotvector)
        oldcurve.ctrlpoints = np.array(ctrlpoints)
        oldcurve.weights = [frac(weight) for weight in weights]

        newcurve = oldcurve.deepcopy()
        newcurve.knot_insert([frac(1, 2)])

        denmax = 128
        nodes_sample = [frac(i, denmax) for i in range(denmax + 1)]
        points_old = oldcurve(nodes_sample)
        points_new = newcurve(nodes_sample)
        for oldpt, newpt in zip(points_old, points_new):
            diff = oldpt - newpt
            distsquare = sum(diff**2)
            assert float(distsquare) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestInsKnotCircle::test_quarter_circle_standard"])
    def test_quarter_circle_symmetric(self):
        knotvector = [0, 0, 0, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 1), (0, 1)]
        weights = [2, np.sqrt(2), 2]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints, dtype="float64")
        curve.weights = np.array(weights, dtype="float64")

        newcurve = curve.deepcopy()
        newcurve.knot_insert([0.5])

        nodes_sample = np.linspace(0, 1, 129)
        points_old = curve(nodes_sample)
        points_new = newcurve(nodes_sample)
        for oldpt, newpt in zip(points_old, points_new):
            assert abs(np.linalg.norm(oldpt - newpt)) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestInsKnotCircle::test_quarter_circle_standard",
            "TestInsKnotCircle::test_quarter_circle_symmetric",
        ]
    )
    def test_half_circle(self):
        knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 2), (-1, 2), (-1, 0)]
        weights = [3, 1, 1, 3]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints, dtype="float64")
        curve.weights = np.array(weights, dtype="float64")

        newcurve = curve.deepcopy()
        newcurve.knot_insert([0.5])

        nodes_sample = np.linspace(0, 1, 129)
        points_old = curve(nodes_sample)
        points_new = newcurve(nodes_sample)
        for oldpt, newpt in zip(points_old, points_new):
            assert abs(np.linalg.norm(oldpt - newpt)) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestInsKnotCircle::test_quarter_circle_standard",
            "TestInsKnotCircle::test_quarter_circle_symmetric",
        ]
    )
    def test_full_circle(self):
        knotvector = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]
        ctrlpoints = [(1, 0), (1, 2), (-1, 2), (-1, 0), (-1, -2), (1, -2), (1, 0)]
        weights = [3, 1, 1, 3, 1, 1, 3]
        curve = Curve(knotvector)
        curve.ctrlpoints = np.array(ctrlpoints, dtype="float64")
        curve.weights = np.array(weights, dtype="float64")

        newcurve = curve.deepcopy()
        newcurve.knot_insert([0.25, 0.75])

        nodes_sample = np.linspace(0, 1, 129)
        points_old = curve(nodes_sample)
        points_new = newcurve(nodes_sample)
        for oldpt, newpt in zip(points_old, points_new):
            assert abs(np.linalg.norm(oldpt - newpt)) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestInsKnotCircle::test_begin",
            "TestInsKnotCircle::test_quarter_circle_standard",
            "TestInsKnotCircle::test_quarter_circle_symmetric",
            "TestInsKnotCircle::test_half_circle",
            "TestInsKnotCircle::test_full_circle",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(6)
@pytest.mark.dependency(
    depends=["test_begin", "TestCircle::test_end", "TestInsKnotCircle::test_end"]
)
def test_end():
    pass
