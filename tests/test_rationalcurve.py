import functools
from copy import copy
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
        degree = 2
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


class TestAddSubMulDiv:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestBuild::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestAddSubMulDiv::test_begin"])
    def test_bezier_known(self):
        degree = 2
        knotvector = GeneratorKnotVector.bezier(degree, frac)
        curvea = Curve(knotvector)
        curveb = Curve(knotvector)
        curvea.ctrlpoints = [0, 1, 2]  # A(u) = 2*u
        curveb.ctrlpoints = [1, 1, 2]  # B(u) = 1 + u^2

        divatob = curvea / curveb  # C(u) = 2*u/(1+u^2)
        assert divatob.knotvector == [0, 0, 0, 1, 1, 1]
        assert divatob.weights == (1, 1, 2)
        assert divatob.ctrlpoints == (0, 1, 1)

    @pytest.mark.order(6)
    @pytest.mark.timeout(10)
    @pytest.mark.skip(
        reason="Standard fraction fails due to lack of precision. sympy.Rational works"
    )
    @pytest.mark.dependency(
        depends=["TestAddSubMulDiv::test_begin", "TestAddSubMulDiv::test_bezier_known"]
    )
    def test_random_bezier_fractions(self):
        maxdenom = 2
        for degree in range(1, 4):
            knotvector = GeneratorKnotVector.bezier(degree, frac)
            npts = knotvector.npts
            curvea = Curve(knotvector)
            curveb = Curve(knotvector)
            randnumbers = [np.random.randint(maxdenom + 1) for i in range(npts)]
            curvea.ctrlpoints = [1 + frac(num, maxdenom) for num in randnumbers]
            randnumbers = [np.random.randint(maxdenom + 1) for i in range(npts)]
            curvea.weights = [1 + frac(num, maxdenom) for num in randnumbers]
            randnumbers = [np.random.randint(maxdenom + 1) for i in range(npts)]
            curveb.ctrlpoints = [1 + frac(num, maxdenom) for num in randnumbers]
            randnumbers = [np.random.randint(maxdenom + 1) for i in range(npts)]
            curveb.weights = [1 + frac(num, maxdenom) for num in randnumbers]

            aaddb = curvea + curveb
            badda = curveb + curvea
            asubb = curvea - curveb
            bsuba = curveb - curvea
            amulb = curvea * curveb
            bmula = curveb * curvea
            adivb = curvea / curveb
            bdiva = curveb / curvea

            randnumbers = [np.random.randint(maxdenom + 1) for i in range(npts)]
            usample = [frac(num, maxdenom) for num in range(maxdenom + 1)]
            avals = curvea(usample)
            bvals = curveb(usample)
            for ai, bi, ui in zip(avals, bvals, usample):
                assert abs(aaddb(ui) - (ai + bi)) < 1e-9
                assert abs(badda(ui) - (bi + ai)) < 1e-9
                assert abs(asubb(ui) - (ai - bi)) < 1e-9
                assert abs(bsuba(ui) - (bi - ai)) < 1e-9
                assert abs(amulb(ui) - (ai * bi)) < 1e-9
                assert abs(bmula(ui) - (bi * ai)) < 1e-9
                assert abs(adivb(ui) - (ai / bi)) < 1e-9
                assert abs(bdiva(ui) - (bi / ai)) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=["TestAddSubMulDiv::test_begin", "TestAddSubMulDiv::test_bezier_known"]
    )
    def test_random_bezier_float64(self):
        for degree in range(1, 4):
            knotvector = GeneratorKnotVector.bezier(degree, np.float64)
            npts = knotvector.npts
            curvea = Curve(knotvector)
            curveb = Curve(knotvector)
            curvea.ctrlpoints = np.random.uniform(1, 2, npts)
            curvea.weights = np.random.uniform(1, 2, npts)
            curveb.ctrlpoints = np.random.uniform(1, 2, npts)
            curveb.weights = np.random.uniform(1, 2, npts)

            aaddb = curvea + curveb
            badda = curveb + curvea
            asubb = curvea - curveb
            bsuba = curveb - curvea
            amulb = curvea * curveb
            bmula = curveb * curvea
            adivb = curvea / curveb
            bdiva = curveb / curvea

            usample = np.linspace(0, 1, 9)
            avals = curvea(usample)
            bvals = curveb(usample)
            for ai, bi, ui in zip(avals, bvals, usample):
                assert abs(aaddb(ui) - (ai + bi)) < 1e-9
                assert abs(badda(ui) - (bi + ai)) < 1e-9
                assert abs(asubb(ui) - (ai - bi)) < 1e-9
                assert abs(bsuba(ui) - (bi - ai)) < 1e-9
                assert abs(amulb(ui) - (ai * bi)) < 1e-9
                assert abs(bmula(ui) - (bi * ai)) < 1e-9
                assert abs(adivb(ui) - (ai / bi)) < 1e-9
                assert abs(bdiva(ui) - (bi / ai)) < 1e-9

    @pytest.mark.order(6)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=["TestAddSubMulDiv::test_begin", "TestAddSubMulDiv::test_bezier_known"]
    )
    def test_others(self):
        knotvector = GeneratorKnotVector.bezier(3)
        curve = Curve(knotvector)

        curve.ctrlpoints = [2, 2, 2, 2]
        inverse = 1 / curve
        assert id(inverse) != id(curve)
        assert inverse != curve
        invinverse = 1 / inverse

        assert id(invinverse) != id(curve)
        assert invinverse == curve

        curve.weights = [1, 1, 1, 1]
        inverse = 1 / curve
        assert id(inverse) != id(curve)
        assert inverse != curve
        invinverse = 1 / inverse

        assert id(invinverse) != id(curve)
        assert invinverse == curve

    @pytest.mark.order(6)
    @pytest.mark.timeout(10)
    @pytest.mark.dependency(
        depends=["TestAddSubMulDiv::test_begin", "TestAddSubMulDiv::test_bezier_known"]
    )
    def test_zero_division(self):
        knotvector = GeneratorKnotVector.bezier(3)
        curve = Curve(knotvector)

        curve.ctrlpoints = [2, 2, 2, 2]
        with pytest.raises(ValueError):
            curve.weights = [1, -3, 3, -1]
        with pytest.raises(ValueError):
            curve.weights = [1, -3, 5, -1]
        with pytest.raises(ValueError):
            curve.weights = [1, -4, 5, -1]

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestAddSubMulDiv::test_begin",
            "TestAddSubMulDiv::test_bezier_known",
            "TestAddSubMulDiv::test_random_bezier_float64",
            "TestAddSubMulDiv::test_others",
            "TestAddSubMulDiv::test_zero_division",
        ]
    )
    def test_end(self):
        pass


class TestCircle:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestAddSubMulDiv::test_end"])
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
        for degree in range(1, 4):
            for npts in range(degree + 1, degree + 3):
                knotvector = GeneratorKnotVector.uniform(degree, npts, frac)
                randnums = [np.random.randint(denmax + 1) for i in range(npts)]
                ctrlpoints = [frac(num, denmax) for num in randnums]
                oldcurve = Curve(knotvector)
                oldcurve.ctrlpoints = ctrlpoints

                newcurve = copy(oldcurve)
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
        for degree in range(1, 4):
            for npts in range(degree + 1, degree + 3):
                knotvector = GeneratorKnotVector.uniform(degree, npts, frac)
                randnums = [np.random.randint(denmax + 1) for i in range(npts)]
                ctrlpoints = [frac(num, denmax) for num in randnums]
                weights = [frac(1)] * npts
                oldcurve = Curve(knotvector)
                oldcurve.ctrlpoints = ctrlpoints
                oldcurve.weights = weights

                newcurve = copy(oldcurve)
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
        for degree in range(1, 4):
            for npts in range(degree + 1, degree + 3):
                knotvector = GeneratorKnotVector.uniform(degree, npts, frac)
                randnums = [np.random.randint(denmax + 1) for i in range(npts)]
                ctrlpoints = [frac(num, denmax) for num in randnums]
                weights = [frac(np.random.randint(1, denmax + 1), denmax)] * npts
                oldcurve = Curve(knotvector)
                oldcurve.ctrlpoints = ctrlpoints
                oldcurve.weights = weights

                newcurve = copy(oldcurve)
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
        for degree in range(1, 4):
            for npts in range(degree + 1, degree + 3):
                knotvector = GeneratorKnotVector.uniform(degree, npts, frac)
                randnums = [np.random.randint(denmax + 1) for i in range(npts)]
                ctrlpoints = [frac(num, denmax) for num in randnums]
                randnums = [np.random.randint(1, denmax + 1) for i in range(npts)]
                weights = [frac(num, denmax) for num in randnums]
                oldcurve = Curve(knotvector)
                oldcurve.ctrlpoints = ctrlpoints
                oldcurve.weights = weights

                newcurve = copy(oldcurve)
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

        newcurve = copy(oldcurve)
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

        newcurve = copy(curve)
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

        newcurve = copy(curve)
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

        newcurve = copy(curve)
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


class TestCleanRational:
    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestRandomInsertKnot::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(6)
    @pytest.mark.timeout(1)
    @pytest.mark.skip(reason="Needs correction")
    @pytest.mark.dependency(depends=["TestInsKnotCircle::test_begin"])
    def test_divpolybezier(self):
        degree_num, degree_den = 3, 1
        assert degree_num > degree_den
        roots_quo, roots_den = [], []
        while len(roots_den) < degree_den:
            number = np.random.uniform(-1, 2)
            number = frac(number).limit_denominator(4)
            if number < -0.2 or 1.2 < number:
                roots_den.append(number)
        for i in range(degree_num - degree_den):
            number = np.random.uniform(0.2, 0.8)
            number = frac(number).limit_denominator(4)
            roots_quo.append(number)

        prod = lambda vals: functools.reduce(lambda a, b: a * b, vals)
        funct_denom = lambda x: prod([x - v for v in roots_den])
        funct_quoti = lambda x: prod([x - v for v in roots_quo])
        funct_numer = lambda x: funct_quoti(x) * funct_denom(x)

        vector_num = GeneratorKnotVector.bezier(degree_num, frac)
        vector_den = GeneratorKnotVector.bezier(degree_den, frac)
        vector_quo = GeneratorKnotVector.bezier(degree_num - degree_den, frac)
        curve_numer = Curve(vector_num)
        curve_denom = Curve(vector_den)
        curve_quoti = Curve(vector_quo)
        curve_numer.fit_function(funct_numer)
        curve_denom.fit_function(funct_denom)
        curve_denom.degree_increase(degree_num - degree_den)
        curve_quoti.fit_function(funct_quoti)

        test_curve = curve_numer / curve_denom
        test_curve.clean()

        good_vector = GeneratorKnotVector.bezier(degree_num - degree_den, frac)
        good_curve = Curve(good_vector)
        good_curve.fit_function(funct_quoti)

        assert test_curve == good_curve

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
