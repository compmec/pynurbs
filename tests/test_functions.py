import numpy as np
import pytest

from compmec.nurbs import Function
from compmec.nurbs.heavy import binom
from compmec.nurbs.knotspace import GeneratorKnotVector


@pytest.mark.order(3)
@pytest.mark.dependency(depends=["tests/test_knotspace.py::test_end"], scope="session")
def test_begin():
    pass


class TestBezier:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestBezier::test_begin"])
    def test_creation(self):
        bezier = Function([0, 0, 1, 1])
        assert callable(bezier)
        assert bezier.degree == 1
        assert bezier.npts == 2
        bezier = Function([0, 0, 0, 1, 1, 1])
        assert callable(bezier)
        assert bezier.degree == 2
        assert bezier.npts == 3

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestBezier::test_creation"])
    def test_random_creation(self):
        for degree in range(1, 6):
            knotvector = GeneratorKnotVector.bezier(degree)
            bezier = Function(knotvector)
            assert callable(bezier)
            assert bezier.degree == degree
            assert bezier.npts == degree + 1

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_random_creation"])
    def test_evalfuncs_degree1(self):
        bezier = Function([0, 0, 1, 1])
        assert bezier.degree == 1
        assert bezier.npts == 2
        assert callable(bezier[0, 0])
        assert callable(bezier[1, 0])
        assert callable(bezier[0, 1])
        assert callable(bezier[1, 1])
        assert callable(bezier[:, 0])
        assert callable(bezier[:, 1])
        assert callable(bezier[:])

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_random_creation"])
    def test_evalfuncs_degree2(self):
        bezier = Function([0, 0, 0, 1, 1, 1])
        assert bezier.degree == 2
        assert bezier.npts == 3
        assert callable(bezier[0, 0])
        assert callable(bezier[1, 0])
        assert callable(bezier[2, 0])
        assert callable(bezier[0, 1])
        assert callable(bezier[1, 1])
        assert callable(bezier[2, 1])
        assert callable(bezier[0, 2])
        assert callable(bezier[1, 2])
        assert callable(bezier[2, 2])
        assert callable(bezier[:, 0])
        assert callable(bezier[:, 1])
        assert callable(bezier[:, 2])
        assert callable(bezier[:])

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_evalfuncs_degree1",
            "TestBezier::test_evalfuncs_degree2",
        ]
    )
    def test_shape_calls(self):
        for degree in range(1, 6):
            npts = degree + 1
            vector = GeneratorKnotVector.bezier(degree)
            bezier = Function(vector)
            assert bezier.degree == degree
            assert bezier.npts == npts

            npts_sample = 33
            nodes_test = np.linspace(vector[0], vector[-1], npts_sample)
            for j in range(degree + 1):
                for i in range(npts):
                    for node in nodes_test:
                        value = bezier[i, j](node)
                        float(value)  # Verify if it's a float value
                for node in nodes_test:
                    values = bezier[:, j](node)
                    values = np.array(values, dtype="float64")
                    assert values.shape == (npts,)
                matrix = bezier[:, j](nodes_test)
                matrix = np.array(matrix, dtype="float64")
                assert matrix.shape == (npts, npts_sample)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_shape_calls"])
    def test_sum_equal_to_1(self):
        for degree in range(1, 6):
            npts = degree + 1
            vector = GeneratorKnotVector.bezier(degree)
            bezier = Function(vector)
            assert bezier.degree == degree
            assert bezier.npts == npts

            npts_sample = 33
            nodes_test = np.linspace(vector[0], vector[-1], npts_sample)
            for j in range(degree + 1):
                matrix = bezier[:, j](nodes_test)
                matrix = np.array(matrix, dtype="float64")
                assert matrix.shape == (npts, npts_sample)
                assert np.all(matrix >= 0)
                for k in range(npts_sample):
                    assert abs(np.sum(matrix[:, k]) - 1) < 1e-9

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_shape_calls"])
    def test_standard_index(self):
        for degree in range(1, 6):
            npts = degree + 1
            vector = GeneratorKnotVector.bezier(degree)
            bezier = Function(vector)
            assert bezier.degree == degree
            assert bezier.npts == npts

            npts_sample = 33
            nodes_test = np.linspace(vector[0], vector[-1], npts_sample)
            matrix_dire = bezier(nodes_test)
            matrix_none = bezier[:](nodes_test)
            matrix_degr = bezier[:, degree](nodes_test)

            np.testing.assert_allclose(matrix_dire, matrix_degr)
            np.testing.assert_allclose(matrix_none, matrix_degr)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_evalfuncs_degree1",
            "TestBezier::test_standard_index",
        ]
    )
    def test_singlevalues_degree1(self):
        knotvector = [0, 0, 1, 1]  # degree = 1, npts = 2
        bezier = Function(knotvector)
        assert bezier[0, 0](0.0) == 0
        assert bezier[0, 0](0.5) == 0
        assert bezier[0, 0](1.0) == 0
        assert bezier[1, 0](0.0) == 1
        assert bezier[1, 0](0.5) == 1
        assert bezier[1, 0](1.0) == 1
        assert bezier[0, 1](0.0) == 1
        assert bezier[0, 1](0.5) == 0.5
        assert bezier[0, 1](1.0) == 0
        assert bezier[1, 1](0.0) == 0
        assert bezier[1, 1](0.5) == 0.5
        assert bezier[1, 1](1.0) == 1

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_evalfuncs_degree2",
            "TestBezier::test_singlevalues_degree1",
        ]
    )
    def test_singlevalues_degree2(self):
        knotvector = [0, 0, 0, 1, 1, 1]  # degree = 2, npts = 3
        bezier = Function(knotvector)
        assert bezier[0, 0](0.0) == 0
        assert bezier[0, 0](0.5) == 0
        assert bezier[0, 0](1.0) == 0
        assert bezier[1, 0](0.0) == 0
        assert bezier[1, 0](0.5) == 0
        assert bezier[1, 0](1.0) == 0
        assert bezier[2, 0](0.0) == 1
        assert bezier[2, 0](0.5) == 1
        assert bezier[2, 0](1.0) == 1
        assert bezier[0, 1](0.0) == 0
        assert bezier[0, 1](0.5) == 0
        assert bezier[0, 1](1.0) == 0
        assert bezier[1, 1](0.0) == 1
        assert bezier[1, 1](0.5) == 0.5
        assert bezier[1, 1](1.0) == 0
        assert bezier[2, 1](0.0) == 0
        assert bezier[2, 1](0.5) == 0.5
        assert bezier[2, 1](1.0) == 1
        assert bezier[0, 2](0.0) == 1
        assert bezier[0, 2](0.5) == 0.25
        assert bezier[0, 2](1.0) == 0
        assert bezier[1, 2](0.0) == 0
        assert bezier[1, 2](0.5) == 0.5
        assert bezier[1, 2](1.0) == 0
        assert bezier[2, 2](0.0) == 0
        assert bezier[2, 2](0.5) == 0.25
        assert bezier[2, 2](1.0) == 1

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_singlevalues_degree1",
            "TestBezier::test_shape_calls",
        ]
    )
    def test_tablevalues_degree1(self):
        bezier = Function([0, 0, 1, 1])
        assert bezier.degree == 1
        assert bezier.npts == 2
        nodes_test = np.linspace(0, 1, 11)

        matrix_test = bezier[:, 0](nodes_test)
        matrix_good = [[0] * 11, [1] * 11]
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = bezier[:, 1](nodes_test)
        matrix_good = [np.linspace(1, 0, 11), np.linspace(0, 1, 11)]
        np.testing.assert_allclose(matrix_test, matrix_good)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_singlevalues_degree2",
            "TestBezier::test_tablevalues_degree1",
        ]
    )
    def test_tablevalues_degree2(self):
        bezier = Function([0, 0, 0, 1, 1, 1])
        assert bezier.degree == 2
        assert bezier.npts == 3
        nodes_test = np.linspace(0, 1, 11)

        matrix_test = bezier[:, 0](nodes_test)
        matrix_good = np.array([[0] * 11, [0] * 11, [1] * 11])
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = bezier[:, 1](nodes_test)
        matrix_good = [[0] * 11, np.linspace(1, 0, 11), np.linspace(0, 1, 11)]
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = bezier[:, 2](nodes_test)
        matrix_good = np.array(
            [
                [1.0, 0.81, 0.64, 0.49, 0.36, 0.25, 0.16, 0.09, 0.04, 0.01, 0.0],
                [0.0, 0.18, 0.32, 0.42, 0.48, 0.50, 0.48, 0.42, 0.32, 0.18, 0.0],
                [0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0],
            ]
        )
        np.testing.assert_allclose(matrix_test, matrix_good)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_tablevalues_degree1",
            "TestBezier::test_tablevalues_degree2",
        ]
    )
    def test_tablevalues_random_degree(self):
        for degree in range(1, 6):
            knotvector = GeneratorKnotVector.bezier(degree)
            bezier = Function(knotvector)
            assert bezier.degree == degree
            assert bezier.npts == degree + 1

            nodestest = np.linspace(0, 1, 11)
            matrix_test = bezier[:, degree](nodestest)
            matrix_good = np.zeros((degree + 1, len(nodestest)))
            for i, node in enumerate(nodestest):
                for j in range(degree + 1):
                    value = binom(degree, j) * (1 - node) ** (degree - j) * node**j
                    matrix_good[j, i] = value
            np.testing.assert_allclose(matrix_test, matrix_good)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_tablevalues_random_degree"])
    def test_shifted_scaled_bezier(self):
        for degree in range(1, 6):
            knotvector = GeneratorKnotVector.bezier(degree)
            shiftval = np.random.uniform(-1, 1)
            scaleval = np.exp(np.random.uniform(-1, 1))
            # knotvector.shift(shiftval)
            # knotvector.scale(scaleval)
            bezier = Function(knotvector)
            assert bezier.degree == degree
            assert bezier.npts == degree + 1

            nodesgood = np.linspace(0, 1, 11)
            nodestest = np.linspace(knotvector[0], knotvector[-1], 11)
            matrix_test = bezier[:, degree](nodestest)
            matrix_good = np.zeros((degree + 1, len(nodestest)))
            for i, node in enumerate(nodesgood):
                for j in range(degree + 1):
                    value = binom(degree, j) * (1 - node) ** (degree - j) * node**j
                    matrix_good[j, i] = value
            np.testing.assert_allclose(matrix_test, matrix_good)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_shifted_scaled_bezier"])
    def test_degree_operations(self):
        knotvector = GeneratorKnotVector.bezier(3)
        bezier = Function(knotvector)
        assert bezier.degree == 3
        assert bezier.npts == 4
        bezier.degree = 2
        assert bezier.degree == 2
        assert bezier.npts == 3
        bezier.degree -= 1
        assert bezier.degree == 1
        assert bezier.npts == 2

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_begin",
            "TestBezier::test_creation",
            "TestBezier::test_random_creation",
            "TestBezier::test_evalfuncs_degree1",
            "TestBezier::test_evalfuncs_degree2",
            "TestBezier::test_shape_calls",
            "TestBezier::test_sum_equal_to_1",
            "TestBezier::test_standard_index",
            "TestBezier::test_singlevalues_degree1",
            "TestBezier::test_singlevalues_degree2",
            "TestBezier::test_tablevalues_degree1",
            "TestBezier::test_tablevalues_degree2",
            "TestBezier::test_tablevalues_random_degree",
            "TestBezier::test_shifted_scaled_bezier",
            "TestBezier::test_degree_operations",
        ]
    )
    def test_end(self):
        pass


class TestSpline:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestBezier::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestSpline::test_begin"])
    def test_creation(self):
        spline = Function([0, 0, 1, 1])
        assert callable(spline)
        assert spline.degree == 1
        assert spline.npts == 2
        spline = Function([0, 0, 0.5, 1, 1])
        assert callable(spline)
        assert spline.degree == 1
        assert spline.npts == 3
        spline = Function([0, 0, 0, 1, 1, 1])
        assert callable(spline)
        assert spline.degree == 2
        assert spline.npts == 3
        spline = Function([0, 0, 0, 0.5, 1, 1, 1])
        assert callable(spline)
        assert spline.degree == 2
        assert spline.npts == 4

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestSpline::test_creation"])
    def test_random_creation(self):
        for degree in range(1, 6):
            npts = np.random.randint(degree + 1, degree + 9)
            knotvector = GeneratorKnotVector.random(degree, npts)
            spline = Function(knotvector)
            assert callable(spline)
            assert spline.degree == degree
            assert spline.npts == npts

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestSpline::test_random_creation"])
    def test_evalfuncs_degree1npts3(self):
        spline = Function([0, 0, 0.5, 1, 1])
        assert spline.degree == 1
        assert spline.npts == 3
        assert callable(spline[0, 0])
        assert callable(spline[1, 0])
        assert callable(spline[2, 0])
        assert callable(spline[0, 1])
        assert callable(spline[1, 1])
        assert callable(spline[2, 1])
        assert callable(spline[:, 0])
        assert callable(spline[:, 1])
        assert callable(spline[:])

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestSpline::test_evalfuncs_degree1npts3"])
    def test_tablevalues_degree1npts3(self):
        spline = Function([0, 0, 0.5, 1, 1])
        assert spline.degree == 1
        assert spline.npts == 3
        nodes_test = np.linspace(0, 1, 11)

        matrix_test = spline[:, 0](nodes_test)
        matrix_good = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = spline[:, 1](nodes_test)
        matrix_good = [
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestSpline::test_tablevalues_degree1npts3"])
    def test_tablevalues_degree2npts4(self):
        spline = Function([0, 0, 0, 0.5, 1, 1, 1])
        assert spline.degree == 2
        assert spline.npts == 4
        nodes_test = np.linspace(0, 1, 11)

        matrix_test = spline[:, 0](nodes_test)
        matrix_good = [
            [0] * 11,
            [0] * 11,
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = spline[:, 1](nodes_test)
        matrix_good = [
            [0] * 11,
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = spline[:, 2](nodes_test)
        matrix_good = [
            [1, 0.64, 0.36, 0.16, 0.04, 0.0, 0.00, 0.00, 0.00, 0.00, 0],
            [0, 0.34, 0.56, 0.66, 0.64, 0.5, 0.32, 0.18, 0.08, 0.02, 0],
            [0, 0.02, 0.08, 0.18, 0.32, 0.5, 0.64, 0.66, 0.56, 0.34, 0],
            [0, 0.00, 0.00, 0.00, 0.00, 0.0, 0.04, 0.16, 0.36, 0.64, 1],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestSpline::test_tablevalues_degree2npts4"])
    def test_tablevalues_degree3npts5(self):
        knotvector = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
        spline = Function(knotvector)
        assert spline.degree == 3
        assert spline.npts == 5
        nodes_test = np.linspace(0, 1, 11)

        matrix_test = spline[:, 0](nodes_test)
        matrix_good = [
            [0] * 11,
            [0] * 11,
            [0] * 11,
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = spline[:, 1](nodes_test)
        matrix_good = [
            [0] * 11,
            [0] * 11,
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = spline[:, 2](nodes_test)
        matrix_good = [
            [0] * 11,
            [1, 0.64, 0.36, 0.16, 0.04, 0.0, 0.00, 0.00, 0.00, 0.00, 0],
            [0, 0.34, 0.56, 0.66, 0.64, 0.5, 0.32, 0.18, 0.08, 0.02, 0],
            [0, 0.02, 0.08, 0.18, 0.32, 0.5, 0.64, 0.66, 0.56, 0.34, 0],
            [0, 0.00, 0.00, 0.00, 0.00, 0.0, 0.04, 0.16, 0.36, 0.64, 1],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

        matrix_test = spline[:, 3](nodes_test)
        matrix_good = [
            [1, 0.512, 0.216, 0.064, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0],
            [0, 0.434, 0.592, 0.558, 0.416, 0.250, 0.128, 0.054, 0.016, 0.002, 0],
            [0, 0.052, 0.176, 0.324, 0.448, 0.500, 0.448, 0.324, 0.176, 0.052, 0],
            [0, 0.002, 0.016, 0.054, 0.128, 0.250, 0.416, 0.558, 0.592, 0.434, 0],
            [0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.008, 0.064, 0.216, 0.512, 1],
        ]
        np.testing.assert_allclose(matrix_test, matrix_good)

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestSpline::test_tablevalues_degree3npts5",
        ]
    )
    def test_degree_operation(self):
        spline = Function([0, 0, 0, 0, 1, 2, 2, 2, 2])
        assert spline.degree == 3
        assert spline.npts == 5
        spline.degree = 2
        assert spline.degree == 2
        assert spline.npts == 3
        assert spline.knotvector == [0, 0, 0, 2, 2, 2]

        spline = Function([0, 0, 0, 1, 2, 3, 3, 3])
        assert spline.degree == 2
        assert spline.npts == 5
        spline.degree -= 1
        assert spline.degree == 1
        assert spline.npts == 2
        assert spline.knotvector == [0, 0, 3, 3]

        spline = Function([0, 0, 0, 1, 1, 2, 2, 2])
        assert spline.degree == 2
        assert spline.npts == 5
        spline.degree -= 1
        assert spline.degree == 1
        assert spline.npts == 3
        assert spline.knotvector == [0, 0, 1, 2, 2]

        spline = Function([0, 0, 0, 1, 1, 2, 3, 3, 3])
        assert spline.degree == 2
        assert spline.npts == 6
        spline.degree += 1
        assert spline.degree == 3
        assert spline.npts == 9
        assert spline.knotvector == [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3]

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestSpline::test_begin",
            "TestSpline::test_creation",
            "TestSpline::test_random_creation",
            "TestSpline::test_evalfuncs_degree1npts3",
            "TestSpline::test_tablevalues_degree1npts3",
            "TestSpline::test_tablevalues_degree2npts4",
            "TestSpline::test_tablevalues_degree3npts5",
            "TestSpline::test_degree_operation",
        ]
    )
    def test_end(self):
        pass


class TestRational:
    @pytest.mark.order(3)
    @pytest.mark.dependency(depends=["TestSpline::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestRational::test_begin"])
    def test_creation(self):
        for degree in range(1, 6):
            npts = np.random.randint(degree + 1, degree + 9)
            knotvector = GeneratorKnotVector.random(degree, npts)
            rational = Function(knotvector)
            rational.weights = np.random.uniform(0.1, 1, npts)
            assert callable(rational)
            assert rational.degree == degree
            assert rational.npts == npts

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestRational::test_creation"])
    def test_fail_creation(self):
        degree, npts = 3, 7
        knotvector = GeneratorKnotVector.random(degree, npts)
        rational = Function(knotvector)
        with pytest.raises(ValueError):
            rational.weights = 1
        with pytest.raises(ValueError):
            rational.weights = 1 * np.ones(degree)
        with pytest.raises(ValueError):
            rational.weights = -1 * np.ones(npts)

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestRational::test_begin"])
    def test_compare_spline(self):
        knotvector = [0, 0, 0, 1, 2, 2, 2]
        spline = Function(knotvector)
        rational = spline.deepcopy()
        rational.weights = np.ones(rational.npts)

        assert rational == spline

        rational = Function([0, 0, 0, 2, 4, 4, 4])
        rational.weights = np.ones(rational.npts)
        assert rational != spline

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestRational::test_begin"])
    def test_values_rational_equal_spline(self):
        knotvector = [0, 0, 0, 1, 2, 2, 2]
        spline = Function(knotvector)
        rational = spline.deepcopy()
        rational.weights = np.ones(rational.npts)

        nodes_sample = np.linspace(0, 2, 33)
        for node in nodes_sample:
            assert np.all(rational(node) == spline(node))

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestRational::test_begin"])
    def test_quarter_circle_standard(self):
        knotvector = [0, 0, 0, 1, 1, 1]
        rational = Function(knotvector)
        weights = [1, 1, 2]
        rational.weights = weights

        nodes_sample = np.linspace(0, 1, 129)
        good_matrix = [
            (1 - nodes_sample) ** 2,
            2 * nodes_sample * (1 - nodes_sample),
            2 * nodes_sample**2,
        ]
        good_matrix = np.array(good_matrix) / (1 + nodes_sample**2)
        test_matrix = rational(nodes_sample)
        np.testing.assert_allclose(test_matrix, good_matrix)

    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestRational::test_quarter_circle_standard"])
    def test_quarter_circle_symmetric(self):
        knotvector = [0, 0, 0, 1, 1, 1]
        rational = Function(knotvector)
        weights = [2, np.sqrt(2), 2]
        rational.weights = weights

        nodes_sample = np.linspace(0, 1, 129)
        good_matrix = [
            2 * (1 - nodes_sample) ** 2,
            2 * np.sqrt(2) * nodes_sample * (1 - nodes_sample),
            2 * nodes_sample**2,
        ]
        denomin = 2 * (1 - 2 * nodes_sample + 2 * nodes_sample**2)
        denomin += 2 * np.sqrt(2) * nodes_sample * (1 - nodes_sample)
        good_matrix = np.array(good_matrix) / denomin
        test_matrix = rational(nodes_sample)
        np.testing.assert_allclose(test_matrix, good_matrix)

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestRational::test_begin",
            "TestRational::test_creation",
            "TestRational::test_fail_creation",
            "TestRational::test_compare_spline",
            "TestRational::test_values_rational_equal_spline",
            "TestRational::test_quarter_circle_standard",
            "TestRational::test_quarter_circle_symmetric",
        ]
    )
    def test_end(self):
        pass


class TestOthers:
    @pytest.mark.order(3)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestBezier::test_creation",
            "TestSpline::test_creation",
            "TestRational::test_creation",
        ]
    )
    def test_print(self):
        vector_bezier = GeneratorKnotVector.bezier(3)
        vector_spline = GeneratorKnotVector.uniform(3, 5)
        vector_rational = vector_spline.deepcopy()
        weights = np.random.uniform(1, 2, 5)
        bezier = Function(vector_bezier)
        spline = Function(vector_spline)
        rational = Function(vector_rational)
        rational.weights = weights

        bezier.__str__()
        spline.__str__()
        rational.__str__()
        bezier.__repr__()
        spline.__repr__()
        rational.__repr__()

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_creation"])
    def test_specific_cases(self):
        bezier = Function([0, 0, 1, 1])
        assert bezier != 1
        assert bezier != "Asd"
        assert bezier != [0, 0, 1, 1]

        assert bezier.knots[0] == 0
        assert bezier.knots[1] == 1

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_creation"])
    def test_fail_getitem_index(self):
        bezier = Function([0, 0, 1, 1])
        with pytest.raises(IndexError):
            bezier[0, -1]
        with pytest.raises(IndexError):
            bezier[bezier.npts, 0]
        with pytest.raises(IndexError):
            bezier[bezier.npts, 0, 0]
        with pytest.raises(TypeError):
            bezier["asd", 0]
        with pytest.raises(TypeError):
            bezier[0, "asd"]

    @pytest.mark.order(3)
    @pytest.mark.timeout(5)
    @pytest.mark.dependency(depends=["TestBezier::test_creation"])
    def test_fractions(self):
        from fractions import Fraction as frac

        bezier = Function([frac(0), frac(0), frac(1), frac(1)])

        assert type(bezier[0](0)) is frac
        assert type(bezier[1](0)) is frac
        assert type(bezier[0](1)) is frac
        assert type(bezier[1](1)) is frac
        assert type(bezier[0](frac(1, 2))) is frac
        assert type(bezier[1](frac(1, 2))) is frac
        assert type(bezier[0](0.5)) is float
        assert type(bezier[1](0.5)) is float

    @pytest.mark.order(3)
    @pytest.mark.dependency(
        depends=[
            "TestOthers::test_print",
            "TestOthers::test_specific_cases",
            "TestOthers::test_fail_getitem_index",
            "TestOthers::test_fractions",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestBezier::test_end",
        "TestSpline::test_end",
        "TestRational::test_end",
        "TestOthers::test_end",
    ]
)
def test_end():
    pass
