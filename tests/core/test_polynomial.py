import pytest

from pynurbs.core.custom_math import Math
from pynurbs.core.polynomial import Polynomial, derivate, integrate, scale, shift


@pytest.mark.order(12)
@pytest.mark.dependency()
def test_build():
    Polynomial([0])  # p(x) = 0
    Polynomial([1])  # p(x) = 1
    Polynomial([1, 2])  # p(x) = 1 + 2 * x
    Polynomial([1, 2, 3])  # p(x) = 1 + 2 * x + 3 * x^2
    Polynomial([1.0, 2, -3.0])  # p(x) = 1.0 + 2 * x - 3.0 * x^2


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build"])
def test_degree():
    poly = Polynomial([0])  # p(x) = 0
    assert poly.degree == 0
    poly = Polynomial([1])  # p(x) = 1
    assert poly.degree == 0
    poly = Polynomial([1, 2])  # p(x) = 1 + 2 * x
    assert poly.degree == 1
    poly = Polynomial([1, 2, 3])  # p(x) = 1 + 2 * x + 3 * x^2
    assert poly.degree == 2


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build", "test_degree"])
def test_evaluate():
    poly = Polynomial([0])  # p(x) = 0
    assert poly(0) == 0
    assert poly(-1) == 0
    assert poly(2) == 0
    poly = Polynomial([1])  # p(x) = 1
    assert poly(0) == 1
    assert poly(-1) == 1
    assert poly(2) == 1
    poly = Polynomial([1, 2])  # p(x) = 1 + 2 * x
    assert poly(0) == 1
    assert poly(-1) == 1 + 2 * (-1)
    assert poly(2) == 1 + 2 * (+2)
    poly = Polynomial([1, 2, 3])  # p(x) = 1 + 2 * x + 3 * x^2
    assert poly(0) == 1
    assert poly(-1) == 1 + 2 * (-1) + 3 * (-1) * (-1)
    assert poly(2) == 1 + 2 * (+2) + 3 * (+2) * (+2)


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build", "test_degree", "test_evaluate"])
def test_neg():
    polya = Polynomial([1, 2, 3, 4])
    polyb = Polynomial([-1, -2, -3, -4])

    assert -polya == polyb


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build", "test_degree", "test_evaluate"])
def test_add():
    """
    BasisFunctions to test if the polynomials coefficients
    are correctly computed
    """
    import numpy as np

    ntests = 100
    maxdeg = 6
    tsample = np.linspace(-1, 1, 17)
    for _ in range(ntests):
        dega, degb = np.random.randint(0, maxdeg + 1, 2)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        coefsb = np.random.uniform(-1, 1, degb + 1)
        polya = Polynomial(coefsa)
        polyb = Polynomial(coefsb)
        polyc = polya + polyb
        valuesa = polya(tsample)
        valuesb = polyb(tsample)
        valuesc = polyc(tsample)

        np.testing.assert_allclose(valuesa + valuesb, valuesc)

    for _ in range(ntests):
        dega = np.random.randint(0, maxdeg + 1)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        const = np.random.uniform(-1, 1)
        polya = Polynomial(coefsa)
        polyb = polya + const
        polyc = const + polya
        valuesa = polya(tsample)
        valuesb = polyb(tsample)
        valuesc = polyc(tsample)

        np.testing.assert_allclose(valuesa + const, valuesb)
        np.testing.assert_allclose(const + valuesa, valuesc)


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build", "test_degree", "test_evaluate"])
def test_sub():
    """
    BasisFunctions to test if the polynomials coefficients
    are correctly computed
    """
    import numpy as np

    ntests = 100
    maxdeg = 6
    tsample = np.linspace(-1, 1, 17)
    for _ in range(ntests):
        dega, degb = np.random.randint(0, maxdeg + 1, 2)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        coefsb = np.random.uniform(-1, 1, degb + 1)
        polya = Polynomial(coefsa)
        polyb = Polynomial(coefsb)
        polyc = polya - polyb
        valuesa = polya(tsample)
        valuesb = polyb(tsample)
        valuesc = polyc(tsample)

        np.testing.assert_allclose(valuesa - valuesb, valuesc)

    for _ in range(ntests):
        dega = np.random.randint(0, maxdeg + 1)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        const = np.random.uniform(-1, 1)
        polya = Polynomial(coefsa)
        polyb = polya - const
        polyc = const - polya
        valuesa = polya(tsample)
        valuesb = polyb(tsample)
        valuesc = polyc(tsample)

        np.testing.assert_allclose(valuesa - const, valuesb)
        np.testing.assert_allclose(const - valuesa, valuesc)


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build", "test_degree", "test_evaluate"])
def test_mul():
    """
    BasisFunctions to test if the polynomials coefficients
    are correctly computed
    """
    import numpy as np

    ntests = 100
    maxdeg = 6
    tsample = np.linspace(-1, 1, 17)
    for _ in range(ntests):
        dega, degb = np.random.randint(0, maxdeg + 1, 2)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        coefsb = np.random.uniform(-1, 1, degb + 1)
        polya = Polynomial(coefsa)
        polyb = Polynomial(coefsb)
        polyc = polya * polyb
        valuesa = polya(tsample)
        valuesb = polyb(tsample)
        valuesc = polyc(tsample)

        np.testing.assert_allclose(valuesa * valuesb, valuesc)

    for _ in range(ntests):
        dega = np.random.randint(0, maxdeg + 1)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        const = np.random.uniform(-1, 1)
        polya = Polynomial(coefsa)
        polyb = polya * const
        polyc = const * polya
        valuesa = polya(tsample)
        valuesb = polyb(tsample)
        valuesc = polyc(tsample)

        np.testing.assert_allclose(valuesa * const, valuesb)
        np.testing.assert_allclose(const * valuesa, valuesc)


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build", "test_degree", "test_evaluate"])
def test_truediv():
    """
    BasisFunctions to test if the polynomials coefficients
    are correctly computed
    """
    import numpy as np

    ntests = 100
    maxdeg = 6
    for _ in range(ntests):
        dega = np.random.randint(0, maxdeg + 1)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        divisor = np.random.randint(1, 10)
        coefsb = [coef / divisor for coef in coefsa]
        assert Polynomial(coefsa) / divisor == Polynomial(coefsb)


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build", "test_degree", "test_evaluate"])
def test_pow():
    poly = Polynomial([-1, 1])
    assert poly**2 == Polynomial([1, -2, 1])
    assert poly**3 == Polynomial([-1, 3, -3, 1])
    assert poly**4 == Polynomial([1, -4, 6, -4, 1])


@pytest.mark.order(12)
@pytest.mark.dependency(
    depends=["test_build", "test_degree", "test_evaluate", "test_add", "test_mul"]
)
def test_derivate():
    poly = Polynomial([0])
    assert derivate(poly, 1) == 0
    assert derivate(poly, 2) == 0

    poly = Polynomial([3])
    assert derivate(poly, 1) == 0
    assert derivate(poly, 2) == 0

    poly = Polynomial([1, 1, 1, 1, 1])
    assert derivate(poly, 1) == Polynomial([1, 2, 3, 4])
    assert derivate(poly, 2) == Polynomial([2, 6, 12])
    assert derivate(poly, 3) == Polynomial([6, 24])


@pytest.mark.order(12)
@pytest.mark.dependency(
    depends=["test_build", "test_degree", "test_evaluate", "test_add", "test_mul"]
)
def test_integrate():
    poly = Polynomial([1])
    assert integrate(poly, (-3, 4)) == 7

    poly = Polynomial([1, -2])
    assert integrate(poly, (-3, 3)) == 6

    poly = Polynomial([1, 2, 5])
    assert integrate(poly, (-2, 1)) == 15


@pytest.mark.order(12)
@pytest.mark.dependency(
    depends=["test_build", "test_degree", "test_evaluate", "test_add", "test_mul"]
)
def test_shift():
    """
    BasisFunctions to test if the polynomials coefficients
    are correctly computed
    """
    import numpy as np

    ntests = 100
    maxdeg = 6
    tsample = np.linspace(-1, 1, 17)
    for _ in range(ntests):
        dega = np.random.randint(0, maxdeg + 1)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        polya = Polynomial(coefsa)
        polyb = shift(polya, 1)
        valuesa = polya(tsample)
        valuese = polyb(1 + tsample)

        np.testing.assert_allclose(valuese, valuesa)


@pytest.mark.order(12)
@pytest.mark.dependency(
    depends=["test_build", "test_degree", "test_evaluate", "test_add", "test_mul"]
)
def test_scale():
    """
    BasisFunctions to test if the polynomials coefficients
    are correctly computed
    """
    import numpy as np

    ntests = 100
    maxdeg = 6
    tsample = np.linspace(-1, 1, 17)
    for _ in range(ntests):
        dega = np.random.randint(0, maxdeg + 1)
        coefsa = np.random.uniform(-1, 1, dega + 1)
        polya = Polynomial(coefsa)
        polyb = scale(polya, 2)
        valuesa = polya(2 * tsample)
        valuesb = polyb(tsample)

        np.testing.assert_allclose(valuesb, valuesa)


@pytest.mark.order(12)
@pytest.mark.dependency(depends=["test_build"])
def test_print():
    poly = Polynomial([0])
    assert str(poly) == "0"
    poly = Polynomial([1])
    assert str(poly) == "1"
    poly = Polynomial([0, 1])
    assert str(poly) == "x"
    repr(poly)


@pytest.mark.order(12)
@pytest.mark.dependency(
    depends=[
        "test_build",
        "test_degree",
        "test_evaluate",
        "test_neg",
        "test_add",
        "test_sub",
        "test_mul",
        "test_truediv",
        "test_pow",
        "test_integrate",
        "test_shift",
        "test_scale",
    ]
)
def test_all():
    pass
