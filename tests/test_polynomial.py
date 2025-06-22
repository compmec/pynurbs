import pytest

from pynurbs.polynomial import Polynomial, scale, shift


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_build():
    Polynomial([])  # p(x) = 0
    Polynomial([1])  # p(x) = 1
    Polynomial([1, 2])  # p(x) = 1 + 2 * x
    Polynomial([1, 2, 3])  # p(x) = 1 + 2 * x + 3 * x^2
    Polynomial([1.0, 2, -3.0])  # p(x) = 1.0 + 2 * x - 3.0 * x^2


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build"])
def test_degree():
    poly = Polynomial([])  # p(x) = 0
    assert poly.degree == 0
    poly = Polynomial([1])  # p(x) = 1
    assert poly.degree == 0
    poly = Polynomial([1, 2])  # p(x) = 1 + 2 * x
    assert poly.degree == 1
    poly = Polynomial([1, 2, 3])  # p(x) = 1 + 2 * x + 3 * x^2
    assert poly.degree == 2


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build", "test_degree"])
def test_evaluate():
    poly = Polynomial([])  # p(x) = 0
    assert poly.eval(0) == 0
    assert poly.eval(-1) == 0
    assert poly.eval(2) == 0
    poly = Polynomial([1])  # p(x) = 1
    assert poly.eval(0) == 1
    assert poly.eval(-1) == 1
    assert poly.eval(2) == 1
    poly = Polynomial([1, 2])  # p(x) = 1 + 2 * x
    assert poly.eval(0) == 1
    assert poly.eval(-1) == 1 + 2 * (-1)
    assert poly.eval(2) == 1 + 2 * (+2)
    poly = Polynomial([1, 2, 3])  # p(x) = 1 + 2 * x + 3 * x^2
    assert poly.eval(0) == 1
    assert poly.eval(-1) == 1 + 2 * (-1) + 3 * (-1) * (-1)
    assert poly.eval(2) == 1 + 2 * (+2) + 3 * (+2) * (+2)


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build", "test_degree", "test_evaluate"])
def test_add():
    """
    Function to test if the polynomials coefficients
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


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_build", "test_degree", "test_evaluate"])
def test_mul():
    """
    Function to test if the polynomials coefficients
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


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=["test_build", "test_degree", "test_evaluate", "test_add", "test_mul"]
)
def test_shift():
    """
    Function to test if the polynomials coefficients
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


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=["test_build", "test_degree", "test_evaluate", "test_add", "test_mul"]
)
def test_scale():
    """
    Function to test if the polynomials coefficients
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
