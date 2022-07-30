import pytest
import numpy as np


@pytest.mark.order(5)
@pytest.mark.dependency(
	depends=["tests/test_knotoperations.py::test_end"],
    scope='session')
def test_begin():
    pass




@pytest.mark.order(5)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_end():
    pass

def main():
    test_begin()
    test_end()

if __name__ == "__main__":
    main()
