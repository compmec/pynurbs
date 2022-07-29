import pytest
from compmec.nurbs import SplineBaseFunction
from compmec.nurbs import KnotVector


def test_insert():
    U = KnotVector([0, 0, 1, 1])
    U.insert_knot(0.5)
    print(U)
    # U = KnotVector([0, 0, 0, 1, 1, 1])
    # U = KnotVector([0, 0, 0, 0, 1, 1, 1, 1])
    # U = KnotVector([0, 0, 0, 0, 0.5, 1, 1, 1, 1])

def main():
    test_insert()


if __name__ == "__main__":
    main()