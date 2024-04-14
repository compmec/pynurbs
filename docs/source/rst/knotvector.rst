Knot Vector
=============

One of the main objects used by this library is the **KnotVector**.

It's a specific type of ordoned vector which defines the parametric interval and the curve's internal behavior, like continuity and smoothness.

A Generator can be used to create these knotvectors by passing the ``degree`` and the ``npts`` (number of points). 

.. code-block:: python

    from pynurbs import GeneratorKnotVector

    GeneratorKnotVector.bezier(degree=1)
    # [0, 0, 1, 1]
    GeneratorKnotVector.uniform(degree=1, npts=3)
    # [0, 0, 0.5, 1, 1]

Construct a Knot Vector
-----------------------

A **knotvector** is a vector of numbers. They can be integers, floats, fractions or a custom number (like ``mpmath.mpf``)


Custom Knot Vector
------------------

You can also create your own custom knotvector by passing a list of custom values.
For example, take fractional knots.

.. code-block:: python

    from fractions import Fraction
    from pynurbs import KnotVector
    
    zero, half, one = Fraction(0), Fraction(1, 2), Fraction(1)
    vector = [zero, zero, half, one, one]
    knotvector = KnotVector(vector)
    print(knotvector.degree)  # 1
    print(knotvector.npts)  # 3

Another way is to use the ``GeneratorKnotVector`` with the specific type you want

.. code-block:: python

    from fractions import Fraction
    from pynurbs import GeneratorKnotVector
    
    knotvector = GeneratorKnotVector.uniform(degree = 1, npts = 3, cls = Fraction)
    print(knotvector.degree)  # 1
    print(knotvector.npts)  # 3
    print(knotvector)  # (0, 0, 1/2, 1, 1)