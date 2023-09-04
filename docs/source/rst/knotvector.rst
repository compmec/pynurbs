Knot Vector
=============

One of the main objects used by this library is the **KnotVector**.
It's a specific type of ordoned vector which defines the parametric interval and the curve's internal behavior, like continuity and smoothness.

A Generator can be used to create these knotvectors by passing the ``degree`` and the ``npts`` (number of points). 

.. code-block:: python

    from compmec.nurbs import GeneratorKnotVector
    GeneratorKnotVector.bezier(degree=1)
    # [0, 0, 1, 1]
    GeneratorKnotVector.uniform(degree=1, npts=3)
    # [0, 0, 0.5, 1, 1]

You can also create your own custom knotvector by passing a list of custom values.
For example, take fractional knots.

.. code-block:: python

    from fractions import Fraction
    from compmec.nurbs import KnotVector
    zero, half, one = Fraction(0), Fraction(1, 2), Fraction(1)

    vector = [zero, zero, half, one, one]
    knotvector = KnotVector(vector)
    print(knotvector.degree)  # 1
    print(knotvector.npts)  # 3

-------
Methods
-------

.. autoclass:: nurbs.GeneratorKnotVector
    :members:

.. autoclass:: nurbs.KnotVector
    :members:


------
Theory
------

A KnotVector of degree :math:`p` and :math:`n` number of points is described by

.. math::
    \mathbf{U} = \left[u_0, \ u_{1}, \ \cdots, \ u_{n+p}\right]

There are a total of :math:`(n+p+1)` elements in this vector which satisfies

.. math::
    \underbrace{u_0 =  u_{1} = \cdots = u_{p}}_{p+1 \ \text{knots}} < u_{p+1} \le \cdots \le u_{n-1} < \underbrace{u_{n} = u_{n+1} = \cdots =  u_{n+p}}_{p+1 \ \text{knots}}

The basis curves are piecewise polynomials. They are class :math:`C^{\infty}` for each interval :math:`\left(u_{i}, \ u_{i+1}\right)`, but at :math:`u_{i}` they are only class :math:`C^{p-m_i}`. Where :math:`m_{i}` is the multiplicity of the knot :math:`u_i` inside the KnotVector.

