Knot Vector
=============

To compute the basis functions, it's used a specific type of ordoned vector, which we call **KnotVector**.

This KnotVector defines not only the parametric interval extremities, but also the internal knots, which are responsible for the behavior of the curve, like continuity and smoothness.

The construction of a KnotVector may be complicated.
A Generator can be used to create these knotvectors by setting the ``degree`` and the ``npts`` (number of points). 

.. code-block:: python

    GeneratorKnotVector.bezier(degree=1)
    # [0, 0, 1, 1]
    GeneratorKnotVector.bezier(degree=2)
    # [0, 0, 0, 1, 1, 1]
    GeneratorKnotVector.uniform(degree=1, npts=3)
    # [0, 0, 0.5, 1, 1]
    GeneratorKnotVector.uniform(degree=2, npts=6)
    # [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1]
    GeneratorKnotVector.random(degree=2, npts=6)
    # [0, 0, 0, 0.21, 0.57, 0.61, 1, 1, 1]
    GeneratorKnotVector.weight(degree=1, [1, 2, 1])
    # [0, 0, 1, 3, 4, 4]



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

