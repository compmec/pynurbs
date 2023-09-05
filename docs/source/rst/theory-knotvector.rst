===========
Knot Vector
===========

A KnotVector of degree :math:`p` and :math:`n` number of points is described by

.. math::
    \mathbf{U} = \left[u_0, \ u_{1}, \ \cdots, \ u_{n+p}\right]

There are a total of :math:`(n+p+1)` elements in this vector which satisfies

.. math::
    \underbrace{u_0 =  u_{1} = \cdots = u_{p}}_{p+1 \ \text{knots}} < u_{p+1} \le \cdots \le u_{n-1} < \underbrace{u_{n} = u_{n+1} = \cdots =  u_{n+p}}_{p+1 \ \text{knots}}

The basis curves are piecewise polynomials. They are class :math:`C^{\infty}` for each interval :math:`\left(u_{i}, \ u_{i+1}\right)`, but at :math:`u_{i}` they are only class :math:`C^{p-m_i}`. Where :math:`m_{i}` is the multiplicity of the knot :math:`u_i` inside the KnotVector.


