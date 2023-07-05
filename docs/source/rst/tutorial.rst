Tutorial
=================================

---------------
What is NURBS?
---------------

NURBS is a powerful tool to parametrize curves and surfaces.
It allows design smooth curves, usually applied on CAD.
It also generalizes straight lines, circles, ellipses and many other shapes.
Almost every curve can be expressed using NURBS and this library gives you tools to do it.


.. note::
    For a more formal definition, please see `wikipedia <https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline>`_ or refer to "The NURBS book" from "Les Piegl" and "Wayne Tiller".


---------------
The elements
---------------

There are three main elements for NURBS:

* Knot Vector
* Basis Functions
* Curves

A NURBS curve :math:`\mathbf{C}(u)` is expressed by a linear combination of functions :math:`f_i(u)` and control points :math:`\mathbf{P}_i`:

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{n-1} f_i(u) \cdot \mathbf{P}_{i}

There are two main caracteristics of a curve: **degree** (``degree``) and **number of points** (often abbreviated to ``npts``):

* A curve of degree :math:`1` is a set of straight segments

.. note::
    Usually the degree stays between :math:`0` and :math:`4`
