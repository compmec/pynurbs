.. _tutorial:
===========
Get started
===========

This tutorial is meant for these which don't feel confident about knowing NURBS.

If you feel confident, please skip to :ref:`basics`

---------------
What is NURBS?
---------------

NURBS is a powerful way to represent curves and surfaces.
It allows design smooth curves, usually applied on CAD.
It also generalizes straight lines, circles, ellipses and many other shapes.
Almost every curve can be expressed using NURBS and this library gives you tools to do it.

There is an awesome introductory video `The Continuity of Splines <https://youtu.be/jvPPXbo87ds?si=Ri02m_L6rGR0N7sS>`_ that shows the main idea behind. I recommend you to see this video and then come back.

.. note::
    For a more formal definition, please see `wikipedia <https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline>`_ or refer to "The NURBS book" from "Les Piegl" and "Wayne Tiller".


---------------
The elements
---------------

There are three main elements for NURBS:

* Knot Vector
* Basis Functions
* Curves

A NURBS curve :math:`\mathbf{C}(u)` is expressed by a linear combination of :math:`n` basis functions :math:`f_i(u)` and control points :math:`\mathbf{P}_i`:

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{n-1} f_i(u) \cdot \mathbf{P}_{i}

There are two main caracteristics of a curve: **degree** (``degree``) and **number of points** (often abbreviated to ``npts``):

* A curve of degree :math:`1` is a set of straight segments


* A curve of degree :math:`2` is a set of parabolic intervals



.. note::
    Usually the degree stays between :math:`0` and :math:`4`
