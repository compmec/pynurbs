===============
Basis Functions
===============

------
Bezier
------

Bezier curves are the most simple type of NURBS, needing only the degree :math:`p` to define the :math:`(p+1)` basis functions:

.. math::
    B_{i,p}(u) = \binom{p}{i}\left(1-u\right)^{p-i}\cdot u^{i}

At the interval :math:`u \in \left[0, \ 1\right]` and :math:`\forall i=0, \  \cdots,  \ p`

Bezier curves can be described also by Splines, which uses the following knotvectors

* Degree 1

.. math::
    \mathbf{U} = \left[0, \ 0, \ 1, \ 1\right]

* Degree 2

.. math::
    \mathbf{U} = \left[0, \ 0, \ 0, \ 1, \ 1, \ 1\right]

* Degree :math:`p`

.. math::
    \mathbf{U} = \left[\underbrace{0, \ \cdots, \ 0}_{p+1}, \ \underbrace{1, \ \cdots, \ 1}_{p+1}\right]


-----------------------------------------------------------------------


--------
B-Spline
--------

B-Splines uses the KnotVector :math:`\mathbf{U}` and are defined recursively by

.. math::
    N_{i,0}(u) = \begin{cases} 1 \ \ \ \text{if} \ u \in \left[u_i, \ u_{i+1}\right) \\ 0 \ \ \ \text{else} \end{cases}

.. math::
    N_{i,j}(u) = \dfrac{u - u_{i}}{u_{i+j}-u_{i}} \cdot N_{i,j-1}(u) + \dfrac{u_{i+j+1}-u}{u_{i+j+1}-u_{i+1}} \cdot N_{i+1,j-1}(u)

Nice properties are obtained:

* The function :math:`N_{i,j}` is local: :math:`N_{i,j}(u)=0 \ \ \forall u \notin \left[u_{i}, \ u_{i+j+1}\right)`
* At the interval :math:`\left[u_{i}, \ u_{i+j+1}\right)`, only the :math:`(j+1)` functions are non-zero: :math:`N_{i-j,j}`, :math:`\cdots`, :math:`N_{i-1,j}` and :math:`N_{i,j}`
* The functions are non-negative: :math:`N_{i,j}(u)\ge 0 \ \ \forall u`
* The sum of these functions are always equal to :math:`1`: :math:`\sum_{i=0}^{n-1}N_{i,j}(u) = 1 \ \ \forall u`
* All derivatives of :math:`N_{i,j}(u)` exist in :math:`\left(u_{k}, \ u_{k+1}\right)`. At the knot :math:`u_k`, it's :math:`p-m` continuously differentiable, where :math:`m` is the multiplicity of the knot
* Except for :math:`j=0`, `N_{i,j}` reaches only one maximum.


-----------------------------------------------------------------------

-----------------
Rational B-Spline
-----------------


Rational B-Splines also uses the KnotVector :math:`\mathbf{U}` along a weight vector :math:`\mathbf{w}`.
It's defined by

.. math::
    R_{i,j}(u) = \dfrac{w_{i} \cdot N_{i,j}(u)}{\sum_{k=0}^{n-1} w_{k} \cdot N_{k,j}(u)}

.. math::
    \mathbf{w} = \left[w_0, \ w_1, \ \cdots, \ w_{n-1}\right]