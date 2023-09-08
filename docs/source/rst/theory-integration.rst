

=====================
Numerical Integration
=====================

Numerical integration as the name says evaluates numerically the integral :math:`I`:

.. math::
    I = \int_{a}^{b} f(u) \ du

Many methods were developed and here we develop one specific for our case.

Divide the domain
=================

The numerical integration divides the whole domain in many parts to avoid the `Runge's phenomenon <https://en.wikipedia.org/wiki/Runge%27s_phenomenon>`_ and not to use very high degree algorithms.

Since our base functions are already divided in :math:`m` intervals, based on the knotvector, the integral :math:`I` is divided in the sum of many :math:`I_{j}`, for each interval :math:`\left[\bar{u}_{j}, \ \bar{u}_{j+1}\right]`

.. math::
    I = \sum_{j=0}^{m-1} I_{j}

.. math::
    I_{j} = \int_{\bar{u}_j}^{\bar{u}_{j+1}} f(u) \ du

For each interval :math:`\left[\bar{u}_{j}, \ \bar{u}_{j+1}\right]`, the function :math:`f(u)` is either a bezier or a rational bezier:

1. Non-rational

.. math::
    f(u) = \sum_{i=0}^{p} \alpha_{i} \cdot u^{i}

2. Rational

.. math::
    f(u) = \dfrac{\sum_{i=0}^{p} \alpha_{i} \cdot u^{i}}{\sum_{i=0}^{p} \beta_{i} \cdot u^{i}}

Although the polynomials are not described in canonical base, the values :math:`\alpha` and :math:`\beta` are not known à priori, we use methods to integrate polynomials.

Integrator array
================

The integral over :math:`\left[\bar{u}_j, \ \bar{u}_{j+1}\right]` can be transformed to an equivalent integral on :math:`\left[-1, \ 1\right]` by variable transformation. So we present the method for integrating in :math:`\left[-1, \ 1\right]`

Almost all methods are the same: transforms the integral as a sum of :math:`q` weights :math:`w_k` multiplied by function evaluation at specific nodes :math:`v_k`

.. math::
    \int_{\bar{u}_j}^{\bar{u}_{j+1}} f(u) \ du  = \sum_{k = 0}^{q-1} w_k \cdot f(v_k)

By defining the nodes :math:`v_k` and weights :math:`w_k` you have different methods as Newton-cotes and gauss quadrature.

We call the **Integrator array** instead of weights because it can confuse with the weights of a rational curve.


-------------------
Closed Newton cotes
-------------------

The closed newton cotes formula states that :math:`v_k` are equally spaced points including the boundary :math:`\left[-1, 1\right]`

.. math::
    v_k = -1 + \dfrac{2k}{q-1}

.. math::
    \int_{\bar{u}_j}^{\bar{u}_{j+1}} f(u) \ du  = \sum_{k = 0}^{q-1} w_k \cdot f(v_k)

.. list-table:: Nodes and weights for closed-newton-cotes in :math:`\left[-1, 1\right]`
    :widths: 20 20 20
    :header-rows: 1
    :align: center

    * - Number :math:`q`
      - Nodes :math:`v_k`
      - Weights :math:`w_k`
    * - :math:`2`
      - :math:`-1`, :math:`1`
      - :math:`1`, :math:`1`
    * - :math:`3`
      - :math:`-1`,   :math:`0`,   :math:`1`
      - :math:`\dfrac{1}{3}`,   :math:`\dfrac{4}{3}`,   :math:`\dfrac{1}{3}`
    * - :math:`4`
      - :math:`-1`,   :math:`\dfrac{-1}{3}`,   :math:`\dfrac{1}{3}`, :math:`1`
      - :math:`\dfrac{1}{4}`,   :math:`\dfrac{3}{4}`,   :math:`\dfrac{3}{4}`,   :math:`\dfrac{1}{4}`

The code bellow is an algorithm that uses ``sympy`` to compute the **integrator array** (called as ``weights``) from given nodes. That code can be used for other formulas like **Open Newton Cotes**.

.. code-block:: python

    import sympy as sp

    def find_weights(nodes, a, b):
        x = sp.symbols("x", real=True)
        nnodes = len(nodes)
        coefs = [sp.symbols("c%d"%i) for i in range(nnodes)]
        poly = sum([ci*(x**i) for i, ci in enumerate(coefs)])
        good_integ = sp.integrate(poly, (x, a, b))
        weights = [sp.symbols("w%d"%i) for i in range(nnodes)]
        funcvals = [sum([ci*(xj**i) for i, ci in enumerate(coefs)]) for xj in nodes]
        test_integ = sum([wj*fj for wj, fj in zip(weights, funcvals)])
        error = test_integ - good_integ
        equations = [sp.diff(error, coef) for coef in coefs]
        equations = [sp.simplify(equation) for equation in equations]
        solution = sp.solve(equations, weights)
        weights = [solution[wi] for wi in weights]
        return weights

    # Closed newton cotes
    a, b = -1, 1
    print("Closed newton cotes")
    for q in range(2, 5):
        print(f"For q = {q}")
        h = (b-a)*sp.Rational(1, q-1)
        nodes = [a+i*h for i in range(q)]
        weights = find_weights(nodes, a, b)
        print(f"    nodes = {nodes}")
        print(f"    weigs = {weights}")

-------------------------
Gauss-Legendre quadrature
-------------------------

The Gauss quadrature is widely used and has the property such can compute exactly the integral of a polynomial of degree :math:`2q-1` with only :math:`q` evaluation points at specific nodes :math:`v_{k}`.

.. math::
    I_j = \int_{\bar{u}_{j}}^{\bar{u}_{j+1}} \mathbf{C}(u) \ du = \sum_{i=0}^{q-1} w_{i} \cdot \mathbf{C}(v_k)

Since we work with polynomials, interpolate exactly the polynomial of degree :math:`p` would require :math:`q = \text{ceil}\left(\dfrac{p+1}{2}\right)` evaluation nodes instead of :math:`q = p` as in Newton-cotes.

.. note::
    As it's possible to work with ``fraction`` module and the nodes :math:`v_k` may be irrational, converting :math:`v_k` to fraction can introduce very large values on the denominator.

    Since the integral of a polynomial of rational coefficients is a rational value, using Newton-cotes with :math:`q = p` is preferable

The nodes and weights are in the table bellow and can be obtained by ``numpy`` (docs `here <https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html>`_).

.. list-table:: Nodes and weights for gauss-legendre-quadrature in :math:`\left[-1, 1\right]`
    :widths: 20 20 20
    :header-rows: 1
    :align: center

    * - Number :math:`q`
      - Nodes :math:`v_k`
      - Weights :math:`w_k`
    * - :math:`1`
      - :math:`0`
      - :math:`2`
    * - :math:`2`
      - :math:`-\dfrac{1}{\sqrt{3}}`, :math:`\dfrac{1}{\sqrt{3}}`
      - :math:`1`, :math:`1`
    * - :math:`3`
      - :math:`-\sqrt{\dfrac{3}{5}}`,   :math:`0`,   :math:`\sqrt{\dfrac{3}{5}}`
      - :math:`\dfrac{5}{9}`,   :math:`\dfrac{8}{9}`,   :math:`\dfrac{5}{9}`

The python code bellow finds the **nodes** and the **integrator array** (``weights``) 

.. code-block:: python

    import numpy as np

    for q in range(1, 4):
        nodes, weights = np.polynomial.legendre.leggauss(q)
        print("  nodes = ", nodes)
        print("weights = ", weights)


Integral over a curve
===================================

A curve is defined by

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{n-1} f_{i}(u) \cdot \mathbf{P}

The first notion of integral is the same as the scalar integral

.. math::
    \int_{a}^{b} \mathbf{C}(u) \ du = \sum_{i=0}^{n-1} \left(\int_{a}^{b} f_{i}(u) \ du\right) \cdot \mathbf{P}

But since we allow custom objects, it's interesting to have operation such:


* The scalar product with a function :math:`g(u)`

    .. math::
        \int_{a}^{b} g(u) \cdot \mathbf{C}(u) \ du


* The scalar product with a function :math:`g(u)` which depends on coordinates

    .. math::
        \int_{a}^{b} g(u) \cdot \mathbf{C}(u) \ du

    .. math::
        g(u) = x(u)^a \cdot y(u)^b \cdot z(u)^{c}

* The 'lenght' of a curve multiplied by a weight function

    .. math::
        \int_{a}^{b} g(u) \|\mathbf{C}(u)\| \ du

* If :math:`\mathbf{P}` are :math:`n`-dimentional points, the inner product

    .. math::
        \int_{a}^{b} \langle g(u), \mathbf{C}(u) \rangle \ du


Integral of non-rational bezier
===============================

For this case we will consider

.. math::
    f(u) = \mathbf{C}(u)

Bezier curves are described as 

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{p} B_{i,p}(u) \cdot \mathbf{P}_i

The integral is therefore

.. math::
    \int_{\bar{u}_{j}}^{\bar{u}_{j+1}} \mathbf{C}(u) \ du = \sum_{i=0}^{p} \left(\int_{\bar{u}_{j}}^{\bar{u}_{j+1}} B_{i,p}(u) \ du\right) \cdot \mathbf{P}_i

So, the objective is integrate the basis function :math:`B_{i,p}`

.. math::
    I_{j} = \int_{\bar{u}_{j}}^{\bar{u}_{j+1}} B_{i,p}(u) \ du = \int_{\bar{u}_{j}}^{\bar{u}_{j+1}} \binom{p}{i} \left[1 - \dfrac{u-\bar{u}_{j}}{\bar{u}_{j+1}-\bar{u}_{j}}\right]^{p-i} \cdot \left[\dfrac{u-\bar{u}_{j}}{\bar{u}_{j+1}-\bar{u}_{j}}\right]^{i} \ du

.. math::
    I_{j} = (\bar{u}_{j+1}-\bar{u}_{j})\binom{p}{i}\int_{0}^{1}  \left(1 - t\right)^{p-i} \cdot t^{i} \ dt

This integral is well known as the `Beta function <https://en.wikipedia.org/wiki/Beta_function>`_:

.. math::
    \int_{0}^{1} \left(1 - t\right)^{p-i} \cdot t^{i} \ dt = \dfrac{1}{p+1} \cdot \dfrac{1}{\binom{p}{i}}

Therefore, the integral on :math:`\left[\bar{u}_{j}, \ \bar{u}_{j+1}\right]` is

.. math::
    \int_{\bar{u}_{j}}^{\bar{u}_{j+1}} B_{i,p}(u) \ du = \dfrac{\bar{u}_{j+1}-\bar{u}_{j}}{p+1}


.. math::
    \int_{\bar{u}_{j}}^{\bar{u}_{j+1}} \mathbf{C}(u) \ du = \dfrac{\bar{u}_{j+1}-\bar{u}_{j}}{p+1} \cdot \sum_{i=0}^{p} \mathbf{P}_i

Bezier interpolation
====================

The formula for the integral works fine when the curve is already a bezier one. But for splines, the control points are not the same as the local bezier and therefore it's necessary a bezier interpolation.

.. note::
    In fact, it's possible to develop a method without the interpolation by extracting the bezier segment from the spline. If the algorithms of knot insertion are fast enough, decomposing in bezier segments can be also be used, mainly when the integration is called many times

Since :math:`f(u)` is a polynomial of degree :math:`p` in the interval :math:`\left[\bar{u}_{j}, \ \bar{u}_{j+1}\right]`, then a bezier curve of degree must be also degree :math:`p`.

.. note::
    If the degree of interpolation is :math:`q < p`, then selecting different positions for nodes :math:`v_k` can lead to reduce the error (compared with equally distributed nodes). As example, one can choose `Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`_ 





