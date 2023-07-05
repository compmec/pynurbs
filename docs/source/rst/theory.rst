Theory
=============

---------------
Least Square
---------------

Many implementations shown in NURBS book didn't work, like ``knot_insert``, ``knot_remove``, ``degree_increase`` and ``degree_decrease``.
Another approach was made to contour this problem, which is by using the **LeastSquare** method.
This method can be also understood as Galerkin's method, which reduces the integral of the residual square.

Let :math:`\mathbf{C}` and :math:`\mathbf{D}` be two curves defined by

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{n-1} f_{i}(u) \cdot \mathbf{P}_i

.. math::
    \mathbf{D}(u) = \sum_{i=0}^{m-1} g_{i}(u) \cdot \mathbf{Q}_i

The objective is to find the values of :math:`\mathbf{Q}` such :math:`\mathbf{D}` keeps as near as possible to :math:`\mathbf{C}`.
It's done by reducing the residual square :math:`J`

.. math::
    J\left(\mathbf{Q}\right) = \int_{a}^{b} \|\mathbf{C}(u)-\mathbf{D}(u)\|^2 \ du

Using the projection method, it's possible to get :math:`m` equations

.. math::
    \int_{a}^{b} \left[ \mathbf{C}(u)-\mathbf{D}(u)\right] \cdot g_{i}(u) \ du = 0

.. math::
    \forall i = 0, \ \cdots, \ m-1

Expand to

.. math::
    \left[M\right] \cdot \mathbf{Q} = \left[L\right]\cdot \mathbf{P}

.. math::
    \left[M\right]_{ij} = \int_{a}^{b} g_{i}(u) \cdot g_{j}(u) \ du
.. math::
    \left[L\right]_{ij} = \int_{a}^{b} g_{i}(u) \cdot f_{j}(u) \ du

Now it depends only on the computation of the terms :math:`M` and :math:`K` to find :math:`\mathbf{Q}` from :math:`\mathbf{P}`.

    
.. math::
    \mathbf{Q}^{\star} = \underbrace{\left[M\right]^{-1}\left[L\right]}_{\left[T\right]}\cdot \mathbf{P}

Another important value is about the error:

    If :math:`\mathbf{V}` is obtained from a ``knot_insert`` or ``degree_increase`` from :math:`\mathbf{U}`, then :math:`\mathbf{D}` is equal to :math:`\mathbf{C}`, giving a zero error.
    But if :math:`\mathbf{V}` is obtained from a ``knot_remove`` or ``degree_decrease`` from :math:`\mathbf{U}`, then :math:`\mathbf{D}` may not be equal to :math:`\mathbf{C}`, giving a non-zero error.

    We use :math:`J` to evaluate this error:

    .. math::
        J_{min} = J(\mathbf{Q}^{\star}) = \mathbf{P}^{T} \cdot \left[E\right] \cdot \mathbf{P}

    .. math::
        \left[E\right] = \left[K\right] - \left[L\right]^{T}\left[M\right]^{-1}\left[L\right]

    .. math::
        \left[K\right]_{ij} = \int_{a}^{b} f_{i}(u) \cdot f_{j}(u) \ du


.. note::
    There are :math:`n` control points :math:`\mathbf{P}` and :math:`m` control points :math:`\mathbf{Q}`

    * :math:`K` is a :math:`(n \times n)` matrix
    * :math:`L` is a :math:`(m \times n)` matrix
    * :math:`M` is a :math:`(m \times m)` matrix
    * :math:`T` is a :math:`(m \times n)` matrix
    * :math:`E` is a :math:`(n \times n)` matrix


----------------------------------------------------------------

---------------------
Integration by bezier
---------------------

Numerical integration is used by **LeastSquare** method.
We developed a method to compute the integral :math:`I`:

.. math::
    I = \int_{a}^{b} f(x) \ dx

The first step is divide the interval in many smalls :math:`I_i`

.. math::
    I = \int_{a}^{b} f(x) \ dx = \sum_{i} \int_{x_i}^{x_{i+1}} f(x) dx = \sum_{i} I_{i}

Then we interpolate :math:`f(x)` into :math:`g(x)` at the interval :math:`\left[u_{i}, \ u_{i+1}\right]`, a linear combination of weights :math:`g_j` and bezier functions of degree :math:`z`

.. math::
    f(x) \approx g(x) = \sum_{j=0}^{z} g_{j} \cdot B_{j,z}\left(\dfrac{x-x_{i}}{x_{i+1}-x_{i}}\right)

.. math::
    B_{j,z}(u) = \binom{z}{j} (1-u)^{z-j} \cdot u^j

Integrate the interpolation functions to get

.. math::
    I_{i} = \int_{x_i}^{x_{i+1}} g(x) \ dx = \sum_{j=0}^{z} \int_{0}^{1} g_{j} \cdot B_{j,z}(u) \ du = \dfrac{1}{z+1} \sum_{j=0}^{z} g_{j}

To find coefficients :math:`g_{j}` for each interval :math:`\left[x_i, \ x_{i+1}\right]`, one can say :math:`g(x)` matches :math:`f(x)` in distribued nodes inside the interval, obtain a linear system and solve

.. math::
    u_{k} = \dfrac{y_k - x_{i}}{x_{i+1}-x_{i}} \Leftrightarrow y_{k} = (1-u_k) \cdot x_{i} + u_{k} \cdot x_{i+1}

.. math::
    \underbrace{\begin{bmatrix}B_{0,z}(u_0) & B_{1,z}(u_0) & \cdots & B_{z,z}(u_0) \\ B_{0,z}(u_1) & B_{1,z}(u_1) & \cdots & B_{z,z}(u_1) \\ \vdots & \vdots & \ddots & \vdots \\ B_{0,z}(u_z) & B_{1,z}(u_z) & \cdots & B_{z,z}(u_z) \end{bmatrix}}_{\left[A\right]}\begin{bmatrix}g_{0} \\ g_{1} \\ \vdots \\ g_{z}\end{bmatrix} = \begin{bmatrix}f\left(y_0\right) \\ f\left(y_1\right) \\ \vdots \\ f\left(y_z\right)\end{bmatrix}

Assuming the nodes :math:`u_k` don't change, the matrix :math:`\left[A\right]` is constant and can be inverted only once.

Also, since we only need the value of :math:`\sum g_{j}`, we can get an *integrator vector* :math:`V`:

.. math::
    V = \dfrac{1}{z+1} \cdot \left[1, \ 1, \ \cdots, \ 1\right] \cdot \left[A\right]^{-1} 

.. math::
    I_{i} = V \cdot \begin{bmatrix}f\left((1-u_0) \cdot x_{i} + u_{0} \cdot x_{i+1} \right) \\ f\left((1-u_1) \cdot x_{i} + u_{1} \cdot x_{i+1} \right) \\ \vdots \\  f\left((1-u_z) \cdot x_{i} + u_{z} \cdot x_{i+1} \right) \end{bmatrix}

It's possible to use **equally distribued nodes** inside the interval :math:`[0, \ 1]` by setting :math:`u_k = k/z`, but it's preferable to use `Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`_:

.. math::
    u_k = \sin^2 \left(\dfrac{2k+1}{4(z+1)} \cdot \pi\right)
