

=====================
Numerical Integration
=====================

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
