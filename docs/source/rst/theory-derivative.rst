

=====================
Derivative
=====================

This page contains the theory behind the application of derivatives on curves, surfaces and so on.

Derivative of curves
=====================

A curve is a linear combination of basis functions :math:`f(u)` and control points :math:`\mathbf{P}`

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{n-1} f_{i}(u) \cdot \mathbf{P}_i

The :math:`k`-derivative of a curve is

.. math::
    \mathbf{C}^{(k)}(u) = \dfrac{d^{k}\mathbf{C}}{du^{k}} = \sum_{i=0}^{n-1} \dfrac{d^{k}f_{i}(u)}{du^{k}} \cdot \mathbf{P}_i

Mainly the algorithms compute other basis functions :math:`\mathbf{g}` and new control points :math:`\mathbf{Q}` to describe this derivative

.. math::
    \mathbf{C}^{(k)}(u) = \sum_{i=0}^{m-1} g_{i}(u) \cdot \mathbf{Q}_i

Bellow we see how to find these basis functions :math:`\mathbf{g}` and the control points :math:`\mathbf{Q}`


-------------------------
Non-rational Bezier Curve
-------------------------

A non-rational bezier curve of degree :math:`p` is defined by

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{p} B_{i,p}(u) \cdot \mathbf{P}_i = \left[B_{p}\right]^{T} \cdot \mathbf{P}

Since the knotvector is

.. math::
    U = \left[\underbrace{a, \ \cdots, \ a}_{p+1}, \ \underbrace{b, \ \cdots, \ b}_{p+1} \right]

We define the basis functions as

.. math::
    B_{i,p}(u) = \binom{p}{i} \cdot \left[1-\frac{u-a}{b-a}\right]^{p-i} \cdot \left[\frac{u-a}{b-a}\right]^i

After some manipulations, the first derivative is

.. math::
    \mathbf{C}'(u) = \sum_{i=0}^{p-1} B_{i,p-1}(u) \cdot \dfrac{p}{b-a} \cdot \left(\mathbf{P}_{i+1}-\mathbf{P}_{i}\right)

Then we rewrite as

.. math::
    \mathbf{C}'(u) = \left[B_{p-1}\right]^{T} \cdot \mathbf{Q}

.. math::
    \mathbf{Q}_{i} = \dfrac{p}{b-a} \cdot \left(\mathbf{P}_{i+1}-\mathbf{P}_{i}\right)

Or in matricial form


.. math::
   \mathbf{Q} = \begin{bmatrix}\mathbf{Q}_0 \\ \mathbf{Q}_{1} \\ \vdots \\ \mathbf{Q}_{p-1} \end{bmatrix} = \left[D_{p}\right] \cdot \begin{bmatrix}\mathbf{P}_0 \\ \mathbf{P}_{1} \\ \vdots \\ \mathbf{P}_{p-1}\\ \mathbf{P}_{p}\end{bmatrix} = \left[D_{p}\right] \cdot \mathbf{P}

.. math::
    \left[D_{p}\right] = \dfrac{p}{b-a} \cdot \begin{bmatrix} -1 & 1 & 0 & \cdots & 0 & 0 \\ 0 & -1 & 1 & \cdots & 0 & 0 \\ \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\ 0 & 0 & \cdots & -1 & 1 & 0 \\ 0 & 0 & \cdots & 0 & -1 & 1 \\ \end{bmatrix}_{p \times (p+1)}

Higher derivatives can be decomposed by multiplying matrices 

.. math::
    \mathbf{C}(u) = \left[B_{p}\right]^{T} \cdot \mathbf{P}

.. math::
    \mathbf{C}'(u) = \left[B_{p-1}\right]^{T} \cdot \left[D_{p}\right] \cdot \mathbf{P}

.. math::
    \mathbf{C}''(u) = \left[B_{p-2}\right]^{T} \cdot \left[D_{p-1}\right]\left[D_{p}\right] \cdot \mathbf{P}

.. math::
    \left[D_{p-1}\right]\left[D_{p}\right] = \dfrac{p\cdot (p-1)}{(b-a)^2} \cdot \begin{bmatrix} 1 & -2 & 1 & 0 & \cdots & 0 & 0 \\ 0 & 1 & -2 & 1 & \cdots & 0 & 0 \\ \vdots & \vdots & \ddots & \ddots & \ddots & \vdots & \vdots \\ 0 & 0 & \cdots & 1 & -2 & 1 & 0 \\ 0 & 0 & \cdots & 0 & 1 & -2 & 1 \\ \end{bmatrix}_{(p-1) \times (p+1)}


Therefore

.. math::
    \mathbf{C}^{(k)}(u) = \left[B_{p-k}\right]^{T} \cdot \mathbf{Q}


.. math::
    \mathbf{Q} = \left(\prod_{i=p+1-k}^{p}\left[D_{i}\right] \right) \cdot \mathbf{P}



-------------------------
Non-rational Spline Curve
-------------------------

A non-rational spline curve is defined with the knot vector :math:`U`

.. math::
    U = \left[u_{0}, \ \cdots, u_{p}, \ u_{p+1}, \ \cdots, \ u_{n-1}, \ u_{n}, \ \cdots, \ u_{n+p} \right]

With

* :math:`u_0 = u_1 = \cdots = u_{p}`
* :math:`u_n = u_{n+1} = \cdots = u_{n+p}`

The spline basis function are

.. math::
    N_{i,0}(u) = \begin{cases}1 \ \ \ \text{if} \ u_{i} \le u < u_{i+1}\\ 0  \ \ \ \text{else}  \end{cases}

.. math::
    N_{i,j}(u) = \dfrac{u-u_{i}}{u_{i+j}-u_{i}} \cdot N_{i,j-1}(u) + \dfrac{u_{i+j+1}-u}{u_{i+j+1}-u_{i+1}} \cdot N_{i+1,j-1}(u)

The spline curve is therefore

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{n-1} N_{i,p}(u) \cdot \mathbf{P}_i = \left[N_{p}(u)\right]^{T} \cdot \mathbf{P}

Derivating may be complicated, but we can represent the same way as for bezier curve:

.. math::
    \mathbf{C}'(u) = \dfrac{d}{du}\left[N_{p}(u)\right]^{T} \cdot \mathbf{P} = \left[N_{p-1}(u)\right]^{T} \cdot \left[D_{p}\right]\cdot \mathbf{P}

.. math::
    \left[D_{j}\right] = \begin{bmatrix} -\alpha_{1,j} & \alpha_{1,j} & 0 & \cdots & 0 & 0 \\ 0 & -\alpha_{2,j} & \alpha_{2,j} & \cdots & 0 & 0 \\ \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\ 0 & 0 & \cdots & -\alpha_{n-2,j} & \alpha_{n-2,j} & 0 \\ 0 & 0 & \cdots & 0 & -\alpha_{n-1,j} & \alpha_{n-1,j} \\ \end{bmatrix}_{(n-1) \times n}

.. math::
    \alpha_{i,j} = \begin{cases}\dfrac{j}{u_{i+j}-u_{i}} \ \ \ \ \text{if}  \ \ \ \ \ u_{i+j}  \ne u_{i}  \\ 0  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{else}\end{cases}

Therefore

.. math::
    \mathbf{C}^{(k)}(u) = \left[N_{p-k}(u)\right]^{T} \cdot \mathbf{Q}

.. math::
    \mathbf{Q} = \left(\prod_{i=p+1-k}^{p}\left[D_{i}\right] \right) \cdot \mathbf{P}


-------------------------
Rational Bezier Curve
-------------------------

A Rational bezier curve of degree :math:`p` is defined by

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{p} R_{i,p}(u) \cdot \mathbf{P}_i = \left[R_{p}\right]^{T} \cdot \mathbf{P}

With knotvector and weight vector

.. math::
    U = \left[\underbrace{a, \ \cdots, \ a}_{p+1}, \ \underbrace{b, \ \cdots, \ b}_{p+1} \right]

.. math::
    w = \left[w_0, \ w_1, \ \cdots, \ w_{p} \right]

And basis function

.. math::
    B_{i,p}(u) = \binom{p}{i} \cdot \left[1-\frac{u-a}{b-a}\right]^{p-i} \cdot \left[\frac{u-a}{b-a}\right]^i
.. math::
    R_{i,p}(u) = \dfrac{w_{i}\cdot B_{i,p}(u)}{\sum_{j=0}^{p} w_{j} \cdot B_{j,p}(u)}

Computing this derivative is complicated since it involves the fraction of two functions.
For simplicity, we will use the bezier functions :math:`A_{i}(u)` and :math:`\omega(u)`

.. math::
    A_{i}(u) = w_{i}\cdot B_{i,p}(u)
.. math::
    \omega(u) = \sum_{j=0}^{p} w_{j} \cdot B_{j,p}(u)

.. math::
    R_{i,p}(u) = \dfrac{A_{i}(u)}{\omega(u)}

Derivating

.. math::
    \dfrac{d}{du} R_{i,p}(u) = \dfrac{A_{i}'(u) \cdot \omega(u) - A_i(u) \cdot \omega'(u)}{\omega^2(u)}