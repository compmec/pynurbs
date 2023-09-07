

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