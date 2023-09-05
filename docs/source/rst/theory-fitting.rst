
=======
Fitting
=======


---------------
Least Square
---------------

Many implementations shown in NURBS book didn't work, like ``knot_insert``, ``knot_remove``, ``degree_increase`` and ``degree_decrease``.
Another approach was made to contour this problem, which is by using the **Least Square** method.

.. note::
    Mathematically speaking, we use the Galerkin's method, which reduces the integral of the residual square.


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

Now it depends only on the computation of the terms :math:`M` and :math:`K` and solving the linear system to find :math:`\mathbf{Q}` from :math:`\mathbf{P}`.

    
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

