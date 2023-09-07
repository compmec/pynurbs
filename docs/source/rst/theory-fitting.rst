
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

Formulation
-----------

Let :math:`\mathbf{C}` and :math:`\mathbf{D}` be two curves defined by

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{n-1} f_{i}(u) \cdot \mathbf{P}_i

.. math::
    \mathbf{D}(u) = \sum_{i=0}^{m-1} g_{i}(u) \cdot \mathbf{Q}_i

With respectivelly knot vectors :math:`\mathbf{U}` and :math:`\mathbf{V}`

The objective is to find the values of :math:`\mathbf{Q}` such :math:`\mathbf{D}` keeps as near as possible to :math:`\mathbf{C}`.
It's made by reducing the residual square :math:`J`

.. math::
    J\left(\mathbf{Q}\right) = \dfrac{1}{2}\int_{a}^{b} \|\mathbf{C}(u)-\mathbf{D}(u)\|^2 \ du

.. math::
    J\left(\mathbf{Q}\right) = \dfrac{1}{2}\mathbf{Q}^{T} \cdot \left[GG\right] \cdot \mathbf{Q} - \mathbf{Q}^{T} \cdot \left[GF\right] \cdot \mathbf{P} + \dfrac{1}{2}\mathbf{P}^{T} \cdot \left[FF\right] \cdot \mathbf{P}

With 

.. math::
    \left[GG\right]_{ij} = \int_{a}^{b} g_{i}(u) \cdot g_{j}(u) \ du

.. math::
    \left[GF\right]_{ij} = \int_{a}^{b} g_{i}(u) \cdot f_{j}(u) \ du

.. math::
    \left[FF\right]_{ij} = \int_{a}^{b} f_{i}(u) \cdot f_{j}(u) \ du


Minimizing
----------

Since :math:`J` is convex, minimizing :math:`J` is obtained by talking its gradient and setting it to zero

.. math::
    \nabla J = \mathbf{0} \Rightarrow \left[GG\right] \cdot \mathbf{Q}^{\star} = \left[GF\right] \cdot \mathbf{P}

Now it depends only on the computation of the terms :math:`[GG]` and :math:`[GF]` and solving the linear system to find :math:`\mathbf{Q}^{\star}` from :math:`\mathbf{P}`. It's a linear transformation by using matrix :math:`T`:

.. math::
    \mathbf{Q}^{\star} = \left[T\right] \cdot \mathbf{P}

.. math::
    \left[T\right] = \left[GG\right]^{-1}\left[GF\right]

Error
-----

An important mesure is about the ``error``, which is in fact the value of :math:`J_{min} = J\left(\mathbf{Q}^{\star}\right)`:

.. math::
    J\left(\mathbf{Q}^{\star}\right) = \dfrac{1}{2}\left(\left[T\right] \cdot \mathbf{P}\right)^{T} \cdot \left[GG\right] \cdot \left(\left[T\right] \cdot \mathbf{P}\right) -\left(\left[T\right] \cdot \mathbf{P}\right)^{T} \cdot \left[GF\right] \cdot \mathbf{P} + \dfrac{1}{2}\mathbf{P} \cdot \left[FF\right] \cdot \mathbf{P}

Expanding and simplifying we get

.. math::
    J\left(\mathbf{Q}^{\star}\right) = \dfrac{1}{2} \mathbf{P}^{T} \cdot 
    [E] \cdot \mathbf{P}

.. math::
    \left[E\right] = \left[FF\right] - \left[GF\right]^{T} \cdot \left[GG\right]^{-1} \cdot \left[GF\right]

Essentially the error comes from the transcription from the set of basis functions :math:`\ \Upsilon_{U}` to the other set of basis functions :math:`\Upsilon_{V}`. If :math:`\Upsilon_{U} \subset \Upsilon_{V}`, that means, the basis functions of :math:`\mathbf{D}` describes completelly the basis functions of :math:`\mathbf{C}`, then the error is zero.

For example, if :math:`\mathbf{V}` (knotvector from :math:`\mathbf{D}`) is obtained from a ``knot_insert`` or ``degree_increase`` from :math:`\mathbf{U}` (knotvector from :math:`\mathbf{C}`), then :math:`\Upsilon_{U} \subset \Upsilon_{V}` hence :math:`J\left(\mathbf{Q}^{\star}\right) = 0`.

But if :math:`\mathbf{V}` is obtained from a ``knot_remove`` or ``degree_decrease`` from :math:`\mathbf{U}`, then :math:`\mathbf{D}` may not be equal to :math:`\mathbf{C}`, giving a non-zero error.






------------------------------
Least Square with restrictions
------------------------------

Sometimes, it's needed that the new curve (after transformation) interpolate exactly at some nodes. For example, the extremities points of a curve don't move when applying ``knot_remove`` (left image bellow) or ``degree_decrease``, while the standard least square given above would give a result like in the right figure.

|pic1|  |pic2|

.. |pic1| image:: ../img/force_removal_knot.png
   :width: 48%

.. |pic2| image:: ../img/fitting_function.png
   :width: 48%


Formulation
-----------

Let :math:`\mathbf{C}` and :math:`\mathbf{D}` be two curves defined by

.. math::
    \mathbf{C}(u) = \sum_{i=0}^{n-1} f_{i}(u) \cdot \mathbf{P}_i

.. math::
    \mathbf{D}(u) = \sum_{i=0}^{m-1} g_{i}(u) \cdot \mathbf{Q}_i

With respectivelly knot vectors :math:`\mathbf{U}` and :math:`\mathbf{V}`


The objective is to find the values of :math:`\mathbf{Q}` such :math:`\mathbf{D}` keeps as near as possible to :math:`\mathbf{C}` and satisfies the interpolation restrictions:

.. math::
    \mathbf{D}(z_{k}) = \mathbf{C}(z_{k})

For :math:`1 \le k \le m` nodes :math:`z_{i} \in \left[a, \ b\right]`

The same way, we reduce the residual square :math:`J`, but we add lagrange multipliers :math:`\lambda` related to constraint functions :math:`h(\mathbf{Q})`

.. math::
    J\left(\mathbf{Q}\right) = \dfrac{1}{2}\int_{a}^{b} \|\mathbf{C}(u)-\mathbf{D}(u)\|^2 \ du

.. math::
    \bar{J}\left(\mathbf{Q}, \lambda\right) = J\left(\mathbf{Q}\right) + \sum_{i=0}^{k-1}
    \lambda_{i} \cdot h_{i}\left(\mathbf{Q}\right)

.. math::
    \bar{J}\left(\mathbf{Q}, \lambda\right) = J\left(\mathbf{Q}\right) + \lambda^{T} \cdot \left(\left[G\right]\cdot \mathbf{Q} - \left[F\right] \cdot \mathbf{P}\right)

With 

.. math::
    \left[G\right]_{ij} = g_{j}(z_i)

.. math::
    \left[F\right]_{ij} = f_{j}(z_i)


Minimizing
----------

The same way as before, but with two variables

.. math::
    \dfrac{\partial \bar{J}}{\partial \mathbf{Q}} = \mathbf{0} \Rightarrow \left[GG\right] \cdot \mathbf{Q}^{\star} + \left[G\right]^{T} \cdot \lambda = \left[GF\right] \cdot \mathbf{P}

.. math::
    \dfrac{\partial \bar{J}}{\partial \lambda} = \mathbf{0} \Rightarrow \left[G\right] \cdot \mathbf{Q} = \left[F\right] \cdot \mathbf{P}

We mount the expanded matrix with these two equations

.. math::
    \begin{bmatrix}\left[GG\right] & \left[G\right]^{T} \\ \left[G\right] & \left[0\right] \end{bmatrix}\begin{bmatrix}\mathbf{Q}  \\ \lambda \end{bmatrix} = \begin{bmatrix}\left[GF\right]  \\ \left[F\right]  \end{bmatrix} \cdot \mathbf{P}

If it's solvable (the matrix is not singular), it has the `solution <https://mathoverflow.net/questions/365524/solve-linear-system-with-bordered-positive-definite-matrix>`_ for :math:`\mathbf{Q}`:

.. math::
    \left[LL\right] = \left[G\right] \left[GG\right]^{-1}\left[G\right]^{T}

.. math::
    \left[LG\right] = \left[LL\right]^{-1} \left[G\right] \left[GG\right]^{-1}

.. math::
    \left[QG\right] = \left[GG\right]^{-1} \left( \left[I\right] - \left[G\right]^{T}\cdot \left[LG\right]\right)

.. math::
    \left[QF\right] = \left[GG\right]^{-1} \left[G\right]^{T} \left[LL\right]^{-1} 

.. math::
    \left[T\right] = \left[QG\right] \left[GF\right] + \left[QF\right] \left[F\right]

.. math::
    \mathbf{Q} = \left[T\right] \cdot \mathbf{P}

For the error, the expression of the matrix in terms of basis matrix is too complex.
We use the computed matrix :math:`\left[T\right]` to this expression



.. warning::
    Repeted nodes makes the expanded matrix become singular (det = 0)


.. list-table:: Dimension of matrices
    :widths: 20 20 20 20
    :header-rows: 1
    :align: center

    * - Number rows
      - :math:`k`
      - :math:`m`
      - :math:`n`
    * - :math:`k`
      - :math:`LL`
      - :math:`G, LG`
      - :math:`F`
    * - :math:`m`
      - :math:`QF`
      - :math:`GG, QG`
      - :math:`T, GF`
    * - :math:`n`
      - 
      - 
      - :math:`E, FF`



------------------------------
Discrete Least Square
------------------------------

This type is used when you want to find :math:`\mathbf{D}(u)` near :math:`\mathbf{C}(u)`, but you cannot express :math:`\mathbf{C}(u)` as a linear combination of points and basis function.

That means, you want to find :math:`\mathbf{Q}` from

* :math:`m` basis functions :math:`g`
* :math:`k` nodes :math:`z_{i} \in \left[a, \ b\right]`
* :math:`k` points :math:`\mathbf{Z}_{i} = \mathbf{C}(z_i)`

.. math::
    \mathbf{D}(u) = \sum_{i=0}^{m-1} g_{i}(u) \cdot \mathbf{Q}_i

    

.. note::
    There are three cases for :math:`k`. The first one gives error

    * :math:`k < m` is a under-determined problem which has no unique solution
    * :math:`k = m` is a interpolation problem
    * :math:`k > m` is a over-determined problem which we use the least squares method

Then, define the residual function :math:`J` and get its minimum

.. math::
    J\left(\mathbf{Q}\right) = \dfrac{1}{2} \sum_{i=0}^{k-1} \|\mathbf{D}(z_i) - \mathbf{Z}_i\|^2


We derivate and set it to zero



**TO DO**