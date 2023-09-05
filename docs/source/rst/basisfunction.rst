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

Bezier curves can be described also by Splines, which uses the knotvector :math:`\textbf{U}`

* Degree 1

.. math::
    \mathbf{U} = \left[0, \ 0, \ 1, \ 1\right]

* Degree :math:`p`

.. math::
    \mathbf{U} = \left[\underbrace{0, \ \cdots, \ 0}_{p+1}, \ \underbrace{1, \ \cdots, \ 1}_{p+1}\right]

Use example
-----------

.. code-block:: python

    import numpy as np
    from matplotlib import pyplot as plt
    from compmec.nurbs import GeneratorKnotVector, Function

    degree = 2
    knotvector = GeneratorKnotVector.bezier(degree)
    bezier = Function(knotvector)

    uplot = np.linspace(0, 1, 129)
    for i in range(degree + 1):
        label = r"$B_{%d,%d}$" % (i, degree)
        plt.plot(uplot, bezier[i](uplot), label=label)
    plt.legend()
    plt.show()

.. image:: ../img/Bezier-Basis-Functions-2.png
  :width: 70 %
  :alt: Bezier Basis Functions of degree 2
  :align: center


-----------------------------------------------------------------------


--------
B-Spline
--------

B-Splines uses the knotvector :math:`\mathbf{U}` and is recursevelly defined by

.. math::
    N_{i,0}(u) = \begin{cases}1 \ \ \ \text{if} \  u_{i}  \le u < u_{i+1}  \\ 0 \ \ \ \text{else} \end{cases}

.. math::
    N_{i,j}(u) = \dfrac{u - u_i}{u_{i+j}-u_{i}} \cdot N_{i,j-1}(u) + \dfrac{u_{i+j+1}-u}{u_{i+j+1}-u_{i+1}} \cdot N_{i+1,j-1}(u)
    

Use example
-----------

.. code-block:: python

    import numpy as np
    from matplotlib import pyplot as plt
    from compmec.nurbs import GeneratorKnotVector, Function
    
    degree, npts = 2, 5
    knotvector = GeneratorKnotVector.uniform(degree, npts)
    spline = Function(knotvector)
    
    uplot = np.linspace(0, 1, 129)
    for i in range(npts):
        label = r"$N_{%d,%d}$" % (i, degree)
        plt.plot(uplot, spline[i](uplot), label=label)
    plt.legend()
    plt.show()

.. image:: ../img/Spline-BasisFunctions-2-5.png
  :width: 70 %
  :alt: BSpline Basis Functions of degree 2 and npts 5
  :align: center


-----------------------------------------------------------------------

-----------------
Rational B-Spline
-----------------

Like B-Splines, Rational B-Splines also uses the knotvector :math:`\mathbf{U}`, but along a weight vector :math:`\mathbf{w}`.
It's defined by

.. math::
    R_{i,j}(u) = \dfrac{w_{i} \cdot N_{i,j}(u)}{\sum_{k=0}^{n-1} w_{k} \cdot N_{k,j}(u)}

.. math::
    \mathbf{w} = \left[w_0, \ w_1, \ \cdots, \ w_{n-1}\right]

Use example
-----------

.. code-block:: python

    import numpy as np
    from matplotlib import pyplot as plt
    from compmec.nurbs import GeneratorKnotVector, Function
    
    degree, npts = 2, 5
    knotvector = GeneratorKnotVector.uniform(degree, npts)
    rational = Function(knotvector)
    rational.weights = [1, 2, 0.5, 5, 2]

    uplot = np.linspace(0, 1, 129)
    for i in range(npts):
        label = r"$R_{%d,%d}$" % (i, degree)
        plt.plot(uplot, rational[i](uplot), label=label)
    plt.legend()
    plt.show()

.. image:: ../img/Rational-BasisFunctions-2-5.png
  :width: 70 %
  :alt: Rational BSpline Basis Functions of degree 2 and 5 npts
  :align: center


