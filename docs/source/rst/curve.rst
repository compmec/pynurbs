===============
Curves
===============


The main object used by this library is the **Curve**.
It allows representing **Bezier**, **BSplines** and **Rational BSplines** by the same class.
You find a simple example bellow with a curve of degree 2 and 4  


.. code-block:: python

    import numpy as np
    from matplotlib import pyplot as plt
    from pynurbs import GeneratorKnotVector, Curve
    
    # Define curve
    knotvector = GeneratorKnotVector.uniform(degree = 2, npts = 4)
    curve = Curve(knotvector)
    ctrlpoints = [(2, 4), (1, 1), (3, 2), (0, 3)]
    curve.ctrlpoints = np.array(ctrlpoints)
    
    # Plot curve
    uplot = np.linspace(0, 1, 129)
    points = curve(uplot)  # shape (129, 2)
    xplot = [point[0] for point in points]
    yplot = [point[1] for point in points]
    plt.plot(xplot, yplot, color="b")
    xvertices = [point[0] for point in ctrlpoints]
    yvertices = [point[1] for point in ctrlpoints]
    plt.plot(xvertices, yvertices, marker=".", ls="dotted", color="r")
    plt.grid()
    plt.show()

.. image:: ../img/Curve-Example-2-4.png
  :width: 70 %
  :alt: Example of parametric bspline curve of degree 2 and 4 control points
  :align: center


You can also create your own custom knotvector by passing a list of custom values.
For example, take fractional knots.

.. code-block:: python

    from fractions import Fraction
    from pynurbs import KnotVector
    zero, half, one = Fraction(0), Fraction(1, 2), Fraction(1)

    vector = [zero, zero, half, one, one]
    knotvector = KnotVector(vector)
    print(knotvector.degree)  # 1
    print(knotvector.npts)  # 3

