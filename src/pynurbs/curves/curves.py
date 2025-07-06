from __future__ import annotations

from fractions import Fraction
from numbers import Real
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from ..core.custom_math import isscalar, number_type
from ..knotspace import KnotVector
from ..operations import heavy
from ..operations.knotvector import insert_knots, remove_knots
from ..operations.least_square import fit_function, func2func, spline2spline
from ..operations.tools import vectorize
from .base import BaseCurve


class Curve(BaseCurve):
    def __init__(
        self,
        knotvector: KnotVector,
        ctrlpoints: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(knotvector)
        self.ctrlpoints = ctrlpoints
        self.weights = weights

    def __str__(self) -> str:
        if self.npts == self.degree + 1:
            msg = "Bezier"
        elif self.weights is None:
            msg = "Spline"
        else:
            msg = "Rational Spline"
        msg += f" curve of degree {self.degree}"
        msg += f" and {self.npts} control points\n"
        msg += f"KnotVector = {self.knotvector}\n"
        if self.ctrlpoints is None:
            return msg
        msg += "ControlPoints = ["
        msg += ", ".join([str(point) for point in self.ctrlpoints])
        msg += "]\n"
        return msg

    @vectorize(1, 0)
    def eval(self, node: Real) -> Any:
        """Point evaluation function

        :param nodes: The nodes to evaluates
        :type nodes: float | tuple[float]
        :raises TypeError: If ``nodes`` is not a number or a list of numbers
        :raises ValueError: If at least node is outside ``[umin, umax]``
        :return: The point computed by using control points
        :rtype: Any | tuple[Any]

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0.5, 1, 1])
        >>> curve.ctrlpoints = (1, 2, -3)
        >>> curve(0)
        1.0
        >>> curve(0.2)
        1.4
        >>> curve(0.5)]
        2.0
        >>> curve([0, 0.5, 1])
        (1.0, 2.0, -3.0)

        """
        return self(node)

    def knot_insert(self, nodes: Tuple[float]) -> None:
        """Insert given nodes inside knotvector

        :param nodes: The nodes to be inserted
        :type nodes: tuple[float]
        :raises TypeError: If ``nodes`` is not a number or a list of numbers
        :raises ValueError: If it's not possible to insert the knots

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0.5, 1, 1])
        >>> curve.ctrlpoints = (1, 2, -3)
        >>> curve.knot_insert([0.2, 0.7])
        >>> curve.knotvector
        (0, 0, 0.2, 0.5, 0.7, 1, 1)

        """
        nodes = tuple(nodes)
        oldvector = self.knotvector.internal
        newvector = insert_knots(oldvector, nodes)
        if self.ctrlpoints is None and self.weights is None:
            self.knotvector = newvector
        matrix = heavy.Operations.knot_insert(oldvector, nodes)
        self.apply(newvector, matrix)

    def knot_remove(
        self, nodes: Tuple[float], tolerance: float = 1e-9
    ) -> None:
        """Remove given nodes from knotvector

        :param nodes: The nodes to be removed
        :type nodes: tuple[float]
        :param tolerance: Tolerance to remove knots, defaults to ``1``
        :type tolerance: float(, optional)
        :param nodes: Nodes to be assure to, defaults to ``None``
        :type nodes: tuple[float](, optional)
        :raises TypeError: If ``nodes`` is not a number or a list of numbers
        :raises ValueError: If it's not possible to remove the knot

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0, 0.5, 1, 1, 1])
        >>> curve.ctrlpoints = [1, 1.5, -0.5, -3]
        >>> print(curve)
        Spline curve of degree 2 and 4 control points
        KnotVector = (0, 0, 0, 0.5, 1, 1, 1)
        ControlPoints = [1, 1.5, -0.5, -3]
        >>> curve.knot_remove([0.5])
        >>> print(curve)
        Bezier curve of degree 2 and 3 control points
        KnotVector = (0, 0, 0, 1, 1, 1)
        ControlPoints = [1.0, 2.0, -3.0]

        """
        old_tolerance = self.tolerance
        self.tolerance = tolerance
        self.knotvector = remove_knots(self.knotvector.internal, nodes)
        self.tolerance = old_tolerance

    def knot_clean(
        self,
        nodes: Optional[Tuple[float]] = None,
        tolerance: Optional[float] = 1e-9,
    ) -> None:
        """Remove all unnecessary knots.

        If no nodes are given, it tries to remove all internal knots

        Nothing happens if the curve is irreductible

        Nodes equals to extremities are ignored

        Nodes which are not in knotvectors are ignored

        :param nodes: The nodes to be removed, defaults to ``None``, all internal knots
        :type nodes: tuple[float](, optional)
        :param tolerance: The tolerance to remove knots, defaults to ``1e-9``
        :type tolerance: float(, optional)
        :raises TypeError: If ``nodes`` is not a number or a list of numbers
        :raises TypeError: If ``tolerance`` is not a number
        :raises ValueError: If ``tolerance`` is negative

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0, 0.5, 1, 1, 1])
        >>> curve.ctrlpoints = [1, 1.5, -0.5, -3]
        >>> print(curve)
        Spline curve of degree 2 and 4 control points
        KnotVector = (0, 0, 0, 0.5, 1, 1, 1)
        ControlPoints = [1, 1.5, -0.5, -3]
        >>> curve.knot_clean()
        >>> print(curve)
        Bezier curve of degree 2 and 3 control points
        KnotVector = (0, 0, 0, 1, 1, 1)
        ControlPoints = [1.0, 2.0, -3.0]

        """
        if not isscalar(tolerance) or tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        if nodes is None:
            nodes = self.knotvector.knots
        nodes = tuple(set(nodes) - set(self.knotvector.limits))
        oldtolerance = self.tolerance
        self.tolerance = tolerance
        for knot in nodes:
            try:
                while True:
                    self.knot_remove([knot])
            except ValueError:
                pass
        self.tolerance = oldtolerance

    def degree_increase(self, times: Optional[int] = 1):
        """Increase the degree of the curve by an amount ``times``

        :param times: The number of times to increase, defaults to ``1``
        :type times: int(, optional)
        :raises AssertionError: If ``times`` is not a integer >= 0

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0, 0.5, 1, 1, 1])
        >>> curve.ctrlpoints = [1, 1.5, -0.5, -3]
        >>> print(curve)
        Spline curve of degree 2 and 4 control points
        KnotVector = (0, 0, 0, 0.5, 1, 1, 1)
        ControlPoints = [1, 1.5, -0.5, -3]
        >>> curve.degree_increase(1)
        >>> print(curve)
        Spline curve of degree 3 and 6 control points
        KnotVector = (0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1)
        ControlPoints = [1.0, 1.33, 1.17, -0.17, -1.33, -3.0]
        """
        if not isinstance(times, int) or times < 0:
            raise ValueError
        self.degree += int(times)

    def degree_decrease(
        self, times: Optional[int] = 1, tolerance: Optional[float] = 1e-9
    ):
        """Decrease the degree of the curve by an amount ``times``

        :param times: The number of times to reduce degree, defaults to ``1``
        :type times: int(, optional)
        :param tolerance: Tolerance to remove knots, defaults to ``1``
        :type tolerance: float(, optional)
        :raises AssertionError: If ``times`` is not a integer >= 0

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1])
        >>> curve.ctrlpoints = [1, 4/3, 7/6, -1/6, -4/3, -3]
        >>> print(curve)
        Spline curve of degree 3 and 6 control points
        KnotVector = (0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1)
        ControlPoints = [1.0, 1.33, 1.17, -0.17, -1.33, -3.0]
        >>> curve.degree_decrease(1)
        >>> print(curve)
        Spline curve of degree 2 and 4 control points
        KnotVector = (0, 0, 0, 0.5, 1, 1, 1)
        ControlPoints = [1, 1.5, -0.5, -3]
        """
        if not isinstance(times, int) or times < 0:
            raise ValueError(f"times = {times}")
        if tolerance is not None:
            if not isscalar(tolerance) or tolerance <= 0:
                raise ValueError("Tolerance must be None or positive value")
        if times > 0:
            old_tolerance = self.tolerance
            self.tolerance = tolerance
            self.degree -= int(times)
            self.tolerance = old_tolerance

    def degree_clean(self, tolerance: float = 1e-9):
        """Reduces au maximum the degree of the curve for given tolerance.

        Does nothing if cannot reduce the degree

        :param tolerance: The tolerance to reduce degree, defaults to ``1e-9``
        :type tolerance: float(, optional)
        :raises AssertionError: If ``tolerance`` is not a number >= 0

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1])
        >>> curve.ctrlpoints = [1, 4/3, 7/6, -1/6, -4/3, -3]
        >>> print(curve)
        Spline curve of degree 3 and 6 control points
        KnotVector = (0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1)
        ControlPoints = [1.0, 1.33, 1.17, -0.17, -1.33, -3.0]
        >>> curve.degree_clean()
        >>> print(curve)
        Spline curve of degree 2 and 4 control points
        KnotVector = (0, 0, 0, 0.5, 1, 1, 1)
        ControlPoints = [1, 1.5, -0.5, -3]
        """
        if not isscalar(tolerance) or tolerance <= 0:
            raise ValueError("Given tolerance must be positive")
        oldtolerance = self.tolerance
        try:
            self.tolerance = tolerance
            while True:
                self.degree -= 1
        except ValueError:
            self.tolerance = oldtolerance

    def clean(self, tolerance: float = 1e-9):
        """Calls degree_clean and knot_clean

        If the curve is rational, it tries to simplify,

        :param tolerance: The tolerance to reduce degree, defaults to ``1e-9``
        :type tolerance: float(, optional)
        :raises AssertionError: If ``tolerance`` is not a number >= 0

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1])
        >>> curve.ctrlpoints = [1, 4/3, 7/6, -1/6, -4/3, -3]
        >>> print(curve)
        Spline curve of degree 3 and 6 control points
        KnotVector = (0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1)
        ControlPoints = [1.0, 1.33, 1.17, -0.17, -1.33, -3.0]
        >>> curve.degree_clean()
        >>> print(curve)
        Spline curve of degree 2 and 4 control points
        KnotVector = (0, 0, 0, 0.5, 1, 1, 1)
        ControlPoints = [1, 1.5, -0.5, -3]

        """
        self.degree_clean(tolerance=tolerance)
        self.knot_clean(tolerance=tolerance)
        if self.weights is None:
            return
        # Try to reduce to spline
        knotvector = tuple(self.knotvector)
        weights = tuple(self.weights)
        ctrlpoints = tuple(self.ctrlpoints)
        mattrans, materror = func2func(
            knotvector, weights, knotvector, [1] * self.npts
        )
        error = np.dot(
            np.moveaxis(ctrlpoints, 0, -1), np.dot(materror, ctrlpoints)
        )
        error = np.max(abs(error))
        error = max(error, np.dot(weights, np.dot(materror, weights)))
        if error < tolerance:
            self.ctrlpoints = np.dot(mattrans, ctrlpoints)
            self.weights = None
            assert NotImplementedError  # Needs correction
            self.clean(tolerance)

    def split(self, nodes: Optional[Tuple[float]] = None) -> Tuple[Curve]:
        """Separate the current curve at specified nodes

        If no arguments are given, it splits at every knot, returning a
        list of bezier curves

        :param nodes: The positions to split, defaults to ``None``, all internal nodes
        :type tolerance: tuple[float](, optional)

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> knotvector = (0, 0, 0, 0.5, 1, 1, 1)
        >>> ctrlpoints = [2, 1, 3, 0]
        >>> curve = Curve(knotvector, ctrlpoints)
        >>> subcurves = curve.split([0.2, 0.8])
        >>> len(subcurves)
        3
        >>> subcurves[0].knotvector
        (0.0, 0.0, 0.0, 0.2, 0.2, 0.2)
        >>> subcurves[1].knotvector
        (0.2, 0.2, 0.2, 0.5, 0.8, 0.8, 0.8)
        >>> subcurves[2].knotvector
        (0.8, 0.8, 0.8, 1.0, 1.0, 1.0)

        """
        if nodes is None:
            nodes = self.knotvector.knots
        nodes = tuple(nodes)
        newvectors = self.knotvector.split(nodes)
        vector = tuple(self.knotvector)
        matrices = heavy.Operations.split_curve(vector, nodes)
        newcurves = []
        for newvector, matrix in zip(newvectors, matrices):
            matrix = np.array(matrix)
            newcurve = Curve(newvector)
            newcurve.ctrlpoints = np.dot(matrix, self.ctrlpoints)
            if self.weights is not None:
                newcurve.weights = np.dot(matrix, self.weights)
            newcurves.append(newcurve)
        return tuple(newcurves)

    def fit_curve(self, other: Curve, nodes: Tuple[float] = None) -> float:
        """Finds the control points such this curve keeps as near as
        possible to ``other``

        If nodes are given

        * if len(nodes) < npts
            interpolates all nodes, uses least square in the other degrees of freedom
        * if len(nodes) == npts
            interpolate at all points
        * if len(nodes) > npts:
            same as fit_points(other(nodes), nodes)

        :param other: The objective curve
        :type other: Curve
        :param nodes: The positions to fit, defaults to ``None``
        :type nodes: None | tuple[float](, optional)

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> knotvector = (0, 0, 0, 0.5, 1, 1, 1)
        >>> ctrlpoints = [2, 1, 3, 0]
        >>> curvea = Curve(knotvector, ctrlpoints)
        >>> curveb = Curve([0, 0, 0.5, 1, 1])
        >>> curveb.fit_curve(curvea)
        >>> print(curveb)
        Spline curve of degree 1 and 3 control points
        KnotVector = (0, 0, 0.5, 1, 1)
        ControlPoints = [1.417, 2.167, 0.917]

        """
        assert isinstance(other, self.__class__)
        vectora, vectorb = tuple(self.knotvector), tuple(other.knotvector)
        if self.weights is None and other.weights is None:
            lstsq = spline2spline
            transmat, materror = lstsq(vectorb, vectora, nodes)
        else:
            weightsa = self.weights if self.weights else [1] * self.npts
            weightsb = other.weights if other.weights else [1] * other.npts
            lstsq = func2func
            transmat, materror = lstsq(
                vectorb, weightsb, vectora, weightsa, nodes
            )
        transmat = np.array(transmat)
        ctrlpoints = np.dot(transmat, other.ctrlpoints)
        error = np.dot(
            np.moveaxis(other.ctrlpoints, 0, -1),
            np.dot(materror, other.ctrlpoints),
        )
        error = np.max(np.abs(error))
        if other.weights is not None:
            error += np.dot(other.weights, np.dot(materror, other.ctrlpoints))
            weights = np.dot(transmat, weightsb)
            ctrlpoints = [
                point / weig for point, weig in zip(ctrlpoints, weights)
            ]
            self.weights = weights
        self.ctrlpoints = ctrlpoints
        return error

    def fit_function(
        self, function: Callable, nodes: Tuple[float] = None
    ) -> None:
        """Finds the control points such this curve keeps as near as
        possible to ``function``

        If nodes are not given, it uses least square in many intervals
            for subinterval [uk, u_{k+1}] evaluates on
            max(degree+1, 5*npts/len(subintervals)) using chebyshev nodes

        * if len(nodes) < npts
            interpolates all nodes, uses least square in the other degrees of freedom
        * if len(nodes) == npts
            interpolate at all points
        * if len(nodes) > npts:
            same as fit_points(other(nodes), nodes)

        :param other: The objective curve
        :type other: Curve
        :param nodes: The positions to fit, defaults to ``None``
        :type nodes: None | tuple[float](, optional)

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> knotvector = (0, 0, 0.5, 1, 1)
        >>> curve = Curve(knotvector)
        >>> function = lambda x: 1 + x**2
        >>> curve.fit_function(function)
        >>> print(curve)
        Spline curve of degree 1 and 3 control points
        KnotVector = (0, 0, 0.5, 1, 1)
        ControlPoints = [0.969, 1.219, 1.969]

        """
        if nodes is not None:
            raise NotImplementedError
        assert not isinstance(function, self.__class__)
        knots = self.knotvector.knots
        npts_each = 1 + int(
            np.ceil(self.degree * self.npts / (len(knots) - 1))
        )
        nodes = []
        numbtype = number_type(knots)
        if numbtype in (float, np.floating):
            funcnodes = heavy.NodeSample.chebyshev
        else:
            funcnodes = heavy.NodeSample.open_linspace
        nodes_0to1 = funcnodes(npts_each)
        for start, end in zip(knots[:-1], knots[1:]):
            nodes += [start + (end - start) * node for node in nodes_0to1]
        nodes = tuple(nodes)
        funcvals = [function(node) for node in nodes]
        return self.fit_points(funcvals, nodes)

    def fit_points(
        self, points: Tuple[Any], nodes: Tuple[float] = None
    ) -> None:
        """Finds the control points such this curve keeps as near as
        possible to ``points``

        If nodes are not given, it supposes equally distributed nodes

        * if len(points) < npts
            ValueError
        * if len(nodes) == npts
            interpolate at all points
        * if len(nodes) > npts:
            Uses discrete least squares

        :param points: The objective points
        :type points: tuple[any]
        :param nodes: The positions to fit, defaults to ``None``, equally distributed points
        :type nodes: None | tuple[float](, optional)

        Example use
        -----------

        >>> import numpy as np
        >>> from pynurbs import Curve
        >>> knotvector = (0, 0, 0.5, 1, 1)
        >>> curve = Curve(knotvector)
        >>> function = lambda x: 1 + x**2
        >>> usample = np.linspace(0, 1, 129)
        >>> points = function(usample)
        >>> curve.fit_points(points)
        >>> print(curve)
        Spline curve of degree 1 and 3 control points
        KnotVector = (0, 0, 0.5, 1, 1)
        ControlPoints = [0.96, 1.21, 1.96]

        """
        assert len(points) >= self.npts
        if nodes is None:
            umin, umax = self.knotvector.limits
            if isinstance(umin, (int, Fraction)):
                funcnodes = heavy.NodeSample.closed_linspace
            else:
                funcnodes = heavy.NodeSample.chebyshev
            nodes_0to1 = funcnodes(len(points))
            nodes = tuple(umin + (umax - umin) * node for node in nodes_0to1)
        knotvector = tuple(self.knotvector)
        nodes = tuple(nodes)
        weights = None if self.weights is None else tuple(self.weights)
        matrix = fit_function(knotvector, nodes, weights)
        ctrlpoints = np.dot(matrix, points)
        self.ctrlpoints = tuple(ctrlpoints)

    def fit(
        self,
        param: Union[Curve, Callable[[float], float], Tuple[Any]],
        nodes: Optional[Tuple[float]] = None,
    ) -> None:
        """
        Calls ``fit_curve``, ``fit_function`` or ``fit_points`` depending on ``param``
        """
        if isinstance(param, self.__class__):
            return self.fit_curve(param, nodes)
        if callable(param):
            return self.fit_function(param, nodes)
        return self.fit_points(param, nodes)
