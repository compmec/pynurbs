from fractions import Fraction
from typing import Any, Callable, Optional

import numpy as np

from compmec.nurbs import heavy
from compmec.nurbs.curves import Curve
from compmec.nurbs.knotspace import KnotVector


class Derivate:
    def __new__(cls, curve: Curve):
        return Derivate.curve(curve)

    @staticmethod
    def curve(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.ctrlpoints is not None
        if curve.degree == 0:
            limits = curve.knotvector.limits
            zero = 0 * curve.ctrlpoints[0]
            return curve.__class__(limits, [zero])
        if curve.degree + 1 == curve.npts:
            return Derivate.bezier(curve)
        return Derivate.spline(curve)

    @staticmethod
    def bezier(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.degree + 1 == curve.npts
        if curve.weights is None:
            return Derivate.nonrational_bezier(curve)
        return Derivate.rational_bezier(curve)

    @staticmethod
    def spline(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.degree + 1 != curve.npts
        if curve.weights is None:
            return Derivate.nonrational_spline(curve)
        return Derivate.rational_spline(curve)

    @staticmethod
    def nonrational_bezier(curve: Curve) -> Curve:
        """ """
        assert curve.degree + 1 == curve.npts
        assert curve.weights is None
        vector = tuple(curve.knotvector)
        matrix = heavy.Calculus.derivate_nonrational_bezier(vector)
        ctrlpoints = tuple(np.dot(matrix, curve.ctrlpoints))
        newcurve = curve.__class__(vector[1:-1], ctrlpoints)
        newcurve.clean()
        return newcurve

    @staticmethod
    def rational_bezier(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.degree + 1 == curve.npts
        assert curve.weights is not None
        assert np.all(np.array(curve.weights) != 0)

        knotvector = tuple(curve.knotvector)
        matrixup, matrixdo = heavy.Calculus.derivate_rational_bezier(knotvector)
        num, den = curve.fraction()
        matrixup = np.dot(matrixup, den.ctrlpoints)
        matrixdo = np.dot(matrixdo, den.ctrlpoints)

        dennumctrlpts = den.ctrlpoints @ matrixdo
        newnumctrlpts = np.dot(np.transpose(matrixup), num.ctrlpoints)
        newnumctrlpts = [
            point / weight for point, weight in zip(newnumctrlpts, dennumctrlpts)
        ]
        number_bound = 1 + 2 * curve.degree
        newknotvector = number_bound * [knotvector[0]] + number_bound * [knotvector[-1]]
        finalcurve = curve.__class__(newknotvector)
        finalcurve.ctrlpoints = newnumctrlpts
        finalcurve.weights = dennumctrlpts
        return finalcurve

    def nonrational_spline(curve: Curve) -> Curve:
        assert isinstance(curve, Curve)
        assert curve.weights is None
        assert curve.degree + 1 != curve.npts

        knotvector = tuple(curve.knotvector)
        matrix = heavy.Calculus.derivate_nonrational_spline(knotvector)
        ctrlpoints = np.dot(matrix, curve.ctrlpoints)
        newvector = heavy.KnotVector.derivate(knotvector)
        newcurve = curve.__class__(newvector, ctrlpoints)
        return newcurve

    @staticmethod
    def rational_spline(curve: Curve) -> Curve:
        numer, denom = curve.fraction()
        dnumer = Derivate.nonrational_spline(numer)
        dnumer.degree_increase(1)  # Shouldn't be necessary
        ddenom = Derivate.nonrational_spline(denom)
        ddenom.degree_increase(1)  # Needs further correction
        deriva = dnumer * denom - numer * ddenom
        deriva /= denom * denom
        return deriva


class Integrate:
    @staticmethod
    def scalar(
        curve: Curve,
        function: Optional[Callable[[float], Any]] = None,
        method: Optional[str] = None,
        nnodes: Optional[int] = None,
    ) -> float:
        """Computes the integral I

        If no ``function`` is given, it supposes that :math:`g(u)=1`

        If no ``method = (algo, npts)`` is given

        * If ``knotvector`` number type is ``int`` or ``Fraction``
            * ``npts = 1 + curve.degree``
            * ``algo = "closed-newton-cotes"``
        * Else
            * ``npts = 1 + curve.degree``
            * ``algo = "chebyshev"``

        Valid algorithms ``algo`` are ``closed-newton-cotes``, ``open-newton-cotes``, ``chebyshev`` or ``gauss-legendre``

        :param curve: The curve :math:`\mathbf{C}(u)` to integrate
        :type curve: Curve
        :param function: The weight function :math:`g(u)`, defaults to ``None``
        :type function: None | callable[[float], Any](, optional)
        :param method: The integration method, defaults to ``None``
        :type method: None | tuple[str, int](, optional)

        """
        nodes_functs = {
            "closed-newton-cotes": heavy.NodeSample.closed_linspace,
            "open-newton-cotes": heavy.NodeSample.open_linspace,
            "chebyshev": heavy.NodeSample.chebyshev,
            "gauss-legendre": heavy.NodeSample.gauss_legendre,
        }
        array_functs = {
            "closed-newton-cotes": heavy.IntegratorArray.closed_newton_cotes,
            "open-newton-cotes": heavy.IntegratorArray.open_newton_cotes,
            "chebyshev": heavy.IntegratorArray.chebyshev,
            "gauss-legendre": heavy.IntegratorArray.gauss_legendre,
        }
        assert isinstance(curve, Curve)
        if function is None:
            function = lambda u: 1
        if method is not None:
            pass
        elif isinstance(curve.knotvector[0], (int, Fraction)):
            method = "open-newton-cotes"
        else:
            method = "chebyshev"
        if nnodes is None:
            nnodes = 1 + curve.degree
        nodes_func = nodes_functs[method]
        integ_array_func = array_functs[method]
        nodes_0to1 = nodes_func(nnodes)
        integ_array = integ_array_func(nnodes)
        knots = curve.knotvector.knots
        integrals = []
        for start, end in zip(knots[:-1], knots[1:]):
            nodes = tuple(start + (end - start) * node for node in nodes_0to1)
            curve_vals = tuple(curve.eval(node) for node in nodes)
            function_vals = tuple(function(node) for node in nodes)
            new_integral = sum(
                map(np.prod, zip(integ_array, function_vals, curve_vals))
            )
            integrals.append((end - start) * new_integral)
        return sum(integrals)

    @staticmethod
    def lenght(
        curve: Curve,
        function: Optional[Callable[[float], Any]] = None,
        method: Optional[str] = None,
        nnodes: Optional[int] = None,
    ) -> float:
        """Computes the integral I

        The operation ``@`` is needed cause ``norm(curve(u)) = numpy.sqrt(curve(u) @ curve(u))``

        If no ``function`` is given, it supposes that :math:`g(u)=1`

        If ``method == None``:
            * If ``knotvector`` number type is ``int`` or ``Fraction``
                * ``method = "closed-newton-cotes"``
            * Else
                * ``npts = 1 + curve.degree``
                * ``method = "chebyshev"``

        Valid algorithms ``algo`` are ``closed-newton-cotes``, ``open-newton-cotes``, ``chebyshev`` or ``gauss-legendre``

        :param curve: The curve :math:`\mathbf{C}(u)` to integrate
        :type curve: Curve
        :param function: The weight function :math:`g(u)`, defaults to ``None``
        :type function: None | callable[[float], Any](, optional)
        :param method: The integration method, defaults to ``None``
        :type method: None | tuple[str, int](, optional)

        """
        dcurve = Derivate.curve(curve)
        return Integrate.density(dcurve, function, method, nnodes)

    @staticmethod
    def density(
        curve: Curve,
        function: Optional[Callable[[float], Any]] = None,
        method: Optional[str] = None,
        nnodes: Optional[int] = None,
    ) -> float:
        """Computes the integral I

        The operation ``@`` is needed cause ``norm(curve(u)) = numpy.sqrt(curve(u) @ curve(u))``

        If no ``function`` is given, it supposes that :math:`g(u)=1`

        If no ``method = (algo, npts)`` is given

        * If ``knotvector`` number type is ``int`` or ``Fraction``
            * ``npts = 1 + curve.degree``
            * ``algo = "closed-newton-cotes"``
        * Else
            * ``npts = 1 + curve.degree``
            * ``algo = "chebyshev"``

        Valid algorithms ``algo`` are ``closed-newton-cotes``, ``open-newton-cotes``, ``chebyshev`` or ``gauss-legendre``

        :param curve: The curve :math:`\mathbf{C}(u)` to integrate
        :type curve: Curve
        :param function: The weight function :math:`g(u)`, defaults to ``None``
        :type function: None | callable[[float], Any](, optional)
        :param method: The integration method, defaults to ``None``
        :type method: None | tuple[str, int](, optional)

        """
        nodes_functs = {
            "closed-newton-cotes": heavy.NodeSample.closed_linspace,
            "open-newton-cotes": heavy.NodeSample.open_linspace,
            "chebyshev": heavy.NodeSample.chebyshev,
            "gauss-legendre": heavy.NodeSample.gauss_legendre,
        }
        array_functs = {
            "closed-newton-cotes": heavy.IntegratorArray.closed_newton_cotes,
            "open-newton-cotes": heavy.IntegratorArray.open_newton_cotes,
            "chebyshev": heavy.IntegratorArray.chebyshev,
            "gauss-legendre": heavy.IntegratorArray.gauss_legendre,
        }
        assert isinstance(curve, Curve)
        if function is None:
            function = lambda u: 1
        if method is not None:
            pass
        elif isinstance(curve.knotvector[0], (int, Fraction)):
            method = "open-newton-cotes"
        else:
            method = "chebyshev"
        if nnodes is None:
            nnodes = 1 + curve.degree
        nodes_func = nodes_functs[method]
        integ_array_func = array_functs[method]
        nodes_0to1 = nodes_func(nnodes)
        integ_array = integ_array_func(nnodes)
        knots = curve.knotvector.knots
        integrals = []
        for start, end in zip(knots[:-1], knots[1:]):
            nodes = tuple(start + (end - start) * node for node in nodes_0to1)
            curve_vals = tuple(curve.eval(node) for node in nodes)
            abscurve_vals = tuple(np.sqrt(val @ val) for val in curve_vals)
            function_vals = tuple(function(node) for node in nodes)
            new_integral = sum(
                map(np.prod, zip(integ_array, function_vals, abscurve_vals))
            )
            integrals.append((end - start) * new_integral)
        return sum(integrals)

    @staticmethod
    def function(
        knotvector: KnotVector,
        function: Optional[Callable[[float], Any]],
        method: Optional[str] = None,
        nnodes: Optional[int] = None,
    ) -> float:
        """Computes the integral I

        .. math::
            I = \int_{a}^{b} g ( u ) \ du

        The operation ``@`` is needed cause ``norm(curve(u)) = numpy.sqrt(curve(u) @ curve(u))``

        If no ``function`` is given, it supposes that :math:`g(u)=1`

        If no ``method = (algo, npts)`` is given

        * If ``knotvector`` number type is ``int`` or ``Fraction``
            * ``npts = 1 + curve.degree``
            * ``algo = "closed-newton-cotes"``
        * Else
            * ``npts = 1 + curve.degree``
            * ``algo = "chebyshev"``

        Valid algorithms ``algo`` are ``closed-newton-cotes``, ``open-newton-cotes``, ``chebyshev`` or ``gauss-legendre``

        :param curve: The curve :math:`\mathbf{C}(u)` to integrate
        :type curve: Curve
        :param function: The weight function :math:`g(u)`, defaults to ``None``
        :type function: None | callable[[float], Any](, optional)
        :param method: The integration method, defaults to ``None``
        :type method: None | tuple[str, int](, optional)

        """
        nodes_functs = {
            "closed-newton-cotes": heavy.NodeSample.closed_linspace,
            "open-newton-cotes": heavy.NodeSample.open_linspace,
            "chebyshev": heavy.NodeSample.chebyshev,
            "gauss-legendre": heavy.NodeSample.gauss_legendre,
        }
        array_functs = {
            "closed-newton-cotes": heavy.IntegratorArray.closed_newton_cotes,
            "open-newton-cotes": heavy.IntegratorArray.open_newton_cotes,
            "chebyshev": heavy.IntegratorArray.chebyshev,
            "gauss-legendre": heavy.IntegratorArray.gauss_legendre,
        }
        if method is not None:
            pass
        elif isinstance(knotvector[0], (int, Fraction)):
            method = "open-newton-cotes"
        else:
            method = "chebyshev"
        if nnodes is None:
            nnodes = 1 + knotvector.degree
        nodes_func = nodes_functs[method]
        integ_array_func = array_functs[method]
        nodes_0to1 = nodes_func(nnodes)
        integ_array = integ_array_func(nnodes)
        knots = knotvector.knots
        integrals = []
        for start, end in zip(knots[:-1], knots[1:]):
            nodes = tuple(start + (end - start) * node for node in nodes_0to1)
            function_vals = tuple(function(node) for node in nodes)
            new_integral = sum(map(np.prod, zip(integ_array, function_vals)))
            integrals.append((end - start) * new_integral)
        return sum(integrals)
