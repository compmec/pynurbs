from __future__ import annotations

from copy import copy
from numbers import Real
from typing import Any, Iterable, Tuple, Union

import numpy as np
import rbool

from ..core.custom_math import isscalar, supports_linear_operation
from ..core.spline_basis import ImmutableSplineBasis
from ..knotspace import KnotVector
from ..operations import heavy
from ..operations.knotvector import decrease_degree, increase_degree
from ..operations.roots import roots_piecewise
from ..operations.tools import vectorize


def norm(object: Union[float, Tuple[float]], L: int = 0) -> float:
    """
    Computes recursively a norm of an object.
    If L = 0, it means infinity norm
    If L = 1, it means abs norm
    If L = 2, it means euclidean norm
    """
    try:
        soma = 0
        for item in object:
            norma = norm(item, L)
            soma = max(soma, norma) if L == 0 else soma + norma**L
        return soma if L == 0 else soma ** (1 / L)
    except TypeError:
        return abs(object)


class BaseCurve:
    def __init__(
        self,
        knotvector: KnotVector,
        ctrlpoints: Union[None, Iterable[Any]] = None,
        weights: Union[None, Iterable[Any]] = None,
    ):
        if not isinstance(knotvector, KnotVector):
            knotvector = KnotVector(knotvector)
        self.__knotvector = knotvector
        self.__ctrlpoints = ctrlpoints
        self.__weights = weights
        self.tolerance = 1e-9
        self.__denominator = None

    @vectorize(1, 0)
    def __call__(self, node: Real) -> Any:
        if self.ctrlpoints is None:
            raise ValueError("Cannot evaluate")
        vector = self.knotvector.internal
        basis = ImmutableSplineBasis(vector)
        result = basis(node)
        zero = 0 * self.ctrlpoints[0]
        if self.weights is None:
            return sum((r * c for r, c in zip(result, self.ctrlpoints)), zero)
        result = tuple(w * r for w, r in zip(self.weights, result))
        denom = 1 / sum(result)
        return sum(
            (r * c * denom for r, c in zip(result, self.ctrlpoints)), zero
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        if self.knotvector[0] != other.knotvector[0]:
            return False
        if self.knotvector[-1] != other.knotvector[-1]:
            return False
        if (self.ctrlpoints is None) ^ (other.ctrlpoints is None):
            return False
        newknotvec = self.knotvector | other.knotvector
        selfcopy = copy(self)
        selfcopy.knotvector = newknotvec
        othercopy = copy(other)
        othercopy.knotvector = newknotvec
        for poi, qoi in zip(self.ctrlpoints, othercopy.ctrlpoints):
            if norm(poi - qoi) > 1e-9:
                return False
        return True

    def __ne__(self, obj: object):
        return not self.__eq__(obj)

    def __neg__(self):
        if self.ctrlpoints is None:
            raise ValueError
        newcurve = copy(self)
        newctrlpoints = [-1 * ctrlpt for ctrlpt in newcurve.ctrlpoints]
        newcurve.ctrlpoints = newctrlpoints
        return newcurve

    def __add__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copied = copy(self)
            copied.ctrlpoints = [other + point for point in self.ctrlpoints]
            return copied
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            matra, matrb = heavy.MathOperations.add_spline_curve(vecta, vectb)
            curve = self.__class__(self.knotvector | other.knotvector)
            ctrlpoints = np.array(matra) @ self.ctrlpoints
            ctrlpoints += np.array(matrb) @ other.ctrlpoints
            curve.ctrlpoints = ctrlpoints
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa * denb + numb * dena) / (dena * denb)

    def __radd__(self, other: object):
        return self.__add__(other)

    def __sub__(self, other: object):
        return self + (-other)

    def __rsub__(self, other: object):
        return other + (-self)

    def __mul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copied = copy(self)
            copied.ctrlpoints = [point * other for point in copied.ctrlpoints]
            return copied
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            vectmul = heavy.MathOperations.knotvector_mul(vecta, vectb)
            matrix3d = heavy.MathOperations.mul_spline_curve(vecta, vectb)
            ctrlpoints = np.tensordot(
                np.moveaxis(self.ctrlpoints, 0, -1), matrix3d, axes=1
            )
            ctrlpoints = ctrlpoints @ other.ctrlpoints
            curve = self.__class__(vectmul, ctrlpoints)
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa * numb) / (dena * denb)

    def __rmul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        assert not isinstance(other, self.__class__)
        copied = copy(self)
        copied.ctrlpoints = [other * point for point in copied.ctrlpoints]
        return copied

    def __matmul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copied = copy(self)
            copied.ctrlpoints = [point @ other for point in copied.ctrlpoints]
            return copied
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            vectmul = heavy.MathOperations.knotvector_mul(vecta, vectb)
            matrix3d = heavy.MathOperations.mul_spline_curve(vecta, vectb)
            matrix2d = [
                [pt0 @ pt1 for pt0 in self.ctrlpoints]
                for pt1 in other.ctrlpoints
            ]
            matrix3d = np.array(matrix3d)
            matrix2d = np.array(matrix2d)
            newctrlpts = [0] * matrix3d.shape[1]
            for i in range(matrix3d.shape[1]):
                newctrlpt = np.tensordot(matrix3d[:, i, :], matrix2d, axes=2)
                newctrlpts[i] = newctrlpt
            ctrlpoints = newctrlpts
            curve = self.__class__(vectmul, ctrlpoints)
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa @ numb) / (dena * denb)

    def __rmatmul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        assert not isinstance(other, self.__class__)
        copied = copy(self)
        copied.ctrlpoints = [other @ point for point in copied.ctrlpoints]
        return copied

    def __truediv__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copied = copy(self)
            copied.ctrlpoints = [point / other for point in copied.ctrlpoints]
            return copied
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            copyse = copy(self)
            copyot = copy(other)
            vectora, vectorb = tuple(copyse.knotvector), tuple(
                copyot.knotvector
            )
            vectorc = tuple(copyse.knotvector | copyot.knotvector)
            transctrlpts = heavy.Operations.matrix_transformation(
                vectora, vectorc
            )
            transweights = heavy.Operations.matrix_transformation(
                vectorb, vectorc
            )
            weights = np.dot(transweights, copyot.ctrlpoints)
            ctrlpts = np.dot(transctrlpts, copyse.ctrlpoints)
            ctrlpts = [pti / wi for pti, wi in zip(ctrlpts, weights)]
            return self.__class__(vectorc, ctrlpts, weights)

        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa * denb) / (dena * numb)

    def __rtruediv__(self, other: object):
        """
        Example: 1/curve
        """
        if self.ctrlpoints is None:
            raise ValueError
        assert not isinstance(other, self.__class__)
        for point in self.ctrlpoints:
            float(point)
        if self.weights is None:
            newcurve = self.__class__(tuple(self.knotvector))
            newcurve.weights = [copy(point) for point in self.ctrlpoints]
            newcurve.ctrlpoints = [1 / w for w in newcurve.weights]
            return newcurve
        num, den = self.fraction()
        frac = den / num
        return other * frac

    def __or__(self, other: object):
        umaxleft = self.knotvector[-1]
        uminright = other.knotvector[0]
        if umaxleft != uminright:
            error_msg = f"max(Uleft) = {umaxleft} != {uminright} = min(Uright)"
            raise ValueError(error_msg)
        othercopy = copy(other)
        selfcopy = copy(self)
        maxdegree = max(self.degree, other.degree)
        selfcopy.degree = maxdegree
        othercopy.degree = maxdegree
        npts0 = selfcopy.npts
        npts1 = othercopy.npts
        newknotvector = [0] * (maxdegree + npts0 + npts1 + 1)
        newknotvector[:npts0] = selfcopy.knotvector[:npts0]
        newknotvector[npts0:] = othercopy.knotvector[1:]
        newknotvector = KnotVector(newknotvector)
        newctrlpoints = [0] * (npts0 + npts1 - 1)
        newctrlpoints[:npts0] = selfcopy.ctrlpoints[:npts0]
        newctrlpoints[npts0:] = othercopy.ctrlpoints[1:]
        newcurve = self.__class__(newknotvector, newctrlpoints)
        newcurve.knot_clean([umaxleft])
        return newcurve

    @property
    def tolerance(self) -> Union[None, float]:
        return self.__tolerance

    @property
    def knotvector(self):
        """Knot Vector

        :getter: Returns the knotvector of the curve
        :setter: Sets the knotvector of the curve
        :type: KnotVector

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0., 0., 1., 1.])
        >>> curve.knotvector
        (0., 0., 1., 1.)
        >>> curve.knotvector = [0, 0, 0, 1, 1, 1]
        >>> curve.knotvector
        (0, 0, 0, 1, 1, 1)

        """

        return self.__knotvector

    @property
    def degree(self):
        """Polynomial degree of curve

        :getter: Returns the degree of the curve
        :setter: Sets the degree of the curve, by increasing or decreasing
        :type: int

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0., 0., 1., 1.], [1, 2])
        >>> curve.degree = 3  # From 1 to 3
        >>> print(curve.ctrlpoints)
        (1.0, 1.33, 1.67, 2.0)
        >>> curve.degree -= 1
        >>> print(curve.ctrlpoints)
        (1.0, 1.5, 2.0)

        """

        return self.knotvector.degree

    @property
    def npts(self):
        """Number of control points

        :getter: Returns the number of control points of the curve
        :type: int

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0., 0., 1., 1.], [1, 2])
        >>> curve.npts
        2
        >>> curve = Curve([0., 0., 0.5, 1., 1.], [1, 2, 1])
        >>> curve.npts
        3

        """
        return self.knotvector.npts

    @property
    def knots(self):
        return self.knotvector.knots

    @property
    def weights(self):
        """Weights of rational curve

        If weights is None, the curve is a spline

        :getter: Returns the weights of curve
        :setter: Sets the weights for a rational curve
        :type: None | tuple[float]

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0., 0., 1., 1.], [1, 2])
        >>> curve.weights = [1, 2]
        (1, 2)
        >>> curve = Curve([0., 0., 0.5, 1., 1.], [1, 2, 1])
        >>> curve.weights = [1, 2, 1]
        >>> curve.weights
        (1, 2, 1)

        """
        if self.__weights is None:
            return None
        return tuple(self.__weights)

    @property
    def ctrlpoints(self):
        """Control points of the curve

        :getter: Returns the control points of the curve
        :setter: Sets the control points of the curve
        :type: None | tuple[Any]

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0., 0., 1., 1.])
        >>> curve.ctrlpoints = [1, 2]
        >>> curve.ctrlpoints
        (1, 2)
        >>> curve = Curve([0., 0., 0.5, 1., 1.])
        >>> curve.ctrlpoints = [1, 2, 1]
        >>> curve.ctrlpoints
        (1, 2, 1)

        """
        if self.__ctrlpoints is None:
            return None
        return tuple(self.__ctrlpoints)

    @tolerance.setter
    def tolerance(self, value: Union[None, float]):
        if value is not None and (not isscalar(value) or value <= 0):
            raise ValueError
        self.__tolerance = value

    @knotvector.setter
    def knotvector(self, value: KnotVector):
        if not isinstance(value, KnotVector):
            value = KnotVector(value)
        if self.ctrlpoints is not None and self.knotvector != value:
            if self.knotvector.limits != value.limits:
                raise ValueError
            temp_curve = self.__class__(value)
            error = temp_curve.fit_curve(self)
            if self.tolerance is not None and error > self.tolerance:
                error_msg = "Cannot update knotvector cause error is "
                error_msg += f" {float(error):.2e} > {self.tolerance}"
                raise ValueError(error_msg)
            self.__ctrlpoints = temp_curve.ctrlpoints
            self.__weights = temp_curve.weights
        self.__knotvector = value

    @degree.setter
    def degree(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Cannot set degree {value}")
        times = value - self.degree
        if times > 0:
            self.knotvector = increase_degree(self.knotvector.internal, times)
        elif times < 0:
            self.knotvector = decrease_degree(self.knotvector.internal, -times)

    @weights.setter
    def weights(self, value: Tuple[float]):
        if value is None:
            self.__weights = None
            return
        value = tuple(value)
        if not all(map(isscalar, value)):
            raise ValueError
        if not all(number > 0 for number in value):
            raise ValueError
        if len(value) != self.npts:
            raise ValueError

        # Verify if there's roots
        basis = ImmutableSplineBasis(self.knotvector.internal)
        denominator = 0
        for i, weight in enumerate(value):
            denominator += weight * basis[i]
        roots_values = roots_piecewise(denominator)
        if roots_values != rbool.EmptyR1():
            raise ValueError(f"Zero division at {roots_values}")
        self.__weights = tuple(value)

    @ctrlpoints.setter
    def ctrlpoints(self, newpoints: np.ndarray):
        if newpoints is None:
            self.__ctrlpoints = None
            return
        if not all(map(supports_linear_operation, newpoints)):
            raise ValueError
        for point in newpoints:  # Verify if operations are valid for each node
            for knot in self.knotvector.knots:
                knot * point
            for otherpoint in newpoints:
                (
                    point + otherpoint
                )  # Verify if we can sum every point, same type

        if len(newpoints) != self.npts:
            error_msg = (
                f"The number of control points ({len(newpoints)}) must be "
            )
            error_msg += (
                f"the same as npts of KnotVector ({self.knotvector.npts})\n"
            )
            error_msg += f"  knotvector.npts = {self.npts}"
            error_msg += f"  len(ctrlpoints) = {len(newpoints)}"
            raise ValueError(error_msg)

        self.__ctrlpoints = tuple(newpoints)

    def __copy__(self) -> Curve:
        return self.__deepcopy__(None)

    def __deepcopy__(self, memo) -> Curve:
        knotvector = copy(self.knotvector)
        curve = self.__class__(knotvector)
        if self.ctrlpoints is not None:
            curve.ctrlpoints = [copy(point) for point in self.ctrlpoints]
        if self.weights is not None:
            curve.weights = [copy(weight) for weight in self.weights]
        return curve

    def fraction(self) -> Tuple[BaseCurve]:
        """Returns the current curve ``C`` in the form ``A``/``B``

        ``A`` and ``B`` are bsplines of same degree as ``C``
        and same number of points.

        If ``C`` is already a bspline, then ``B = 1``

        :return: The pair ``(A, B)``
        :rtype: tuple[curve]

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0.5, 1, 1])
        >>> curve.ctrlpoints = [2, 4, 2]
        >>> curve.weights = [1, 3, 2]
        >>> A, B = curve.fraction()
        >>> print(A)
        Spline curve of degree 1 and 3 control points
        KnotVector = (0, 0, 0.5, 1, 1)
        ControlPoints = [2, 12, 4]
        >>> print(B)
        Spline curve of degree 1 and 3 control points
        KnotVector = (0, 0, 0.5, 1, 1)
        ControlPoints = [1, 3, 2]

        """
        if self.weights is None:
            numerator = copy(self)
            return numerator, 1
        ctrlpoints = [copy(point) for point in self.ctrlpoints]
        numerator = self.__class__(copy(self.knotvector))
        denominator = self.__class__(copy(self.knotvector))
        numerator.ctrlpoints = [
            wi * pt for wi, pt in zip(self.weights, ctrlpoints)
        ]
        denominator.ctrlpoints = self.weights
        return numerator, denominator

    def apply(self, newknotvector: KnotVector, matrix: Tuple[Tuple[float]]):
        """Applies the linear transformation for every control point

        new ctrlpoints = matrix @ old ctrlpoints
        new weights = matrix @ old weights

        Example use
        -----------

        >>> from pynurbs import Curve
        >>> curve = Curve([0, 0, 0.5, 1, 1])
        >>> curve.ctrlpoints = [2, 4, 2]
        >>> matrix = [(0, 1, 0), (-1, 0, 1), (2, -1, 0)]
        >>> curve.apply(matrix)
        >>> print(curve)
        Spline curve of degree 1 and 3 control points
        KnotVector = (0, 0, 0.5, 1, 1)
        ControlPoints = [4, 0, 0]

        """
        if not isinstance(newknotvector, KnotVector):
            newknotvector = KnotVector(newknotvector)
        oldctrlpoints = self.ctrlpoints
        oldweights = self.weights
        if oldctrlpoints is None and oldweights is None:
            self.knotvector = newknotvector
            return
        self.ctrlpoints = None
        self.weights = None
        self.knotvector = newknotvector
        if oldweights is None:
            self.ctrlpoints = np.dot(matrix, oldctrlpoints)
            return
        newweights = np.dot(matrix, oldweights)
        self.weights = newweights

        if oldctrlpoints is not None:
            oldctrlpoints = list(oldctrlpoints)
            for i, weight in enumerate(oldweights):
                oldctrlpoints[i] *= weight
            newctrlpoints = []
            for i, line in enumerate(matrix):
                newctrlpoints.append(0 * oldctrlpoints[0])
                for j, point in enumerate(oldctrlpoints):
                    newpoint = line[j] * point
                    newpoint /= self.weights[i]
                    newctrlpoints[i] += newpoint
            self.ctrlpoints = newctrlpoints
