from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from compmec.nurbs import heavy
from compmec.nurbs.__classes__ import Intface_BaseCurve
from compmec.nurbs.knotspace import KnotVector


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


class BaseCurve(Intface_BaseCurve):
    def __init__(self, knotvector: KnotVector):
        self.__ctrlpoints = None
        self.__weights = None
        self.__knotvector = KnotVector(knotvector)

    def __call__(self, nodes: np.ndarray) -> np.ndarray:
        return self.eval(nodes)

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
        selfcopy = self.copy()
        selfcopy.knotvector = newknotvec
        othercopy = other.copy()
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
        newcurve = self.copy()
        newctrlpoints = [-1 * ctrlpt for ctrlpt in newcurve.ctrlpoints]
        newcurve.ctrlpoints = newctrlpoints
        return newcurve

    def __add__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copy = self.copy()
            copy.ctrlpoints = [other + point for point in self.ctrlpoints]
            return copy
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            matra, matrb = heavy.MathOperations.add_spline_curve(vecta, vectb)
            curve = Curve(self.knotvector | other.knotvector)
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
            copy = self.copy()
            copy.ctrlpoints = [point * other for point in copy.ctrlpoints]
            return copy
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
            curve = Curve(vectmul, ctrlpoints)
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa * numb) / (dena * denb)

    def __rmul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        assert not isinstance(other, self.__class__)
        copy = self.copy()
        copy.ctrlpoints = [other * point for point in copy.ctrlpoints]
        return copy

    def __matmul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copy = self.copy()
            copy.ctrlpoints = [point @ other for point in copy.ctrlpoints]
            return copy
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            vecta, vectb = tuple(self.knotvector), tuple(other.knotvector)
            vectmul = heavy.MathOperations.knotvector_mul(vecta, vectb)
            matrix3d = heavy.MathOperations.mul_spline_curve(vecta, vectb)
            matrix2d = [
                [pt0 @ pt1 for pt0 in self.ctrlpoints] for pt1 in other.ctrlpoints
            ]
            matrix3d = np.array(matrix3d)
            matrix2d = np.array(matrix2d)
            newctrlpts = [0] * matrix3d.shape[1]
            for i in range(matrix3d.shape[1]):
                newctrlpt = np.tensordot(matrix3d[:, i, :], matrix2d, axes=2)
                newctrlpts[i] = newctrlpt
            ctrlpoints = newctrlpts
            curve = Curve(vectmul, ctrlpoints)
            return curve
        numa, dena = self.fraction()
        numb, denb = other.fraction()
        return (numa @ numb) / (dena * denb)

    def __rmatmul__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        assert not isinstance(other, self.__class__)
        copy = self.copy()
        copy.ctrlpoints = [other @ point for point in copy.ctrlpoints]
        return copy

    def __truediv__(self, other: object):
        if self.ctrlpoints is None:
            raise ValueError
        if not isinstance(other, self.__class__):
            copy = self.copy()
            copy.ctrlpoints = [point / other for point in copy.ctrlpoints]
            return copy
        if self.knotvector.limits != other.knotvector.limits:
            raise ValueError
        if self.weights is None and other.weights is None:
            copyse = self.copy()
            copyot = other.copy()
            vectora, vectorb = tuple(copyse.knotvector), tuple(copyot.knotvector)
            vectorc = tuple(copyse.knotvector | copyot.knotvector)
            transctrlpts = heavy.Operations.matrix_transformation(vectora, vectorc)
            transweights = heavy.Operations.matrix_transformation(vectorb, vectorc)
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
            newcurve.weights = [deepcopy(point) for point in self.ctrlpoints]
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
        othercopy = other.copy()
        selfcopy = self.copy()
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
    def knotvector(self):
        """Knot Vector

        :getter: Returns the knotvector of the curve
        :setter: Sets the knotvector of the curve
        :type: KnotVector

        Example use
        -----------

        >>> from compmec.nurbs import Curve
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

        >>> from compmec.nurbs import Curve
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

        >>> from compmec.nurbs import Curve
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

        >>> from compmec.nurbs import Curve
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

        >>> from compmec.nurbs import Curve
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

    @knotvector.setter
    def knotvector(self, value: KnotVector):
        value = KnotVector(value)
        self.update(value)

    @degree.setter
    def degree(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Cannot set degree {value}")
        times = value - self.degree
        if times == 0:
            return
        if times > 0:
            return self.degree_increase(times)
        return self.degree_decrease(-times)

    @weights.setter
    def weights(self, value: Tuple[float]):
        if value is None:
            self.__weights = None
            return
        try:
            value = tuple(value)
            for val in value:
                float(val)
        except TypeError:
            msg = f"Weights must be a vector of floats, received {value}"
            raise ValueError(msg)
        # Verify if there's roots
        vector = tuple(self.knotvector)
        roots = heavy.find_roots(vector, value)
        if roots:
            error_msg = f"Zero division at nodes {roots}"
            raise ValueError(error_msg)
        self.__weights = tuple(value)

    @ctrlpoints.setter
    def ctrlpoints(self, newpoints: np.ndarray):
        if newpoints is None:
            self.__ctrlpoints = None
            return
        if isinstance(newpoints, str):
            raise TypeError
        try:
            iter(newpoints)
        except Exception:
            raise TypeError
        for point in newpoints:  # Verify if operations are valid for each node
            for knot in self.knotvector.knots:
                knot * point
            for otherpoint in newpoints:
                point + otherpoint  # Verify if we can sum every point, same type

        if len(newpoints) != self.npts:
            error_msg = f"The number of control points ({len(newpoints)}) must be "
            error_msg += f"the same as npts of KnotVector ({self.knotvector.npts})\n"
            error_msg += f"  knotvector.npts = {self.npts}"
            error_msg += f"  len(ctrlpoints) = {len(newpoints)}"
            raise ValueError(error_msg)

        self.__ctrlpoints = tuple(newpoints)

    def copy(self) -> Curve:
        """Returns a copy of the object. The knotvector, control points and weights are also copied.

        :return: An exact copy of the instance.
        :rtype: Curve

        Example use
        -----------

        >>> from compmec.nurbs import Curve
        >>> curveA = Curve([0, 0, 0, 1, 1, 1])
        >>> curveB = curveA.copy()
        >>> curveB == curveA
        True
        >>> id(curveB) == id(curveA)
        False

        """
        knotvector = self.knotvector.copy()
        curve = self.__class__(knotvector)
        if self.ctrlpoints is not None:
            curve.ctrlpoints = [deepcopy(point) for point in self.ctrlpoints]
        if self.weights is not None:
            curve.weights = [deepcopy(weight) for weight in self.weights]
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

        >>> from compmec.nurbs import Curve
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
            numerator = self.copy()
            return numerator, 1
        ctrlpoints = [deepcopy(point) for point in self.ctrlpoints]
        numerator = self.__class__(self.knotvector.copy())
        denominator = self.__class__(self.knotvector.copy())
        numerator.ctrlpoints = [wi * pt for wi, pt in zip(self.weights, ctrlpoints)]
        denominator.ctrlpoints = self.weights
        return numerator, denominator

    def update(
        self,
        newknotvector: KnotVector,
        tolerance: Optional[float] = 1e-9,
        nodes: Optional[tuple[float]] = None,
    ):
        """Update the knotvector to newknotvector

        This function compute the new control points and
        new weights such the new curve is near the old curve

        If the error is bigger than tolerance, then raises a ValueError

        If ``nodes`` are given, the new curve will fit on these ``nodes``

        Example use
        -----------

        >>> from compmec.nurbs import Curve
        >>> curve = Curve([0, 0, 1, 1], [1, 2])
        >>> curve.update([0, 0, 0.5, 1, 1])  # Insert knot [0.5]
        >>> curve = Curve([0, 0, 1, 1], [1, 2])
        >>> curve.update([0, 0, 0, 1, 1, 1])  # Degree elevate
        >>> curve = Curve([0, 0, 0.5, 1, 1], [1, 2, 1])
        >>> curve.update([0, 0, 1, 1], nodes = (0, 1))  # Remove knot [0.5]

        """
        newknotvector = KnotVector(newknotvector)
        if newknotvector == self.knotvector:
            return
        if self.ctrlpoints is None:
            self.__knotvector = newknotvector
            return
        if self.knotvector.limits != newknotvector.limits:
            raise ValueError
        temp_curve = self.__class__(newknotvector)
        error = temp_curve.fit_curve(self, nodes)
        if tolerance and error > tolerance:
            error_msg = "Cannot update knotvector cause error is "
            error_msg += f" {float(error):.2e} > {tolerance}"
            raise ValueError(error_msg)
        self.__knotvector = newknotvector
        self.ctrlpoints = temp_curve.ctrlpoints
        self.weights = temp_curve.weights

    def apply(self, newknotvector: KnotVector, matrix: Tuple[Tuple[float]]):
        """Applies the linear transformation for every control point

        new ctrlpoints = matrix @ old ctrlpoints
        new weights = matrix @ old weights

        Example use
        -----------

        >>> from compmec.nurbs import Curve
        >>> curve = Curve([0, 0, 0.5, 1, 1])
        >>> curve.ctrlpoints = [2, 4, 2]
        >>> matrix = [(0, 1, 0), (-1, 0, 1), (2, -1, 0)]
        >>> curve.apply(matrix)
        >>> print(curve)
        Spline curve of degree 1 and 3 control points
        KnotVector = (0, 0, 0.5, 1, 1)
        ControlPoints = [4, 0, 0]

        """
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

    def __eval(self, nodes: Tuple[float]) -> Tuple[Any]:
        """
        Private method to evaluate points in the curve
        """
        vector = tuple(self.knotvector)
        nodes = tuple(nodes)
        degree = int(self.knotvector.degree)
        if self.weights is None:
            eval = heavy.eval_spline_nodes
            matrix = eval(vector, nodes, degree)
        else:
            eval = heavy.eval_rational_nodes
            weights = tuple(self.weights)
            matrix = eval(vector, weights, nodes, degree)
        result = np.moveaxis(matrix, 0, -1) @ self.ctrlpoints
        return tuple(result)

    def eval(self, nodes: Union[float, Tuple[float]]) -> Union[Any, Tuple[Any]]:
        """Point evaluation function

        :param nodes: The nodes to evaluates
        :type nodes: float | tuple[float]
        :raises TypeError: If ``nodes`` is not a number or a list of numbers
        :raises ValueError: If at least node is outside ``[umin, umax]``
        :return: The point computed by using control points
        :rtype: Any | tuple[Any]

        Example use
        -----------

        >>> from compmec.nurbs import Curve
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
        if self.ctrlpoints is None:
            error_msg = "Cannot evaluate: There are no control points"
            raise ValueError(error_msg)
        try:
            nodes = tuple(nodes)
            onevalue = False
        except TypeError:
            nodes = (nodes,)
            onevalue = True
        self.knotvector.valid_nodes(nodes)
        result = self.__eval(nodes)
        return result[0] if onevalue else result

    def knot_insert(self, nodes: Tuple[float]) -> None:
        """Insert given nodes inside knotvector

        :param nodes: The nodes to be inserted
        :type nodes: tuple[float]
        :raises TypeError: If ``nodes`` is not a number or a list of numbers
        :raises ValueError: If it's not possible to insert the knots

        Example use
        -----------

        >>> from compmec.nurbs import Curve
        >>> curve = Curve([0, 0, 0.5, 1, 1])
        >>> curve.ctrlpoints = (1, 2, -3)
        >>> curve.knot_insert([0.2, 0.7])
        >>> curve.knotvector
        (0, 0, 0.2, 0.5, 0.7, 1, 1)

        """
        nodes = tuple(nodes)
        oldvector = tuple(self.knotvector)
        newvector = tuple(self.knotvector + nodes)
        if self.ctrlpoints is None and self.weights is None:
            self.knotvector = newvector
        matrix = heavy.Operations.knot_insert(oldvector, nodes)
        self.apply(newvector, matrix)

    def knot_remove(self, nodes: Tuple[float], tolerance: float = 1e-9) -> None:
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

        >>> from compmec.nurbs import Curve
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
        nodes = tuple(nodes)
        for node in nodes:
            float(node)
        newknotvec = self.knotvector.copy() - tuple(nodes)
        knots = newknotvec.knots if newknotvec.degree != 0 else None
        self.update(newknotvec, tolerance, knots)

    def knot_clean(
        self, nodes: Optional[Tuple[float]] = None, tolerance: Optional[float] = 1e-9
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

        >>> from compmec.nurbs import Curve
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
        float(tolerance)
        assert tolerance >= 0
        if nodes is None:
            nodes = self.knotvector.knots
        nodes = tuple(set(nodes) - set(self.knotvector.limits))
        for knot in nodes:
            try:
                while True:
                    self.knot_remove((knot,), tolerance)
            except ValueError:
                pass

    def degree_increase(self, times: Optional[int] = 1):
        """Increase the degree of the curve by an amount ``times``

        :param times: The number of times to increase, defaults to ``1``
        :type times: int(, optional)
        :raises AssertionError: If ``times`` is not a integer >= 0

        Example use
        -----------

        >>> from compmec.nurbs import Curve
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
        if not isinstance(times, int) or times <= 0:
            raise ValueError
        oldvector = tuple(self.knotvector)
        nodes = self.knotvector.knots
        newnodes = times * nodes
        newvector = heavy.KnotVector.insert_knots(oldvector, newnodes)
        matrix = heavy.Operations.degree_increase(oldvector, times)
        self.apply(newvector, matrix)

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

        >>> from compmec.nurbs import Curve
        >>> from compmec.nurbs import Curve
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
        if not isinstance(times, int) or times <= 0:
            raise ValueError(f"times = {times}")
        if tolerance is not None:
            float(tolerance)
            assert tolerance >= 0
        newknotvec = self.knotvector.copy()
        newknotvec.degree -= times
        knots = newknotvec.knots if newknotvec.degree != 0 else None
        self.update(newknotvec, tolerance, knots)

    def degree_clean(self, tolerance: float = 1e-9):
        """Reduces au maximum the degree of the curve for given tolerance.

        Does nothing if cannot reduce the degree

        :param tolerance: The tolerance to reduce degree, defaults to ``1e-9``
        :type tolerance: float(, optional)
        :raises AssertionError: If ``tolerance`` is not a number >= 0

        Example use
        -----------

        >>> from compmec.nurbs import Curve
        >>> from compmec.nurbs import Curve
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
        float(tolerance)
        assert tolerance >= 0
        try:
            while True:
                self.degree_decrease(1, tolerance)
        except ValueError:
            pass

    def clean(self, tolerance: float = 1e-9):
        """Calls degree_clean and knot_clean

        If the curve is rational, it tries to simplify,

        :param tolerance: The tolerance to reduce degree, defaults to ``1e-9``
        :type tolerance: float(, optional)
        :raises AssertionError: If ``tolerance`` is not a number >= 0

        Example use
        -----------

        >>> from compmec.nurbs import Curve
        >>> from compmec.nurbs import Curve
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
        mattrans, materror = heavy.LeastSquare.func2func(
            knotvector, weights, knotvector, [1] * self.npts
        )
        error = np.dot(np.moveaxis(ctrlpoints, 0, -1), np.dot(materror, ctrlpoints))
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

        >>> from compmec.nurbs import Curve
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
        vector = tuple(self.knotvector)
        if nodes is None:
            nodes = self.knotvector.knots
        nodes = tuple(nodes)
        matrices = heavy.Operations.split_curve(vector, nodes)
        newvectors = heavy.KnotVector.split(vector, nodes)
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

        >>> from compmec.nurbs import Curve
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
            lstsq = heavy.LeastSquare.spline2spline
            transmat, materror = lstsq(vectorb, vectora, nodes)
        else:
            weightsa = self.weights if self.weights else [1] * self.npts
            weightsb = other.weights if other.weights else [1] * other.npts
            lstsq = heavy.LeastSquare.func2func
            transmat, materror = lstsq(vectorb, weightsb, vectora, weightsa, nodes)
        transmat = np.array(transmat)
        ctrlpoints = np.dot(transmat, other.ctrlpoints)
        error = np.dot(
            np.moveaxis(other.ctrlpoints, 0, -1), np.dot(materror, other.ctrlpoints)
        )
        error = np.max(np.abs(error))
        if other.weights is not None:
            error += np.dot(other.weights, np.dot(materror, other.ctrlpoints))
            weights = np.dot(transmat, weightsb)
            ctrlpoints = [point / weig for point, weig in zip(ctrlpoints, weights)]
            self.weights = weights
        self.ctrlpoints = ctrlpoints
        return error

    def fit_function(self, function: Callable, nodes: Tuple[float] = None) -> None:
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

        >>> from compmec.nurbs import Curve
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
        npts_each = 1 + int(np.ceil(self.degree * self.npts / (len(knots) - 1)))
        nodes = []
        numbtype = heavy.number_type(knots)
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

    def fit_points(self, points: Tuple[Any], nodes: Tuple[float] = None) -> None:
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
        >>> from compmec.nurbs import Curve
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
        fitfunc = heavy.LeastSquare.fit_function
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
        matrix = fitfunc(knotvector, nodes, weights)
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
