from numbers import Real
from typing import Iterable

from .knotvector import ImmutableKnotVector


def insert_knots(
    knotvector: ImmutableKnotVector, nodes: Iterable[Real]
) -> ImmutableKnotVector:
    """
    Insert the given nodes into the knotvector

    Example
    -------
    >>> knotvector = ImmutableKnotVector([0, 0, 0, 1, 1, 1])
    >>> insert_knots(knotvector, [0.5, 0.5])
    (0, 0, 0, 0.5, 0.5, 1, 1, 1)
    """
    if not isinstance(knotvector, ImmutableKnotVector):
        raise TypeError
    nodes = list(nodes)
    if len(nodes) == 0:
        return knotvector
    new_knots = sorted(list(knotvector) + list(nodes))
    return ImmutableKnotVector(new_knots, knotvector.degree)


def remove_knots(
    knotvector: ImmutableKnotVector, nodes: Iterable[Real]
) -> ImmutableKnotVector:
    """
    Remove the given nodes from the knotvector

    Example
    -------
    >>> knotvector = ImmutableKnotVector([0, 0, 0, 0.5, 0.5, 1, 1, 1])
    >>> remove_knots(knotvector, [0.5])
    (0, 0, 0, 0.5, 1, 1, 1)
    """
    if not isinstance(knotvector, ImmutableKnotVector):
        raise TypeError
    nodes = list(nodes)
    new_knots = list(knotvector)
    if len(nodes) == 0:
        return knotvector
    for node in nodes:
        new_knots.remove(node)
    return ImmutableKnotVector(new_knots, knotvector.degree)


def increase_degree(knotvector: ImmutableKnotVector, times: int) -> ImmutableKnotVector:
    """
    Increases the degree of the given knotvector

    Example
    -------
    >>> knotvector = ImmutableKnotVector([0, 0, 0, 0.5, 1, 1, 1])
    >>> increase_degree(knotvector, 1)
    (0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1)
    >>> increase_degree(knotvector, 2)
    (0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1)
    """
    if not isinstance(knotvector, ImmutableKnotVector):
        raise TypeError
    if times < 0:
        raise ValueError
    if times == 0:
        return knotvector
    new_knots = sorted(list(knotvector) + times * list(knotvector.knots))
    return ImmutableKnotVector(new_knots, knotvector.degree + times)


def decrease_degree(knotvector: ImmutableKnotVector, times: int) -> ImmutableKnotVector:
    """
    Decreases the degree of the given knotvector

    Example
    -------
    >>> vector = [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]
    >>> knotvector = ImmutableKnotVector(vector)
    >>> decrease_degree(knotvector, 1)
    (0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1)
    >>> decrease_degree(knotvector, 2)
    (0, 0, 0, 0.5, 1, 1, 1)
    >>> decrease_degree(knotvector, 3)
    (0, 0, 1, 1)
    """
    if not isinstance(knotvector, ImmutableKnotVector):
        raise TypeError
    if times < 0:
        raise ValueError
    if times == 0:
        return knotvector
    knots_to_remove = times * list(knotvector.knots)
    new_knots = list(knotvector)
    for knot in knots_to_remove:
        new_knots.remove(knot)
    final = ImmutableKnotVector(new_knots, knotvector.degree - times)
    return final


def split_knotvector(
    knotvector: ImmutableKnotVector, nodes: Iterable[Real]
) -> Iterable[ImmutableKnotVector]:
    """
    Splits the given knotvector in the given nodes

    Example
    -------
    >>> vector = [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]
    >>> knotvector = ImmutableKnotVector(vector)
    >>> split_knotvector(knotvector, [0.3, 0.7])
    [(0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.3]
     (0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7],
     (0.7, 0.7, 0.7, 0.7, 0.7, 1, 1, 1, 1, 1])]
    """
    if not isinstance(knotvector, ImmutableKnotVector):
        raise TypeError
    degree = knotvector.degree
    nodes = sorted(set(nodes) | {knotvector.knots[0], knotvector.knots[-1]})
    for a, b in zip(nodes[:-1], nodes[1:]):
        middle = list(knot for knot in knotvector if (a < knot < b))
        newknotvect = (degree + 1) * [a] + middle + (degree + 1) * [b]
        yield ImmutableKnotVector(newknotvect)


def union_knotvectors(
    knotvectors: Iterable[ImmutableKnotVector],
) -> ImmutableKnotVector:
    """
    Computes the union of the given knotvectors
    """
    knotvectors = tuple(knotvectors)
    if not all(isinstance(vec, ImmutableKnotVector) for vec in knotvectors):
        raise TypeError
    left, right = knotvectors[0].knots[0], knotvectors[0].knots[-1]
    if any(vec.knots[0] != left or vec.knots[-1] != right for vec in knotvectors):
        raise ValueError
    maxdeg = max(vec.degree for vec in knotvectors)
    internals = {}
    for knotvector in knotvectors:
        if knotvector.degree < maxdeg:
            knotvector = increase_degree(knotvector, maxdeg - knotvector.degree)
        for knot in knotvector.knots[1:-1]:
            if knot not in internals:
                internals[knot] = 0
            internals[knot] = max(internals[knot], knotvector.mult(knot))
    final = [left] * (maxdeg + 1)
    for knot in sorted(internals.keys()):
        final += internals[knot] * [knot]
    final += [right] * (maxdeg + 1)
    return ImmutableKnotVector(final, maxdeg)


def intersect_knotvectors(
    knotvectors: Iterable[ImmutableKnotVector],
) -> ImmutableKnotVector:
    """
    Computes the intersections of the given knotvectors
    """
    knotvectors = tuple(knotvectors)
    if not all(isinstance(vec, ImmutableKnotVector) for vec in knotvectors):
        raise TypeError
    left, right = knotvectors[0].knots[0], knotvectors[0].knots[-1]
    if any(vec.knots[0] != left or vec.knots[-1] != right for vec in knotvectors):
        raise ValueError
    mindeg = min(vec.degree for vec in knotvectors)
    internals = {}
    for knotvector in knotvectors:
        if knotvector.degree > mindeg:
            knotvector = decrease_degree(knotvector, knotvector.degree - mindeg)
        for knot in knotvector.knots[1:-1]:
            if knot not in internals:
                internals[knot] = knotvector.mult(knot)
    for knotvector in knotvectors:
        if knotvector.degree > mindeg:
            knotvector = decrease_degree(knotvector, knotvector.degree - mindeg)
        for knot in internals.keys():
            internals[knot] = min(internals[knot], knotvector.mult(knot))
    final = [left] * (mindeg + 1)
    for knot in sorted(internals.keys()):

        final += internals[knot] * [knot]
    final += [right] * (mindeg + 1)
    return ImmutableKnotVector(final, mindeg)
