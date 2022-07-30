from compmec.nurbs.basefunctions import BaseFunction
from compmec.nurbs.knotoperations import insert_knot_basefunction, remove_knot_controlpoints
from typing import Optional
import numpy as np


def degree_elevation_basefunction(F: BaseFunction, times: Optional[int]=1):
    if times != 1:
        raise ValueError("Canont do degree elevation many times at once. Call it many times")
    U = list(F.U)
    knotvalues = []
    for ui in U:
        if ui not in knotvalues:
            knotvalues.append(ui)
    for ui in knotvalues:
        index = U.index(ui)
        U.insert(index, ui)
    newF = F.__class__(U)
    return newF

def degree_elevation_controlpoints(F: BaseFunction, P: np.ndarray, times: Optional[int]=1):
    """
    Once we have F (the current base function) and P (the current control points)
    we can get the new base function newF using the function above
    And then we use the least equares method to find Q such best fits the curve 
    """
    if times != 1:
        raise ValueError("Canont do degree elevation many times at once. Call it many times")
    newF = degree_elevation_basefunction(F)
    m = newF.n
    knots = []
    for i, ui in enumerate(newF.U):
        if ui not in knots:
            knots.append(ui)
    utest = []
    ndiv = max(3, int(2*m/(len(knots)-1)))
    for a, b in zip(knots[:-1], knots[1:]):
        utest += list(np.linspace(a, b, ndiv, endpoint=False))
    utest += [1]
    utest = np.array(utest) 
    L = newF(utest)
    M = L @ L.T
    Cu = F(utest).T @ P
    dim = Cu.shape[1]
    Q = np.zeros((m, dim))
    for i in range(dim):
        Q[:, i] = np.linalg.solve(M, L @ Cu[:, i])
    return Q

def degree_reduction_basefunction(F: BaseFunction, times: Optional[int]=1):
    if times != 1:
        raise ValueError("Canont do degree elevation many times at once. Call it many times")
    U = list(F.U)
    knotvalues = []
    for ui in U:
        if ui not in knotvalues:
            knotvalues.append(ui)
    for ui in knotvalues:
        U.remove(ui)
    newF = F.__class__(U)
    return newF

def degree_reduction_controlpoints(F: BaseFunction, P: np.ndarray, times: Optional[int]=1):
    """
    Once we have F (the current base function) and P (the current control points)
    we can get the new base function newF using the function above
    And then we use the least equares method to find Q such best fits the curve 
    """
    if times != 1:
        raise ValueError("Canont do degree elevation many times at once. Call it many times")
    newF = degree_reduction_basefunction(F)
    m = newF.n
    knots = []
    for i, ui in enumerate(newF.U):
        if ui not in knots:
            knots.append(ui)
    utest = []
    ndiv = max(3, int(2*m/(len(knots)-1)))
    for a, b in zip(knots[:-1], knots[1:]):
        utest += list(np.linspace(a, b, ndiv, endpoint=False))
    utest += [1]
    utest = np.array(utest) 
    L = newF(utest)
    M = L @ L.T
    Cu = F(utest).T @ P
    dim = Cu.shape[1]
    Q = np.zeros((m, dim))
    for i in range(dim):
        Q[:, i] = np.linalg.solve(M, L @ Cu[:, i])
    return Q
