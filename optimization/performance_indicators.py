# -*- coding: UTF-8 -*-
import numpy as np
from scipy import spatial


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-01.
Performance indicators for multi-objective problems / Pareto Front.
"""
# IGD
class inverted_generational_distance():
    def __init__(self, reference_front=None, p=1.0):
        self.reference_front = reference_front
        self.p = p

    def compute(self, solutions):
        distances = spatial.distance.cdist(self.reference_front, solutions)
        min_dis = np.min(distances, axis=1)
        p_sum = np.power(np.sum(np.power(min_dis, self.p)), 1./self.p)
        return p_sum/len(min_dis)

    def get_name(self):
        return 'IGD'


"""
IGD+:
H. Ishibuchi, H. Masuda, Y. Tanigaki, and Y. Nojima, “Modified distance calculation in generational distance and inverted generational distance,” 
in Proceedings of the 8th International Conference on Evolutionary Multi-criterion Optimization (EMO’15), 2015, pp. 110–125.
"""
class inverted_generational_distance_plus():
    def __init__(self, reference_front=None):
        self.reference_front = reference_front
        self.n_rf = self.reference_front.shape[0]

    def compute(self, solutions):
        n_s = solutions.shape[0]
        u = np.repeat(self.reference_front, n_s, axis=0)
        v = np.tile(solutions, (self.n_rf, 1))
        d = v - u
        d[d < 0] = 0
        D = np.sqrt((d ** 2).sum(axis=1))
        M = np.reshape(D, (self.n_rf, n_s))
        return np.mean(np.min(M, axis=1))

    def get_name(self):
        return 'IGD+'


def indicator_validation():
    pf = np.array([[0, 10],
                   [1, 6],
                   [2, 2],
                   [6, 1],
                   [10, 0]])
    igd = inverted_generational_distance(pf)
    igd_plus = inverted_generational_distance_plus(pf)

    A = np.array([[2, 4],
                  [3, 3],
                  [4, 2]])
    A_igd = igd.compute(A)
    A_igd_plus = igd_plus.compute(A)

    B = np.array([[2, 8],
                  [4, 4],
                  [8, 2]])
    B_igd = igd.compute(B)
    B_igd_plus = igd_plus.compute(B)

    print("A:", A_igd, A_igd_plus)  # 3.707, 1.483
    print("B:", B_igd, B_igd_plus)  # 2.591, 2.260


#indicator_validation()
