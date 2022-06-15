# -*- coding: UTF-8 -*-
import numpy as np
import itertools


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-01.
K. Li, K. Deb, Q. Zhang, and S. Kwong, “An evolutionary many-objective optimization algorithm based on dominance and decomposition,” 
IEEE Transactions on Evolutionary Computation, vol. 19, no. 5, pp. 694–716, 2014.
"""
def generate_vectors(n_objs, layer=1, h=2, h2=1):
    weight_range = np.array((h,) * n_objs)
    reference_vectors = np.array([i for i in itertools.product(*(range(i + 1) for i in weight_range)) if sum(i) == h], dtype=float) / h
    if layer == 1:  # method adopted by Xunzhao
        reference_vectors = (1.0 / 3) / n_objs + 2.0 / 3 * reference_vectors
        reference_vectors = np.append(reference_vectors, np.ones((1, n_objs)) * 1.0 / n_objs, axis=0)
    elif layer == 2:  # method adopted in the publication: An Evolutionary Many-Objective Optimization Algorithm Based on Dominance and Decomposition
        weight_range = np.array((h2,) * n_objs)
        reference_vectors2 = np.array([i for i in itertools.product(*(range(i + 1) for i in weight_range)) if sum(i) == h2], dtype=float) / h2
        reference_vectors = np.append(reference_vectors, 0.5 / n_objs + 0.5 * reference_vectors2, axis=0)
    elif layer == 3:
        weight_range = np.array((h2,) * n_objs)
        reference_vectors2 = np.array([i for i in itertools.product(*(range(i + 1) for i in weight_range)) if sum(i) == h2], dtype=float) / h2
        reference_vectors = np.append(reference_vectors, 0.5 / n_objs + 0.5 * reference_vectors2, axis=0)
        reference_vectors = np.append(reference_vectors, np.ones((1, n_objs)) * 1.0 / n_objs, axis=0)
    print("reference vectors:", len(reference_vectors))
    print(reference_vectors)
    return reference_vectors
