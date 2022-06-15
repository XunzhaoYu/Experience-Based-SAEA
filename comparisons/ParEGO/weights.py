import numpy as np
from itertools import product


def weight_generation(k, s):
    """
    :param k: the number of objectives.
    :param s: parameters of weight vectors, larger H produces more weight vectors.
    :return: weight vectors, type: 2dnarray, shape:(n_weights, m). Where n_weights = C_{H+m-1}^{m-1}
    """
    if k == 2:
        temp = np.array(range(s+1))
        temp2 = [1. * s] * (s+1) - temp
        weight_vectors = np.array([temp, temp2]).T / (s * 1.)
        # print('weight generation (2 obj):', np.shape(weight_vectors), type(weight_vectors))
    elif k == 3:
        weight_range = np.array((s,) * 2)
        temp12 = np.array([i for i in product(*(range(i + 1) for i in weight_range)) if sum(i) <= s])
        temp3 = np.array([[1. * s] - temp12[:, 0] - temp12[:, 1]]).T
        weight_vectors = np.append(temp12, temp3, axis=1)/(s * 1.)
        # print('weight generation (3 obj):', np.shape(weight_vectors), type(weight_vectors))
    else:
        temp = np.array((s,) * k)
        weight_vectors = np.array([i for i in product(*(range(i + 1) for i in temp)) if sum(i) == s - 1])
        weight_vectors = (weight_vectors + .5) / (s - 1 + k * .5)
        # print('num of weight vectors {:d}'.format(len(weight_vectors)))
    return weight_vectors
