import numpy as np
from itertools import product


def weight_generation(m, H):
    """
    :param m: the number of objectives.
    :param H: parameters of weight vectors, larger H produces more weight vectors.
    :return: weight vectors, type: 2dnarray, shape:(n_weights, m). Where n_weights = C_{H+m-1}^{m-1}
    """
    if m == 2:
        temp = np.array(range(H + 1))
        temp2 = [1. * H] * (H + 1) - temp
        weight_vectors = np.array([temp, temp2]).T / (H * 1.)
        # print('weight generation (2 obj):', np.shape(weight_vectors), type(weight_vectors))
    elif m == 3:
        weight_range = np.array((H,) * 2)
        temp12 = np.array([i for i in product(*(range(i + 1) for i in weight_range)) if sum(i) <= H])
        temp3 = np.array([[1. * H] - temp12[:, 0] - temp12[:, 1]]).T
        weight_vectors = np.append(temp12, temp3, axis=1) / (H * 1.)
        # print('weight generation (3 obj):', np.shape(weight_vectors), type(weight_vectors))
    else:
        temp = np.array((H,) * m)
        weight_vectors = np.array([i for i in product(*(range(i + 1) for i in temp)) if sum(i) == H - 1])
        weight_vectors = (weight_vectors + .5) / (H - 1 + m * .5)
        # print('num of weight vectors {:d}'.format(len(weight_vectors)))
    return weight_vectors
