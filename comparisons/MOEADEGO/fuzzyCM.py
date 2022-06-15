import numpy as np


def fuzzy_CM(X, c_size, alpha=2.0, epsilon=0.05):
    k, n_vars = np.shape(X)
    metric = 1. / (alpha - 1)

    # Initialize (u_{ij})^{\alpha}
    membership = np.random.rand(k, c_size)  # u_{ij}
    membership = membership / (np.sum(membership, axis=1).reshape(-1, 1))
    membership_alpha = np.power(membership, alpha)  # u_ij^{alpha}

    # prepare the loop:
    stop_criterion = 1
    # obj = np.float('inf')
    centers = np.zeros((c_size, n_vars))
    distance = np.zeros((k, c_size))  # distance between x^i and cluster center v^j
    while stop_criterion > epsilon:
        # for iter in range(100):
        # update cluster centers (Step2)
        for clu_index in range(c_size):
            centers[clu_index, :] = np.sum(membership_alpha[:, clu_index].reshape(-1, 1) * X, axis=0) / (
            np.sum(membership_alpha[:, clu_index], axis=0))  # + np.finfo(float).eps)
            distance[:, clu_index] = np.sum(np.square(X - centers[clu_index]), axis=1)
        # update membership (Step3)
        tmp = np.power(distance, -metric)
        new_membership = tmp / np.sum(tmp, axis=1).reshape(-1, 1)
        stop_criterion = np.max(np.abs(new_membership - membership))
        membership = new_membership.copy()
        membership_alpha = np.power(membership, alpha)
    return centers, distance, membership
