# -*- coding: UTF-8 -*-
import numpy as np
from numpy.linalg import LinAlgError
from .corr import calc_kernel_matrix

""" Modified by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-13.
Modifications made:
1. Use Ft = np.linalg.solve(C, F) and beta = np.linalg.solve(G, Q.T @ Yt) if you want to improve the computational efficiency.  Note that this may be harmful 
to the modeling accuracy.

pydacefit source: https://github.com/msu-coinlab/pydacefit
"""


def fit(X, Y, regr, kernel, theta):

    # attributes used for convenience
    n_sample, n_var, n_target = X.shape[0], X.shape[1], Y.shape[1]

    # calculate the kernel matrix R
    R = calc_kernel_matrix(X, X, kernel, theta)
    R += np.eye(n_sample) * (10 + n_sample) * 2.220446049250313e-16

    # do the cholesky decomposition
    try:
        C = np.linalg.cholesky(R)
    except LinAlgError:
        print(R)
        raise Exception("Error while doing Cholesky Decomposition.")

    # fit the least squares for regression
    F = regr(X)
    #Ft = np.linalg.lstsq(C, F, rcond=None)[0]
    Ft = np.linalg.solve(C, F)
    Q, G = np.linalg.qr(Ft)
    rcond = 1.0 / np.linalg.cond(G)
    if rcond > 1e15:
        raise Exception('F is too ill conditioned: Poor combination of regression model and design sites')
    Yt = np.linalg.solve(C, Y)
    #beta = np.linalg.lstsq(G, Q.T @ Yt, rcond=None)[0]
    beta = np.linalg.solve(G, Q.T @ Yt)

    # calculate the residual to fit with gaussian process and calculate objective function
    rho = Yt - Ft @ beta
    sigma2 = np.sum(np.square(rho), axis=0) / n_sample
    detR = np.prod(np.power(np.diag(C), (2 / n_sample)))
    obj = np.sum(sigma2) * detR

    # finally gamma to predict values
    gamma = np.linalg.solve(C.T, rho)

    if type(theta) is not np.ndarray:
        theta = np.array([theta])

    return {

        'kernel': kernel,
        'regr': regr,
        'theta': theta,

        'R': R,
        'C': C,
        'F': F,
        'Ft': Ft,  # C^-1 @ F
        'Q': Q,
        'G': G,
        'Yt': Yt,  # C^-1 @ Y
        'beta': beta,  # mu
        'rho': rho,  # C^-1 @ (Y - one @ mu)
        '_sigma2': sigma2,
        'obj': obj,
        'f': obj,
        'gamma': gamma  # R^-1 @ (Y - one @ mu)
    }
