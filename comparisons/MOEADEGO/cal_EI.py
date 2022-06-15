import numpy as np
from scipy.special import erf


def cdf(param):
    return 0.5 + 0.5 * erf(param / np.sqrt(2.))


def pdf(param):
    return np.sqrt(0.5 / np.pi) * np.exp(-0.5 * np.power(param, 2))


def cal_GP(mu, sigma2):
    tau2 = np.abs(np.sum(sigma2))
    tau = np.sqrt(tau2)
    alpha = (mu[0] - mu[1]) / tau
    cdf_PA = cdf(alpha)  # cumulative density probability of positive alpha
    cdf_NA = cdf(-alpha)
    pdf_PA = pdf(alpha)
    y_te = mu[0] * cdf_PA + mu[1] * cdf_NA + tau * pdf_PA
    s2_te = (mu[0] ** 2 + sigma2[0]) * cdf_PA + (mu[1] ** 2 + sigma2[1]) * cdf_NA + np.sum(mu) * pdf_PA - y_te ** 2
    return y_te, s2_te


def cal_EI(mu_hat, sigma2_hat, weight, minimum, g_min, aggregation_method='Tchebycheff'):
    n_objs = len(mu_hat)

    if aggregation_method == 'Tchebycheff' or aggregation_method == 'Augmented_Tchebycheff':
        mu = weight * (mu_hat - minimum)
        sigma2 = np.square(weight) * sigma2_hat
        g_y, g_s2 = cal_GP(mu[:2], sigma2[:2])
        if n_objs > 2:
            for i in range(2, n_objs):
                g_y, g_s2 = cal_GP(np.array([g_y, mu[i]]), np.array([g_s2, sigma2[i]]))
    elif aggregation_method == 'Weight_sum':
        g_y = np.sum(weight * mu_hat)
        g_s2 = np.sum(np.square(weight * sigma2_hat))
    else:
        print("Undefined aggregation method ...")
        g_y, g_s2 = None, None

    if g_s2 > 0.:
        g_s = np.sqrt(g_s2)
        mu_norm = g_min - g_y
        x_norm = mu_norm / g_s
        EI = mu_norm * cdf(x_norm) + g_s * pdf(x_norm)
    else:
        EI = g_min - g_y
    return EI


"""
The following methods are consistent with PlatEMO.
!!! denotes errors.
"""
def cal_GP2(mu, sigma2, weight):
    tau2 = np.sum(sigma2 * np.square(weight))  # !!! tau2 weighted twice.
    tau = np.sqrt(tau2)
    alpha = (mu[0] - mu[1]) / tau
    cdf_PA = cdf(alpha)  # cumulative density probability of positive alpha
    cdf_NA = cdf(-alpha)
    pdf_PA = pdf(alpha)
    y_te = mu[0] * cdf_PA + mu[1] * cdf_NA + tau * pdf_PA
    # s2_te: !!! replace mu with weight; !!! missing '-y_te^2'
    s2_te = (weight[0] ** 2 + sigma2[0]) * cdf_PA + (weight[1] ** 2 + sigma2[1]) * cdf_NA + np.sum(weight) * pdf_PA
    return y_te, s2_te


def cal_EI2(mu_hat, sigma2_hat, weight, minimum, g_min, aggregation_method='Tchebycheff'):
    n_objs = len(mu_hat)

    if aggregation_method == 'Tchebycheff' or aggregation_method == 'Augmented_Tchebycheff':
        mu = weight * (mu_hat - minimum)
        sigma2 = np.square(weight) * sigma2_hat
        g_y, g_s2 = cal_GP2(mu[:2], np.abs(sigma2[:2]), weight[:2])  # !!! weight is useless.
        if n_objs > 2:
            for i in range(2, n_objs):
                g_y, g_s2 = cal_GP2(np.array([g_y, mu[i]]), np.abs(np.array([g_s2, sigma2[i]])), np.array([1, weight[i]]))  # !!! weight is useless.
    elif aggregation_method == 'Weight_sum':
        g_y = np.sum(weight * mu_hat)
        g_s2 = np.sum(np.square(weight * sigma2_hat))
    else:
        print("Undefined aggregation method ...")
        g_y, g_s2 = None, None

    if g_s2 > 0.:
        g_s = np.sqrt(g_s2)
        mu_norm = g_min - g_y  # !!! suggested g_min should be a minimum of a subproblem
        x_norm = mu_norm / g_s
        EI = mu_norm * cdf(g_min - g_y/g_s) + g_s * pdf(x_norm)  # !!! wrong EI: cdf(x_norm)
    else:
        EI = g_min - g_y  # !!! suggested g_min should be a minimum of a subproblem
    return EI

