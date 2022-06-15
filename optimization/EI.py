# -*- coding: UTF-8 -*-
import numpy as np
from scipy.special import erf


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-01.
Expected Improvement.
"""
def cdf(param):
    return 0.5 + 0.5 * erf(param / np.sqrt(2.))


def pdf(param):
    return np.sqrt(0.5 / np.pi) * np.exp(-0.5 * np.power(param, 2))


def EI(minimum, mu, sigma):
    """
    :param minimum: Current minimum
    :param mu: Predicted mean.
    :param sigma: Predicted standard deviation.
    :return: EI: Expected improvement
    """
    norm = (minimum-mu)/sigma
    return (minimum-mu) * cdf(norm) + sigma * pdf(norm)
