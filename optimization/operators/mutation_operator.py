# -*- coding: UTF-8 -*-
import numpy as np
from copy import deepcopy


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-01.
Deb, Kalyanmoy, and Mayank Goyal. "A combined genetic adaptive search (GeneAS) for engineering design." Computer Science and informatics 26 (1996): 30-45.
"""
class Polynomial:
    def __init__(self, probability=0.0, distribution_index=20.):
        """
        :param probability: Mutation probability for each variable.
        :param distribution_index: A small distribution index allows a distant solution to be generated.
        """
        self.PROBABILITY = probability
        self.DISTRIBUTION_INDEX = distribution_index

    def execute(self, population, upperbound, lowerbound, unique=False):
        """
        :param population: The population for mutation operation. Type: 2darray. Shape: (n_samples, n_vars)
        :param upperbound: The upper bound of decision variables. Type: array. Shape: (n_vars)
        :param lowerbound: The lower bound of decision variables. Type: array. Shape: (n_vars)
        :param unique: The flag to ensure mutation operation happens. Type: bool.
        :return: offspring. Type: 2darray. Shape: (n_samples, n_vars)
        # """
        n_samples, n_vars = np.shape(population)
        if self.PROBABILITY == 0.0:
            self.PROBABILITY = 1.0 / n_vars
        boundary = np.tile((upperbound - lowerbound), (n_samples, 1))
        upperbound = np.tile(upperbound, (n_samples, 1))
        lowerbound = np.tile(lowerbound, (n_samples, 1))
        offspring = deepcopy(population)

        rand_mutation = np.random.rand(n_samples, n_vars)
        rand_beta = np.random.rand(n_samples, n_vars)
        do_mutation = rand_mutation < self.PROBABILITY
        while unique & (do_mutation == False).all():
            rand_mutation = np.random.rand(n_samples, n_vars)
            do_mutation = rand_mutation < self.PROBABILITY
        do_mutation1 = do_mutation & (rand_beta <= 0.5)
        do_mutation2 = do_mutation & (rand_beta > 0.5)

        beta = np.zeros((n_samples, n_vars))
        temp1 = (offspring[do_mutation1] - lowerbound[do_mutation1]) / boundary[do_mutation1]
        temp2 = (upperbound[do_mutation2] - offspring[do_mutation2]) / boundary[do_mutation2]
        beta[do_mutation1] = np.power(((2*rand_beta[do_mutation1]) + (1-2*rand_beta[do_mutation1])*np.power((1-temp1), self.DISTRIBUTION_INDEX+1)), 1.0/(self.DISTRIBUTION_INDEX+1))-1
        beta[do_mutation2] = 1-np.power((2*(1-rand_beta[do_mutation2]) + 2*(rand_beta[do_mutation2]-0.5)*np.power((1-temp2), self.DISTRIBUTION_INDEX+1)), 1.0/(self.DISTRIBUTION_INDEX+1))
        offspring[do_mutation] += boundary[do_mutation] * beta[do_mutation]
        # No need for boundary check.
        return offspring


class ValueShift:
    def __init__(self, SHIFT_LEVEL=1.0/100):
        self.SHIFT_LEVEL = SHIFT_LEVEL

    def execute(self, individual, upperbound, lowerbound, unique=False):
        """
        :param population: The population for mutation operation. Type: 2darray. Shape: (n_samples, n_vars)
        :param upperbound: The upper bound of decision variables. Type: array. Shape: (n_vars)
        :param lowerbound: The lower bound of decision variables. Type: array. Shape: (n_vars)
        :param unique: The flag to ensure mutation operation happens. Type: bool.
        :return: offspring. Type: 2darray. Shape: (n_samples, n_vars)
        # """

        n = len(individual)
        self.PROBABILITY = 1.0 / n
        unchanged_flag = True

        while unchanged_flag:
            new_individual = []
            for i in range(0, n):
                rand = np.random.rand()
                if rand < self.PROBABILITY:
                    sign = np.sign(np.random.rand()-0.5)
                    mu = np.random.rand()
                    if mu < 0.0001:
                        mu = 0.0001
                    boundary = upperbound[i] - lowerbound[i]
                    gene = individual[i] + sign * self.SHIFT_LEVEL * mu * boundary
                    if gene > upperbound[i]:
                        gene = upperbound[i]
                    elif gene < lowerbound[i]:
                        gene = lowerbound[i]
                    new_individual.append(gene)
                    if gene != individual[i]:
                        unchanged_flag = False
                else:
                    new_individual.append(individual[i])
            unchanged_flag = unchanged_flag and unique
        return new_individual

