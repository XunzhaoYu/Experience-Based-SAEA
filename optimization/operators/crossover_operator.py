# -*- coding: UTF-8 -*-
import numpy as np
from copy import deepcopy


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-01.
Deb, Kalyanmoy, and Ram Bhushan Agrawal. "Simulated binary crossover for continuous search space." Complex systems 9.2 (1995): 115-148.
"""
class SBX:
    def __init__(self, probability=0.5, distribution_index=5.):
        """
        :param probability: Crossover probability for each pair of individuals.
        :param distribution_index: A small distribution index allows a distant solution to be generated.
        """
        self.PROBABILITY = probability
        self.DISTRIBUTION_INDEX = distribution_index

    def execute(self, parents, upperbound, lowerbound):
        """
        :param parents: The mating population for crossover operation. Type: 2darray. Shape: (n_samples, n_vars)
        :param upperbound: The upper bound of decision variables. Type: array. Shape: (n_vars)
        :param lowerbound: The lower bound of decision variables. Type: array. Shape: (n_vars)
        :return: offspring. Type: 2darray. Shape: (n_samples, n_vars)
        """
        parents = deepcopy(parents)
        n_samples, n_vars = np.shape(parents)
        if n_samples % 2 != 0:
            print("The number of parents should be an even number.")
            return parents
        n_couples = n_samples // 2
        parent1 = parents[:n_couples, :]
        parent2 = parents[n_couples:, :]

        rand_crossover = np.random.rand(n_couples)
        do_crossover = rand_crossover < self.PROBABILITY
        n_crossover = np.count_nonzero(do_crossover)

        rand_beta = np.random.rand(n_crossover, n_vars)
        beta = np.zeros((n_crossover, n_vars))
        beta[rand_beta <= 0.5] = np.power(2 * rand_beta[rand_beta <= 0.5], 1.0 / (self.DISTRIBUTION_INDEX + 1))
        beta[rand_beta > 0.5] = np.power(1.0 / (2 * (1 - rand_beta[rand_beta > 0.5])), 1.0 / (self.DISTRIBUTION_INDEX + 1))

        sum_parent = (parent1[do_crossover] + parent2[do_crossover]) / 2.0
        diff_parent = (parent1[do_crossover] - parent2[do_crossover]) / 2.0
        parent1[do_crossover] = sum_parent + beta * diff_parent
        parent2[do_crossover] = sum_parent - beta * diff_parent
        offspring = np.append(parent1, parent2, axis=0)

        # check boundaries
        offspring = np.minimum(np.maximum(offspring, lowerbound), upperbound)
        return offspring


