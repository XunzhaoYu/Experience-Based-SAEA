# -*- coding: UTF-8 -*-
# --- basic libraries ---
import numpy as np
from time import time
from copy import deepcopy
from pyDOE import lhs
# --- surrogate modeling ---
from models.kriging.pydacefit.dace import *
from models.kriging.pydacefit.regr import *
from models.kriging.pydacefit.corr import *
# --- ParEGO ---
from comparisons.ParEGO.weights import *
# --- optimization libraries ---
from optimization.operators.crossover_operator import *
from optimization.operators.mutation_operator import *
from optimization.EI import *
from optimization.performance_indicators import *
# --- tools ---
from tools.recorder import *
from tools.loader import *

""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-June-15.
J. Knowles, “ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems,” 
IEEE Transactions on Evolutionary Computation, vol. 10, no. 1, pp. 50–66, 2006.
"""


class ParEGO:
    def __init__(self, config, name, dataset, pf, init_path=None):
        self.config = deepcopy(config)
        self.init_path = init_path
        # --- problem setups ---
        self.name = name
        self.n_vars = self.config['x_dim']
        self.n_objs = self.config['y_dim']
        self.upperbound = np.array(self.config['x_upperbound'])
        self.lowerbound = np.array(self.config['x_lowerbound'])
        self.dataset = dataset
        self.true_pf = pf
        self.indicator_IGD_plus = inverted_generational_distance_plus(reference_front=self.true_pf)
        self.indicator_IGD = inverted_generational_distance(reference_front=self.true_pf)

        # --- surrogate setups ---
        self.COE_RANGE = [1e-5, 100.]  # self.config['coe_range']
        #self.EXP_RANGE = self.config['exp_range']
        self.TRAINING_MAX = 11 * self.n_vars - 1 + 25

        # --- optimization algorithm setups ---
        self.EVALUATION_INIT = self.config['evaluation_init']
        self.EVALUATION_MAX = self.config['evaluation_max']
        # --- --- EA algorithm configuration --- ---
        self.POP_SIZE = 20  # self.config['population_size']
        self.SEARCH_EVALUATION_MAX = 10000  # self.config['search_evaluation_max']
        self.method = 'augmented_Tchebycheff'
        # --- --- reference vectors --- ---
        self.weight_vectors = weight_generation(self.n_objs, 4)
        self.n_weight_vectors = len(self.weight_vectors)
        # --- --- crossover operator --- ---
        self.crossover_args = self.config['crossover_args']
        crossover_ops = {
            'SBX': SBX(self.crossover_args[0], self.crossover_args[1])
        }
        self.crossover_op = crossover_ops[self.config['crossover_op']]
        # --- --- mutation operator --- ---
        self.mutation_args = self.config['mutation_args']
        mutation_ops = {
            'polynomial': Polynomial(self.mutation_args[0], self.mutation_args[1]),
            'value_shift': ValueShift(self.mutation_args[0])
        }
        self.mutation_op = mutation_ops[self.config['mutation_op']]

        """
        self.selection_args = self.config['selection_args']
        selection_ops = {
            'TournamentPop': TournamentPop(self.selection_args[0]),
            'Tournament': Tournament(self.selection_args[0], self.selection_args[1])
        }
        self.selection_op = selection_ops[self.config['selection_op']]
        """

        # --- variables declarations ---
        self.time = None
        self.iteration = None
        # --- --- archive and surrogate variables --- ---
        self.X = None
        self.Y = None
        self.archive_size = 0
        self.lambda_index = None
        self.theta = None  # parameters of the ordinal regression surrogates
        self.surrogates = None
        # --- --- non-dominated solution variables --- ---
        self.pf_index = None  # the indexes of pareto set solutions in the archive.
        self.ps = None  # current ps (non-dominated solutions in the decision space)
        self.pf = None  # current pf (pareto front in the objective space)
        self.new_point = None
        self.new_objs = None

        self.pf_changed = None
        self.Y_upperbound = None  # upperbound of Y.
        self.Y_lowerbound = None  # lowerbound of Y.
        self.Y_range = None  # self.Y_upperbound - self.Y_lowerbound
        # --- recorder ---
        self.performance = np.zeros(2)
        self.recorder = None

    """
    Initialization methods:
    set variables and surrogate for a new iteration.
    """
    def variable_init(self, current_iteration):
        """
        initialize surrogate, reset all variables.
        """
        self.time = time()
        self.iteration = current_iteration
        # --- --- archive and surrogate variables --- ---
        self.X, self.Y = self._archive_init()
        self.archive_size = len(self.X)
        self.lambda_index = 0
        self.theta = np.ones((self.n_weight_vectors, self.n_vars))  # * np.mean(self.COE_RANGE)
        self.surrogates = []
        for i in range(self.n_weight_vectors):
            new_surrogate = DACE(regr=regr_constant, corr=corr_gauss, theta=self.theta[i],
                                thetaL=np.ones(self.n_vars) * self.COE_RANGE[0], thetaU=np.ones(self.n_vars) * self.COE_RANGE[1])
            self.surrogates.append(new_surrogate)
        self.Y_upperbound = np.max(self.Y, axis=0)
        self.Y_lowerbound = np.min(self.Y, axis=0)
        # --- pareto front variables ---
        self.pf_index = np.zeros(1, dtype=int)
        self.ps, self.pf = self.ps_init()
        print("Initialization of non-dominated solutions:", np.shape(self.ps))
        print("Initial Pareto Front:")
        print(self.pf)
        self.Y_range = self.Y_upperbound - self.Y_lowerbound
        self.Y_range[self.Y_range == 0] += 0.0001  # avoid NaN caused by dividing zero.
        print("Objective range:", self.Y_range)

        # --- recorder ---
        self.performance[0] = self.indicator_IGD_plus.compute(self.pf)
        self.performance[1] = self.indicator_IGD.compute(self.pf)
        print("Initial IGD+ value: {:.4f}, IGD value: {:.4f}.".format(self.performance[0], self.performance[1]))
        self.recorder = Recorder(self.name)
        self.recorder.init(self.X, self.Y, self.performance, ['IGD+', 'IGD'])
        if self.init_path is None:
            path = self.name + "-" + str(self.iteration).zfill(2) + " initial.xlsx"
            self.recorder.save(path)

    # Invoked by self.variable_init()
    def _archive_init(self):
        """
        Modify this method to initialize your 'self.X'.
        :return X: initial samples. Type: 2darray. Shape: (self.EVALUATION_INIT, self.n_vars)
        :return Y: initial fitness. Type: 2darray. Shape: (self.EVALUATION_INIT, self.n_objs)
        """
        if self.init_path is None:
            X, Y = self.dataset.sample(n_funcs=1, n_samples=self.EVALUATION_INIT, b_variant=False)
            return X, Y
        else:  # load pre-sampled dataset
            """
            str_ei = str(self.EVALUATION_INIT)
            path = self.init_path + "exp2-DTLZ" + str(self.EVALUATION_MAX) + "_optimization/" + self.name + "/" + str_ei + "_" + self.name + "/" + \
                   str_ei + "_" + self.name + "(" + str(self.n_vars) + "," + str(self.n_objs) + ")_" + self.iteration + ".xlsx"
            return load_XY_for_exp2(path, self.n_vars, self.n_objs, self.EVALUATION_INIT) 
            """
            str_ei = str(self.EVALUATION_INIT)
            path = self.init_path + "exp2-plus/" + self.name + "/" + str_ei + "_" + self.name + "/" + \
                   str_ei + "_" + self.name + "(" + str(self.n_vars) + "," + str(self.n_objs) + ")_" + self.iteration + ".xlsx"
            return load_XY(path, self.n_vars, self.n_objs, self.EVALUATION_INIT)
            #"""

    """
    Pareto Set/Front methods
    """
    def ps_init(self):
        ps = np.array([self.X[0]])
        pf = np.array([self.Y[0]])
        for index in range(1, self.archive_size):
            ps, pf = self.get_ps(ps, pf, np.array([self.X[index]]), np.array([self.Y[index]]), index)
        return ps, pf

    def get_ps(self, ps, pf, x, y, index):
        diff = pf - y
        diff = np.around(diff, decimals=4)
        # --- check if y is the same as a point in pf (x is not necessary to be the same as a point in ps) ---
        # --- 检查新的点是否在pf上的一点相同 (obj space上相同不代表decision space上也相同) ---
        for i in range(len(diff)):
            if (diff[i] == 0).all():
                self.pf_index = np.append(self.pf_index, index)
                self.pf_changed = True
                return np.append(ps, x, axis=0), np.append(pf, y, axis=0)
        # exclude solutions (which are dominated by new point x) from the current PS. # *** move to if condition below? only new ps point can exclude older ones.
        index_newPs_in_ps = [index for index in range(len(ps)) if min(diff[index]) < 0]
        self.pf_index = self.pf_index[index_newPs_in_ps]  # self.pf_index[indexes]
        new_pf = pf[index_newPs_in_ps].copy()
        new_ps = ps[index_newPs_in_ps].copy()
        # add new point x into the current PS, update PF.
        if min(np.max(diff, axis=1)) > 0:
            self.pf_index = np.append(self.pf_index, index)
            self.pf_changed = True
            return np.append(new_ps, x, axis=0), np.append(new_pf, y, axis=0)
        else:
            return new_ps, new_pf

    """
    Evaluation on real problem.
    """
    def _population_evaluation(self, population, is_normalized_data=False, upperbound=None, lowerbound=None):
        if is_normalized_data:
            population = population * (upperbound - lowerbound) + lowerbound
        fitnesses = self.dataset.evaluate(population)
        return np.around(fitnesses, decimals=4)

    """
    Main method
    """
    def run(self, current_iteration):
        self.variable_init(current_iteration)
        print(np.shape(self.weight_vectors))
        while self.archive_size < self.EVALUATION_MAX:
            self.lambda_index = np.random.choice(self.n_weight_vectors, 1)[0]
            weight = self.weight_vectors[self.lambda_index]
            print(' ')
            print(self.lambda_index, ' weight:', weight)
            # --- train surrogate ---
            norm_Y = (self.Y - self.Y_lowerbound) / self.Y_range
            weighted_norm_Y = norm_Y * weight
            scalar_costs = np.max(weighted_norm_Y, axis=1) + .05 * np.sum(weighted_norm_Y, axis=1)
            order = np.argsort(scalar_costs)

            training_X, training_Y = self._generate_training_dataset(order, scalar_costs)
            self.surrogates[self.lambda_index].fit(training_X, training_Y)  # , self.dace_training_iteration)
            self.theta[self.lambda_index] = self.surrogates[self.lambda_index].model["theta"]
            print("updated theta:", self.theta[self.lambda_index])

            print(" --- Reproduction: searching for minimal negative EI... --- ")
            self.new_point = self.EA(order, scalar_costs[order[0]])
            self.new_objs = self._population_evaluation(self.new_point)
            print(" --- Evaluate on fitness function... ---")
            print("new point:", self.new_point)
            print("new point objective ", self.new_objs)

            # --- update archive, archive_fitness, distance in model ---
            self.X = np.append(self.X, self.new_point, axis=0)
            self.Y = np.append(self.Y, self.new_objs, axis=0)
            self.archive_size += 1
            self._progress_update()


    def _generate_training_dataset(self, order, costs):
        """
        Generate a dataset for surrogate training.
        :param order: The order of solutions in the archive, sorted by Tchebycheff weighted sum. Shape: (self.archive_size)
        :param costs: The Tchebycheff weighted sum of the archive X. Shape: (self.archive_size)
        :return: A dataset for surrogate training.
        """
        if self.archive_size <= self.TRAINING_MAX:
            training_X = self.X
            training_Y = costs
        else:
            optimal_size = self.TRAINING_MAX // 2
            training_index_opt = order[:optimal_size]
            training_index_rand = np.random.choice(order[optimal_size:], self.TRAINING_MAX - optimal_size, replace=False)
            training_index = np.append(training_index_opt, training_index_rand, axis=0)
            training_X = self.X[training_index]
            training_Y = costs[training_index]
        return training_X, training_Y

    def _population_initialization(self, order):
        """
        Initialize population. select optimal solutions using Tchebycheff weighted sum.
        :param order: The order of solutions in the archive, sorted by Tchebycheff weighted sum. Shape: (self.archive_size)
        :return: An initial population for EA. Shape: (self.POP_SIZE, self.n_vars)
        """
        optimal_size = self.POP_SIZE // 4
        population = deepcopy(self.X[order[:optimal_size], :])
        population = self.mutation_op.execute(population, self.upperbound, self.lowerbound, unique=True)

        rest_population = lhs(self.n_vars, self.POP_SIZE - optimal_size)
        population = np.append(population, rest_population, axis=0)
        return population

    def _population_EI(self, population, best_fit):
        """
        Compute EI values for an EA population.
        :param population: The population to be evaluated. Shape: (self.POP_SIZE, self.n_vars)
        :param best_fit: minimal scalar cost, used for computing EI. Shape: ()
        :return: The negative EI values of given population. Shape: (self.POP_SIZE)
        """
        mu, sigma2 = self.surrogates[self.lambda_index].predict(population, return_mse=True)
        ei = EI(minimum=best_fit, mu=mu, sigma=np.sqrt(sigma2))
        return -ei.reshape(-1)

    def EA(self, X_order, best_fit):
        """
        optimization based on EI.
        :param X_order: The order of solutions in the archive, sorted by Tchebycheff weighted sum. Shape: (self.archive_size)
        :param best_fit: minimal scalar cost, used for computing EI. Shape: ()
        :return: Optimal candidate solution. Shape: (self.n_vars)
        """
        population = self._population_initialization(X_order)  # population for ParEGO, initialized by self.population
        fitness = self._population_EI(population, best_fit)  # expected improvements equal to zero
        evaluation_counter = self.POP_SIZE

        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]
        while evaluation_counter < self.SEARCH_EVALUATION_MAX:  # steady-state EA
            # reproduction
            mating_population = np.zeros((2, self.n_vars))
            mating1 = np.min(np.random.choice(self.POP_SIZE, 2))
            mating2 = np.min(np.random.choice(self.POP_SIZE, 2))
            mating_population[0] = population[mating1].copy()
            mating_population[1] = population[mating2].copy()

            mating_population = self.crossover_op.execute(mating_population, self.upperbound, self.lowerbound)
            mating_population = mating_population[0] if np.random.rand() < 0.5 else mating_population[1]
            offspring = self.mutation_op.execute(mating_population.reshape(1, -1), self.upperbound, self.lowerbound, unique=True)

            offspring_fitness = self._population_EI(offspring, best_fit)
            evaluation_counter += 1

            # environmental selection
            if offspring_fitness < fitness[0]:
                population = np.append(offspring, population[:-1], axis=0)
                fitness = np.append(offspring_fitness, fitness[:-1], axis=0)
        return np.maximum(np.minimum(self.upperbound, population[0]), self.lowerbound).reshape(1, -1)

    def _progress_update(self):
        self.Y_lowerbound = np.minimum(self.Y_lowerbound, self.new_objs)
        self.Y_upperbound = np.maximum(self.Y_lowerbound, self.new_objs)
        self.Y_range = self.Y_upperbound - self.Y_lowerbound
        self.Y_range[self.Y_range == 0] = + 0.0001

        # after used to initialize the kriging model, the archive then used to save Pareto optimal solutions
        self.pf_changed = False
        self.ps, self.pf = self.get_ps(self.ps, self.pf, self.new_point, self.new_objs, self.archive_size - 1)
        if self.pf_changed:
            self.performance[0] = self.indicator_IGD_plus.compute(self.pf)
            self.performance[1] = self.indicator_IGD.compute(self.pf)
        self.recorder.write(self.archive_size, self.new_point[0], self.new_objs[0], self.performance)

        # print results
        t = time() - self.time
        print("ParEGO, Evaluation Count: {:d}.  Total time: {:.0f} mins, {:.2f} secs.".format(self.archive_size, t // 60, t % 60))
        print("Current IGD+ value: {:.4f}, IGD value: {:.4f}.".format(self.performance[0], self.performance[1]))

    def get_result(self):
        path = self.name + "-" + str(self.iteration).zfill(2) + " igd+ " + str(np.around(self.performance[0], decimals=3)) + ".xlsx"
        self.recorder.save(path)
        return self.ps, self.performance[1]



