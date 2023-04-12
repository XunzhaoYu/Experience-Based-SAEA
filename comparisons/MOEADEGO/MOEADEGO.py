# -*- coding: UTF-8 -*-
# --- basic libraries ---
import numpy as np
from time import time
import copy
from copy import deepcopy
from pyDOE import lhs
from sklearn.cluster import KMeans
# --- surrogate modeling ---
from models.kriging.pydacefit.dace import *
from models.kriging.pydacefit.regr import *
from models.kriging.pydacefit.corr import *
# --- MOEA/D-EGO ---
from comparisons.MOEADEGO.weights import *
from comparisons.MOEADEGO.fuzzyCM import *
from comparisons.MOEADEGO.cal_EI import *
# --- optimization libraries ---
from optimization.operators.mutation_operator import *
from optimization.performance_indicators import *
# --- tools ---
from tools.recorder import *
from tools.loader import *

""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Version: 2022-June-15.
Q. Zhang, W. Liu, E. Tsang, and B. Virginas, “Expensive multiobjective optimization by MOEA/D with gaussian process model,” 
IEEE Transactions on Evolutionary Computation, vol. 14, no. 3, pp. 456–474, 2010.
"""


class MOEADEGO:
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
        self.COE_RANGE = self.config['coe_range']  # range of coefficient (theta)
        #self.EXP_RANGE = self.config['exp_range']

        # --- Fuzzy_CM ---
        self.L1 = 80
        self.L2 = 20
        self.ALPHA = 2.0
        self.EPSILON = 0.05

        # --- optimization algorithm setups ---
        self.EVALUATION_INIT = self.config['evaluation_init']
        self.EVALUATION_MAX = self.config['evaluation_max']
        self.K_E = 5  # maximum number of evaluations per iteration
        # --- --- MOEA/D-DE algorithm configuration --- ---
        self.aggregation_method = 'Tchebycheff'
        self.GENERATION_MAX = 50  # !!! G_max = 500

        self.DELTA = 0.9
        self.N_R = 2
        self.CR = 1.0
        self.F = 0.5
        # --- --- reference vectors --- ---
        self.weight_vectors = weight_generation(self.n_objs, 33) + np.ones(self.n_objs) * 1e-5
        print('weights:', np.shape(self.weight_vectors))
        self.n_weight_vectors = len(self.weight_vectors)

        self.NEIGHBOR_SIZE = 20
        self.neighbors = []
        for i in range(self.n_weight_vectors):
            weight_distance = np.sum(np.power(self.weight_vectors - self.weight_vectors[i, :], 2), axis=1).T
            self.neighbors.append(np.argsort(weight_distance)[:self.NEIGHBOR_SIZE])
        self.neighbors = np.array(self.neighbors)
        # --- --- mutation operator --- ---
        self.mutation_args = self.config['mutation_args']
        mutation_ops = {
            'polynomial': Polynomial(self.mutation_args[0], self.mutation_args[1]),
            'value_shift': ValueShift(self.mutation_args[0])
        }
        self.mutation_op = mutation_ops[self.config['mutation_op']]

        # --- variables declarations ---
        self.time = None
        self.iteration = None
        # --- --- archive and surrogate variables --- ---
        self.X = None
        self.Y = None
        self.archive_size = 0
        self.theta = None
        self.surrogates = None
        # --- --- non-dominated solution variables --- ---
        self.pf_index = None  # the indexes of pareto set solutions in the archive.
        self.ps = None  # current ps (non-dominated solutions in the decision space)
        self.pf = None  # current pf (pareto front in the objective space)
        self.new_point = None
        self.new_objs = None
        self.n_reproduction = None

        self.pf_changed = None
        self.Y_upperbound = None  # upperbound of Y.
        self.Y_lowerbound = None  # lowerbound of Y.
        self.Y_range = None  # self.Y_upperbound - self.Y_lowerbound
        # --- FuzzyCM ---
        self.c_size = None
        self.centers = None
        self.g_min = None
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
        self.c_size = int(1 + np.ceil((self.archive_size - self.L1) * 1. / self.L2))
        if self.c_size < 1:
            self.c_size = 1
        self.centers = np.zeros((self.c_size, self.n_vars))
        self.theta = np.ones((self.c_size * self.n_objs, self.n_vars))
        self.surrogates = []  # !!!
        for i in range(self.c_size * self.n_objs):
            new_surrogate = DACE(regr=regr_constant, corr=corr_gauss, theta=self.theta[i],
                                 thetaL=np.ones(self.n_vars) * self.COE_RANGE[0], thetaU=np.ones(self.n_vars) * self.COE_RANGE[1])
            self.surrogates.append(new_surrogate)

        self.Y_lowerbound = np.min(self.Y, axis=0)
        self.g_min = self._g_min_update()
        # --- pareto front variables ---
        self.pf_index = np.zeros(1, dtype=int)
        self.ps, self.pf = self._ps_init()
        print("Initialization of non-dominated solutions:", np.shape(self.ps))
        print("Initial Pareto Front:")
        print(self.pf)

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
            return X[0], Y[0]
        else:  # load pre-sampled dataset
            str_ei = str(self.EVALUATION_INIT)
            path = self.init_path + "exp2-DTLZ" + str(self.EVALUATION_MAX) + "_optimization/" + self.name + "/" + str_ei + "_" + self.name + "/" + \
                   str_ei + "_" + self.name + "(" + str(self.n_vars) + "," + str(self.n_objs) + ")_" + self.iteration + ".xlsx"
            return load_XY_for_exp2(path, self.n_vars, self.n_objs, self.EVALUATION_INIT)

    def _surrogate_reconstruct(self, X, Y, c_size):
        # reconstruct surrogates for every objective.
        old_centers = deepcopy(self.centers)
        self.centers, distance, _ = fuzzy_CM(X, c_size, self.ALPHA, self.EPSILON)
        old_theta = deepcopy(self.theta)
        self.theta = np.ones((c_size * self.n_objs, self.n_vars))
        distance_centers = spatial.distance.cdist(old_centers, self.centers)
        inherit_index = np.argmin(distance_centers, axis=0)
        print("inherit index:", inherit_index)
        for i in range(c_size):
            target_shift = i * self.n_objs
            source_shift = inherit_index[i] * self.n_objs
            self.theta[target_shift: target_shift+self.n_objs] = deepcopy(old_theta[source_shift: source_shift+self.n_objs])

        self.surrogates = []  # !!!
        for i in range(c_size * self.n_objs):
            new_surrogate = DACE(regr=regr_constant, corr=corr_gauss, theta=self.theta[i],
                                 thetaL=np.ones(self.n_vars) * self.COE_RANGE[0], thetaU=np.ones(self.n_vars) * self.COE_RANGE[1])
            self.surrogates.append(new_surrogate)

        # fit surrogates.
        dis_order = np.argsort(distance, axis=0)
        # surrogates = [c1(obj1, obj2, ..., objm), c2(obj1, obj2, ..., objm), ..., c_size(obj1, obj2, ..., objm)]
        for clu in range(c_size):
            clu_index = dis_order[: self.L1, clu]
            clu_shift = clu * self.n_objs
            for obj in range(self.n_objs):
                self.surrogates[clu_shift + obj].fit(X[clu_index, :], Y[clu_index, obj])

    def _surrogate_fit(self, X, Y, c_size):
        if c_size == 1:
            for obj in range(self.n_objs):
                self.surrogates[obj].fit(X, Y[:, obj])
                self.theta[obj] = self.surrogates[obj].model["theta"]
        else:
            self.centers, distance, _ = fuzzy_CM(X, c_size, self.ALPHA, self.EPSILON)
            dis_order = np.argsort(distance, axis=0)
            # surrogates = [c1(obj1, obj2, ..., objm), c2(obj1, obj2, ..., objm), ..., c_size(obj1, obj2, ..., objm)]
            for clu in range(c_size):
                clu_index = dis_order[: self.L1, clu]
                clu_shift = clu * self.n_objs
                for obj in range(self.n_objs):
                    self.surrogates[clu_shift + obj].fit(X[clu_index, :], Y[clu_index, obj])
                    self.theta[clu_shift + obj] = self.surrogates[clu_shift + obj].model["theta"]

    def _surrogate_predict(self, x, MSE=True):
        x = x.reshape(1, -1)
        dis_to_centers = np.sum(np.square(x - self.centers), axis=1)
        surrogate_index = np.argmin(dis_to_centers) * self.n_objs

        mu_hat = np.zeros((self.n_objs,), dtype=float)
        if MSE:
            sigma2_hat = np.zeros((self.n_objs,), dtype=float)
            for obj_index in range(self.n_objs):
                mu_hat[obj_index], sigma2_hat[obj_index] = self.surrogates[surrogate_index + obj_index].predict(x, return_mse=MSE)
            return mu_hat, sigma2_hat
        else:
            for obj_index in range(self.n_objs):
                mu_hat[obj_index] = self.surrogates[surrogate_index + obj_index].predict(x, return_mse=MSE)
            return mu_hat

    """
    Pareto Set/Front methods
    """
    def _ps_init(self):
        ps = np.array([self.X[0]])
        pf = np.array([self.Y[0]])
        for index in range(1, self.archive_size):
            ps, pf = self._ps_update(ps, pf, np.array([self.X[index]]), np.array([self.Y[index]]), index)
        return ps, pf

    def _ps_update(self, ps, pf, x, y, index):
        diff = pf - y
        diff = np.around(diff, decimals=4)
        # --- check if y is the same as a point in pf (x is not necessary to be the same as a point in ps) ---
        for i in range(len(diff)):
            if (diff[i] == 0).all():
                self.pf_index = np.append(self.pf_index, index)
                self.pf_changed = True
                return np.append(ps, x, axis=0), np.append(pf, y, axis=0)

        # exclude solutions (which are dominated by new point x) from the current PS.
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
        self._surrogate_fit(self.X, self.Y, self.c_size)
        while self.archive_size < self.EVALUATION_MAX:
            print(" --- Reproduction: searching for minimal negative EI... --- ")
            population, fitness = self._MOEAD_DE(speedup=True)
            self.new_point = self._delete_same_and_similar(population, fitness, speedup=True)
            self.new_objs = self._population_evaluation(self.new_point)
            print(" --- Evaluate on fitness function... ---")
            print("new point:", self.new_point)
            print("new point objective ", self.new_objs)

            # --- update archive X, archive_fitness Y, and surrogates ---
            self.X = np.append(self.X, self.new_point, axis=0)
            self.Y = np.append(self.Y, self.new_objs, axis=0)
            self.n_reproduction = len(self.new_point)
            self.archive_size += self.n_reproduction
            self._progress_update()

    def _MOEAD_DE(self, speedup=False):
        boundary = self.upperbound - self.lowerbound
        # Stage1: Initialization
        if speedup:  # use mu_hat (fitness) to replace xi.
            population_init = np.random.rand(self.n_weight_vectors, self.n_vars) * boundary + self.lowerbound
            xi_init = np.zeros((self.n_weight_vectors, self.n_weight_vectors), dtype=float)
            for ind_index in range(self.n_weight_vectors):
                mu_hat = self._surrogate_predict(population_init[ind_index], MSE=False)
                xi_init[ind_index] = np.max((mu_hat - self.Y_lowerbound) * self.weight_vectors, axis=1)  # mu_hat
            optimal_index = np.argmin(xi_init, axis=0)
        else:
            population_init = np.random.rand(self.n_weight_vectors, self.n_vars) * boundary + self.lowerbound
            xi_init = np.zeros((self.n_weight_vectors, self.n_weight_vectors), dtype=float)
            for ind_index in range(self.n_weight_vectors):
                mu_hat, sigma2_hat = self._surrogate_predict(population_init[ind_index])
                for pro_index in range(self.n_weight_vectors):
                    xi_init[ind_index, pro_index] = \
                        cal_EI(mu_hat, sigma2_hat, self.weight_vectors[pro_index], self.Y_lowerbound, self.g_min[pro_index], aggregation_method=self.aggregation_method)
            optimal_index = np.argmax(xi_init, axis=0)

        population = population_init[optimal_index]
        xi = np.zeros((self.n_weight_vectors,), dtype=float)
        for i in range(self.n_weight_vectors):
            xi[i] = xi_init[optimal_index[i], i]
        # Stage2: Update:
        for generation in range(self.GENERATION_MAX):
            for ind_index in range(self.n_weight_vectors):
                # 2.1 Selection of Mating/Update Range
                if np.random.rand() < self.DELTA:
                    parent_index = self.neighbors[ind_index].copy()
                else:
                    parent_index = np.array(range(self.n_weight_vectors))
                # 2.2 Reproduction
                mating_inds = np.random.choice(parent_index, 2, replace=False)
                rand_cr = (np.random.rand(self.n_vars) < self.CR)
                offspring = population[ind_index, rand_cr] + self.F * (population[mating_inds[0], rand_cr] - population[mating_inds[1], rand_cr])
                offspring = self.mutation_op.execute(np.array(offspring).reshape(1, -1), self.upperbound, self.lowerbound, unique=True)[0]
                # 2.3 Repair.
                offspring = np.minimum(np.maximum(offspring, self.lowerbound), self.upperbound)
                # 2.4 update of solutions
                c = 0
                parent_index = np.random.permutation(parent_index)
                mate_index = len(parent_index) - 1
                if speedup:  # use mu_hat (fitness) to replace xi.
                    mu_hat = self._surrogate_predict(offspring, MSE=False)
                    while mate_index > 0 and c < self.N_R:
                        offspring_fitness = np.max((mu_hat - self.Y_lowerbound) * self.weight_vectors[parent_index[mate_index]])
                        if offspring_fitness < xi[parent_index[mate_index]]:
                            population[parent_index[mate_index]] = offspring.copy()
                            xi[parent_index[mate_index]] = offspring_fitness
                            c += 1
                        mate_index -= 1
                else:
                    mu_hat, sigma2_hat = self._surrogate_predict(offspring)
                    while mate_index > 0 and c < self.N_R:
                        offspring_xi = cal_EI(mu_hat, sigma2_hat, self.weight_vectors[parent_index[mate_index]], self.Y_lowerbound, self.g_min[parent_index[mate_index]],
                                              aggregation_method=self.aggregation_method)
                        if offspring_xi > xi[parent_index[mate_index]]:
                            population[parent_index[mate_index]] = offspring.copy()
                            xi[parent_index[mate_index]] = offspring_xi
                            c += 1
                        mate_index -= 1
        return population, xi

    def _delete_same_and_similar(self, population, xi, speedup=False):
        # delete same solutions
        selected_index = []
        evaluated_points = copy.deepcopy(self.X)
        min_distances = np.zeros(self.n_weight_vectors)
        for ind_index in range(self.n_weight_vectors):
            distance = np.sum(np.square(evaluated_points - population[ind_index]), axis=1)
            min_distances[ind_index] = min(distance)
            if min_distances[ind_index] > 1e-5:
                selected_index.append(ind_index)
                evaluated_points = np.row_stack((evaluated_points, population[ind_index]))
        if len(selected_index) == 0:
            return deepcopy(population[np.argmax(min_distances)])
        print("before delete: ", np.shape(population))
        selected_pop = population[selected_index]
        selected_xi = xi[selected_index]
        print("after delete: ", np.shape(selected_pop))
        selected_weights = deepcopy(self.weight_vectors[selected_index])

        # delete similar solutions
        new_points = []
        n_evaluation = np.minimum(self.K_E, len(selected_index))
        clusters = KMeans(n_clusters=n_evaluation, random_state=0).fit(selected_weights)
        if speedup:  # use mu_hat (fitness) to replace xi.
            for clu_index in range(n_evaluation):
                cluster_pop_index = (clusters.labels_ == clu_index)
                cluster_pop = selected_pop[cluster_pop_index]
                cluster_size = len(cluster_pop)

                mu_hat = np.zeros((cluster_size, self.n_objs))
                sigma2_hat = np.zeros((cluster_size, self.n_objs))
                cluster_xi = np.zeros(cluster_size)
                for i in range(cluster_size):
                    mu_hat[i], sigma2_hat[i] = self._surrogate_predict(cluster_pop[i])
                    vector_index = np.argmin(np.max((mu_hat[i] - self.Y_lowerbound) * self.weight_vectors, axis=1))
                    cluster_xi[i] = cal_EI(mu_hat[i], sigma2_hat[i], self.weight_vectors[vector_index], self.Y_lowerbound, self.g_min[vector_index],
                                   aggregation_method=self.aggregation_method)
                new_points.append(cluster_pop[np.argmax(cluster_xi)])
        else:
            for clu_index in range(n_evaluation):
                cluster_pop_index = (clusters.labels_ == clu_index)
                cluster_pop = selected_pop[cluster_pop_index]
                cluster_xi = selected_xi[cluster_pop_index]
                new_points.append(cluster_pop[np.argmax(cluster_xi)])
        return np.array(new_points)

    def _g_min_update(self):
        g_min = np.zeros(self.n_weight_vectors)
        if self.aggregation_method == 'Tchebycheff':
            for i in range(self.n_weight_vectors):
                weighted_diff = (self.Y - self.Y_lowerbound) * np.array([self.weight_vectors[i]])
                aggregation_Y = np.max(weighted_diff, axis=1)
                g_min[i] = np.min(aggregation_Y)
        elif self.aggregation_method == 'Augmented_Tchebycheff':
            for i in range(self.n_weight_vectors):
                weighted_diff = (self.Y - self.Y_lowerbound) * np.array([self.weight_vectors[i]])
                aggregation_Y = np.max(weighted_diff, axis=1) + .05 * np.sum(weighted_diff, axis=1)
                g_min[i] = np.min(aggregation_Y)
        elif self.aggregation_method == 'Weight_sum':
            for i in range(self.n_weight_vectors):
                aggregation_Y = np.sum(self.Y * np.array([self.weight_vectors[i]]), axis=1)
                g_min[i] = np.min(aggregation_Y)
        else:
            print("Undefined aggregation method ...")
        return np.array(g_min)

    def _progress_update(self):
        # --- update surrogates ---
        c_size = int(1 + np.ceil((self.archive_size - self.L1) * 1. / self.L2))
        if c_size < 1:
            c_size = 1
        if c_size == self.c_size:
            self._surrogate_fit(self.X, self.Y, self.c_size)
        else:
            self.c_size = c_size
            self._surrogate_reconstruct(self.X, self.Y, self.c_size)

        new_lowerbound = np.min(self.new_objs, axis=0)
        self.Y_lowerbound = np.minimum(self.Y_lowerbound, new_lowerbound)
        self.g_min = self._g_min_update()
        self.pf_changed = False
        for new_index in range(self.n_reproduction):
            index = self.archive_size - self.n_reproduction + new_index
            self.ps, self.pf = self._ps_update(self.ps, self.pf, np.array([self.new_point[new_index]]), np.array([self.new_objs[new_index]]), index)
            if self.pf_changed:
                self.performance[0] = self.indicator_IGD_plus.compute(self.pf)
                self.performance[1] = self.indicator_IGD.compute(self.pf)
            self.recorder.write(index+1, self.new_point[new_index], self.new_objs[new_index], self.performance)
        print("update archive to keep all individuals non-dominated. ", np.shape(self.ps))

        # print results
        t = time() - self.time
        print("MOEA/D-EGO, Evaluation Count: {:d}.  Total time: {:.0f} mins, {:.2f} secs.".format(self.archive_size, t // 60, t % 60))
        print("Current IGD+ value: {:.4f}, IGD value: {:.4f}.".format(self.performance[0], self.performance[1]))

    def get_result(self):
        path = self.name + "-" + str(self.iteration).zfill(2) + " igd+ " + str(np.around(self.performance[0], decimals=3)) + ".xlsx"
        self.recorder.save(path)
        return self.ps, self.performance[1]



