# -*- coding: UTF-8 -*-
# --- basic libraries ---
import numpy as np
from time import time
# --- surrogate modeling ---
from models.kriging.pydacefit.dace import *
from models.kriging.pydacefit.regr import *
from models.kriging.pydacefit.corr import *
# --- OREA ---
from comparisons.OREA.reference_vector import generate_vectors
from comparisons.OREA.labeling_operator import domination_based_ordinal_values
# --- optimization libraries ---
from optimization.operators.crossover_operator import *
from optimization.operators.mutation_operator import *
from optimization.EI import *
from optimization.performance_indicators import *
# --- tools ---
from tools.recorder import *
from tools.loader import *

""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Version: 2022-June-15.
X. Yu, X. Yao, Y. Wang, L. Zhu, and D. Filev, “Domination-based ordinal regression for expensive multi-objective optimization,” 
in Proceedings of the 2019 IEEE Symposium Series on Computational Intelligence (SSCI’19), 2019, pp. 2058–2065.
"""


class OREA:
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
        self.N_LEVELS = 10  # self.config['n_levels']
        self.OVERFITTING_COE = 0.03  # self.config['overfitting_coe']
        self.DACE_TRAINING_ITERATION_INIT = 8  # self.config['dace_training_iteration_init']
        self.DACE_TRAINING_ITERATION = 4  # self.config['dace_training_iteration']
        self.COE_RANGE = [1e-5, 10.]  # self.config['coe_range']
        self.EXP_RANGE = [1., 2.]  # self.config['exp_range']

        # --- optimization algorithm setups ---
        self.EVALUATION_INIT = self.config['evaluation_init']
        self.EVALUATION_MAX = self.config['evaluation_max']
        # --- --- reproduction configuration  --- ---
        self.N_REPRODUCTION = 2
        self.SEARCH_EVALUATION_MAX = 3000  # self.config['search_evaluation_max']
        self.POP_SIZE = 100  # self.config['population_size']
        self.NEIGHBORHOOD_SIZE = 10  # self.config['neighborhood_size']
        self.N_VARIANTS = 100  # self.config['n_variants']
        # --- --- reference vectors --- ---
        self.vectors = generate_vectors(self.n_objs, layer=3, h=2, h2=1)
        self.normalized_vs = self.vectors / np.sqrt(np.sum(np.power(self.vectors, 2), axis=1)).reshape(-1, 1)
        self.n_vectors = len(self.vectors)
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

        # --- variables declarations ---
        self.time = None
        self.iteration = None
        # --- --- archive and surrogate variables --- ---
        self.X = None
        self.Y = None
        self.archive_size = 0
        self.theta = np.zeros((2*self.n_vars))  # parameters of the ordinal regression surrogates
        self.surrogate = None
        self.Y_upperbound = None  # upperbound of Y.
        # --- --- non-dominated solution variables --- ---
        self.pf_index = None  # the indexes of pareto set solutions in the archive.
        self.ps = None  # current ps (non-dominated solutions in the decision space)
        self.pf = None  # current pf (pareto front in the objective space)
        self.pf_upperbound = self.pf_lowerbound = None  # pf boundaries
        self.new_point = None
        self.new_objs = None

        self.Y_range = None  # self.Y_upperbound - self.pf_lowerbound
        self.normalized_pf = None
        self.pf_changed = self.range_changed = None  # update flags
        # --- labeling methods ---
        self.label = self.reference_point = self.rp_index_in_pf = None
        self.region_id = self.region_counter = None
        self.rp_region_set = self.non_empty_region_set = self.candidate_region_set = None
        # --- recorder ---
        self.performance = np.zeros(2)
        self.recorder = None

    """
    Initialization methods:
    set variables and surrogate for a new iteration.
    """
    def variable_init(self, current_iteration):
        """
        Initialize surrogate, reset all variables.
        """
        self.time = time()
        self.iteration = current_iteration
        # --- --- archive and surrogate variables --- ---
        self.X, self.Y = self._archive_init()
        self.archive_size = len(self.X)
        self.theta = np.append(np.ones(self.n_vars), np.ones(self.n_vars))
        self.surrogate = DACE(regr=regr_constant, corr=corr_gauss2, theta=self.theta,
                              thetaL=np.append(np.ones(self.n_vars) * self.COE_RANGE[0], np.ones(self.n_vars) * self.EXP_RANGE[0]),
                              thetaU=np.append(np.ones(self.n_vars) * self.COE_RANGE[1], np.ones(self.n_vars) * self.EXP_RANGE[1]))
        self.Y_upperbound = np.max(self.Y, axis=0)
        self.pf_lowerbound = np.min(self.Y, axis=0)
        # --- pareto front variables ---
        self.pf_index = np.zeros(1, dtype=int)
        self.ps, self.pf = self.ps_init()
        self.pf_upperbound = np.max(self.pf, axis=0)
        print("Initialization of non-dominated solutions:", np.shape(self.ps))
        print("Initial Pareto Front:")
        print(self.pf)
        self.Y_range = self.Y_upperbound - self.pf_lowerbound
        self.Y_range[self.Y_range == 0] += 0.0001  # avoid NaN caused by dividing zero.
        print("Objective range:", self.Y_range)
        self.normalized_pf = (self.pf - self.pf_lowerbound) / self.Y_range  
        # --- --- update flags --- ---
        self.pf_changed = True
        self.range_changed = True

        # --- labeling methods ---
        self.label = None  #np.zeros((1, self.archive_size))
        self.reference_point = np.zeros((1, self.n_objs))
        self.rp_index_in_pf = []  # indexes in pf.

        self.region_id = np.ones((self.archive_size,), dtype=int) * -1  # record the region id for all non-dominated solutions.
        self.region_counter = []  # count how many non-dominated solutions in each region.
        self.rp_region_set = []  # the set of regions which contain Reference Points.
        self.non_empty_region_set = []  # the set of regions with at least one non-dominated solution.
        self.candidate_region_set = []  # the set of regions with the least non-dominated solutions, should be updated once non-dominated solutions are changed.
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
        for i in range(len(diff)):
            if (diff[i] == 0).all():
                self.pf_index = np.append(self.pf_index, index)
                self.pf_changed = True
                return np.append(ps, x, axis=0), np.append(pf, y, axis=0)
        # --- update Y upperbound ---
        for obj in range(self.n_objs):
            if self.Y_upperbound[obj] < y[0][obj]:
                self.Y_upperbound[obj] = y[0][obj]
                self.range_changed = True
        # exclude solutions (which are dominated by new point x) from the current PS.
        index_newPs_in_ps = [index for index in range(len(ps)) if min(diff[index]) < 0]
        self.pf_index = self.pf_index[index_newPs_in_ps]
        new_pf = pf[index_newPs_in_ps].copy()
        new_ps = ps[index_newPs_in_ps].copy()
        # --- add new point x into the current PS, update PF ---
        if min(np.max(diff, axis=1)) > 0:
            self.pf_index = np.append(self.pf_index, index)
            self.pf_changed = True
            # update ideal objective vector (lowerbound):
            for obj in range(self.n_objs):
                if self.pf_lowerbound[obj] > y[0][obj]:
                    self.pf_lowerbound[obj] = y[0][obj]
                    self.range_changed = True
            return np.append(new_ps, x, axis=0), np.append(new_pf, y, axis=0)
        else:
            return new_ps, new_pf

    """
    Evaluation on real problem.
    """
    def _population_evaluation(self, population, is_normalized_data=False, upperbound=None, lowerbound=None):
        if is_normalized_data:
            population = population*(upperbound-lowerbound)+lowerbound
        fitnesses = self.dataset.evaluate(population)
        return np.around(fitnesses, decimals=4)

    """
    Main method
    """
    def run(self, current_iteration):
        self.variable_init(current_iteration)
        current_n_levels = self.N_LEVELS
        while self.archive_size < self.EVALUATION_MAX:
            print(" ")
            print(" --- Labeling and Training Kriging model... --- ")
            self.label = np.zeros(self.archive_size)
            last_n_levels = current_n_levels
            if len(self.pf_index) == self.archive_size:  # if all solutions are non-dominated:
                self.label = np.ones(self.archive_size)
                current_n_levels = 1
                self.rp_index_in_pf = np.arange(0, self.archive_size)
            else:
                self.label, current_n_levels, self.rp_index_in_pf = domination_based_ordinal_values(
                    self.pf_index, self.Y, self.pf_upperbound, self.pf_lowerbound, n_levels=self.N_LEVELS, overfitting_coeff=self.OVERFITTING_COE, b_print=False)
            if current_n_levels == last_n_levels:
                self.surrogate.fit(self.X, self.label, self.DACE_TRAINING_ITERATION)
            else:
                self.surrogate.fit(self.X, self.label, self.DACE_TRAINING_ITERATION_INIT)
            self.theta = self.surrogate.model["theta"]
            print("updated theta:", self.theta)

            print(" --- Reproduction: searching for minimal negative EI... --- ")
            self.new_point = np.zeros((self.N_REPRODUCTION, self.n_vars))
            self._update_reference()

            if len(self.pf_index) == 1:
                for i in range(self.N_REPRODUCTION):
                    self.new_point[i] = self._reproduce_by_one_mutation(self.X[self.pf_index[0]], times_per_gene=self.N_VARIANTS)
            else:
                self.new_point[0] = self._generation_based_reproduction()
                self.new_point[1] = self._individual_based_reproduction()

            self.new_objs = self._population_evaluation(self.new_point, True, self.upperbound, self.lowerbound)
            print(" --- Evaluate on fitness function... ---")
            print("new point:", self.new_point)
            print("new point objective ", self.new_objs)
            # --- update archive, archive_fitness, distance in model ---
            self.X = np.append(self.X, self.new_point, axis=0)
            self.Y = np.append(self.Y, self.new_objs, axis=0)
            self.archive_size += self.N_REPRODUCTION
            self._progress_update()

    def _update_reference(self):
        if self.pf_changed:
            self._get_region_ID(self.normalized_pf)
            print("region id of non-dominated solutions:", self.region_id)
            # --- record indexes of the regions which contain level0 points (reference points) ---
            self.rp_region_set = np.array(list(set(self.region_id[self.rp_index_in_pf])))
            print("region id set of reference points:", self.rp_region_set)
            # --- count how many non-dominated solutions exist in each region ---
            self.region_counter = np.zeros((self.n_vectors), dtype=int)
            for i in range(len(self.region_id)):
                self.region_counter[self.region_id[i]] += 1
            print(self.region_counter)

            # --- delete regions without non-dominated points: all region indexes -> non_empty_region_indexes ---
            n_points_order = np.argsort(self.region_counter)  # rank: min -> max
            min_n_points, min_n_index = 0, 0
            for index, rank in enumerate(n_points_order):
                if self.region_counter[rank] > 0:
                    min_n_points = self.region_counter[rank]  # the minimal number of non-dominated solutions in a non_empty_region
                    min_n_index = index
                    break
            self.non_empty_region_set = n_points_order[min_n_index:]
            # --- select regions with the least non-dominated points: non_empty_region_set -> candidate_region_set ---
            self.candidate_region_set = []
            for region_index in self.non_empty_region_set:
                if self.region_counter[region_index] > min_n_points:
                    break
                self.candidate_region_set.append(region_index)

    def _generation_based_reproduction(self):
        return self._reproduce_by_PSO()

    def _reproduce_by_PSO(self, inertia=0.5, cognitive_rate=1.5, social_rate=1.5):
        # Initialization
        pop = np.random.rand(self.POP_SIZE, self.n_vars) * (self.upperbound - self.lowerbound) + self.lowerbound
        fit = np.zeros(self.POP_SIZE)
        for i in range(self.POP_SIZE):
            fit[i] = self.cal_EI(pop[i])
        n_evaluation = self.POP_SIZE

        previous_pop = deepcopy(pop)
        previous_best = deepcopy(pop)  # archive
        previous_best_fit = deepcopy(fit)
        neighbors = np.zeros((self.POP_SIZE, self.NEIGHBORHOOD_SIZE))
        half_hood = self.NEIGHBORHOOD_SIZE // 2
        for i in range(self.POP_SIZE):
            neighbors[i] = (np.array(range(self.NEIGHBORHOOD_SIZE)) - half_hood + self.POP_SIZE + i) % self.POP_SIZE
        neighbors = neighbors.astype(int)

        # Optimization:
        next_pop = np.zeros((self.POP_SIZE, self.n_vars))
        next_fit = np.zeros(self.POP_SIZE)
        while n_evaluation < self.SEARCH_EVALUATION_MAX:
            # Reproduction
            for i in range(self.POP_SIZE):
                neighbor_indexes = neighbors[i]
                neighbor_best_index = neighbor_indexes[np.argmin(previous_best_fit[neighbor_indexes])]
                neighbor_best = previous_best[neighbor_best_index]
                next_pop[i] = pop[i] + \
                              inertia * (pop[i] - previous_pop[i]) + \
                              cognitive_rate * np.random.rand(self.n_vars) * (previous_best[i] - pop[i]) + \
                              social_rate * np.random.rand(self.n_vars) * (neighbor_best - pop[i])
            next_pop = np.minimum(np.maximum(next_pop, self.lowerbound), self.upperbound)
            for i in range(self.POP_SIZE):
                next_fit[i] = self.cal_EI(next_pop[i])
            n_evaluation += self.POP_SIZE
            # Environmental selection
            previous_pop = deepcopy(pop)
            for i in range(self.POP_SIZE):
                if next_fit[i] < previous_best_fit[i]:
                    previous_best[i] = next_pop[i].copy()
                    previous_best_fit[i] = next_fit[i]
            pop = deepcopy(next_pop)
            fit = next_fit.copy()
        order = np.argsort(fit)
        pop = pop[order]
        return pop[0]

    def _individual_based_reproduction(self):
        print(" --- --- IndReproduction: mating 1: --- --- ")
        # --- randomly pick a candidate region as the target region in this iteration ---
        target_region = np.random.choice(self.candidate_region_set, 1)[0]
        print("target region:", target_region, "from candidate region set", self.candidate_region_set, ". reference vector:", self.vectors[target_region])
        # --- select non-dominated solution(s) with maximal label value in the target region ---
        target_pf_index_in_pf = [s for s in range(len(self.region_id)) if self.region_id[s] == target_region]
        target_pf_index = self.pf_index[target_pf_index_in_pf]  # non-dominated solution indexes in the target region: index in archive
        print("indexes of non-dominated solutions in the target region:", target_pf_index)
        max_value = np.max(self.label[target_pf_index])
        candidate_indexes = [ind for ind in target_pf_index_in_pf if self.label[self.pf_index[ind]] == max_value]  # index in pf_index
        if len(candidate_indexes) == 1:
            mating1_index = self.pf_index[candidate_indexes[0]]
        else:  # --- select based on crowd ---
            candidate_distance = spatial.distance.cdist(self.normalized_pf[candidate_indexes], self.normalized_pf[candidate_indexes])
            candidate_distance += np.eye(len(candidate_indexes)) * 2 * np.amax(candidate_distance)
            mating1_index = self.pf_index[candidate_indexes[np.argmax(np.min(candidate_distance, axis=1), axis=0)]]

        mating_population = np.zeros((2, self.n_vars))
        mating_population[0] = self.X[mating1_index]
        print("mating 1:", mating1_index, mating_population[0], self.Y[mating1_index])

        print(" --- --- IndReproduction: mating 2: --- --- ")
        random_region = target_region
        random_candidate_indexes = []
        # --- if all reference points are located in the target region ---
        if len(self.rp_region_set) == 1 and self.rp_region_set[0] == target_region:
            random_region = np.random.choice(self.candidate_region_set, 1)[0]
            for i, id in enumerate(self.region_id):
                if id == random_region:
                    random_candidate_indexes.append(self.pf_index[i])
        else:  # --- select a random region (not the target one) ---
            while random_region == target_region:
                random_region = np.random.choice(self.rp_region_set, 1)[0]
            for i, id in enumerate(self.region_id[self.rp_index_in_pf]):
                if id == random_region:
                    random_candidate_indexes.append(self.pf_index[self.rp_index_in_pf[i]])
        print("random region:", random_region, "RP indexes in random region", random_candidate_indexes)
        mating2_index = np.random.choice(random_candidate_indexes, 1)[0]
        mating_population[1] = self.X[mating2_index]
        print("mating 2:", mating2_index, mating_population[1], self.Y[mating2_index])

        local_origin = self.crossover_op.execute(mating_population, self.upperbound, self.lowerbound)
        local_origin = local_origin[0] if np.random.rand() < 0.5 else local_origin[1]
        return self._reproduce_by_one_mutation(local_origin, times_per_gene=self.N_VARIANTS)

    def _reproduce_by_one_mutation(self, origin, times_per_gene=100):
        neg_ei = np.zeros((self.n_vars * times_per_gene))
        new_point = np.tile(origin.copy(), (self.n_vars * times_per_gene, 1))

        mutant = self.mutation_op.execute(new_point, self.upperbound, self.lowerbound, unique=True)
        for i in range(self.n_vars * times_per_gene):
            neg_ei[i] = self.cal_EI(mutant[i])
        return mutant[np.argmin(neg_ei)].copy()

    def cal_EI(self, x):  # minimize negative EI equivalent to maximize EI.
        x = np.array(x).reshape(1, -1)
        mu_hat, sigma2_hat = self.surrogate.predict(x, return_mse=True)
        if sigma2_hat <= 0.:
            ei = mu_hat - 1.0
        else:  # cdf(z) = 1/2[1 + erf(z/sqrt(2))].
            ei = EI(minimum=-1.0, mu=-mu_hat, sigma=np.sqrt(sigma2_hat))
        return -ei

    def _get_region_ID(self, region_points, incremental=False):
        projection_length = self.normalized_vs.dot(region_points.T)  # n_vectors * archive_size
        region_id = np.ones((len(region_points)), dtype=int) * -1
        for i in range(len(region_points)):
            region_distance_vector = (region_points[i].reshape(1, -1) - projection_length[:, i].reshape(-1, 1) * self.normalized_vs)
            region_distance = np.sum(np.power(region_distance_vector, 2), axis=1)
            region_id[i] = np.argmin(region_distance)
        if incremental:
            self.region_id = np.append(self.region_id, region_id, axis=0)
        else:
            self.region_id = region_id

    def _progress_update(self):
        self.pf_changed, self.range_changed = False, False
        for new_index in range(self.N_REPRODUCTION):
            index = self.archive_size - self.N_REPRODUCTION + new_index
            self.ps, self.pf = self.get_ps(self.ps, self.pf, np.array([self.new_point[new_index]]), np.array([self.new_objs[new_index]]), index)
            if self.pf_changed:
                self.performance[0] = self.indicator_IGD_plus.compute(self.pf)
                self.performance[1] = self.indicator_IGD.compute(self.pf)
            self.recorder.write(index+1, self.new_point[new_index], self.new_objs[new_index], self.performance)
        print("update archive to keep all individuals non-dominated. ", np.shape(self.ps))

        # update three bounds for pf, and also update normalized pf
        if self.range_changed:
            self.Y_range = self.Y_upperbound-self.pf_lowerbound
            self.Y_range[self.Y_range == 0] =+ 0.0001

        if self.pf_changed:
            print("current pf_index", self.pf_index)
            print("current pf", self.pf)
            self.pf_upperbound = np.max(self.pf, axis=0)
            print("current pf upper bound:", self.pf_upperbound)

        if self.range_changed or self.pf_changed:
            self.normalized_pf = (self.pf-self.pf_lowerbound)/self.Y_range

        # print results
        t = time() - self.time
        print("OREA, Evaluation Count: {:d}.  Total time: {:.0f} mins, {:.2f} secs.".format(self.archive_size, t // 60, t % 60))
        print("Current IGD+ value: {:.4f}, IGD value: {:.4f}.".format(self.performance[0], self.performance[1]))

    def get_result(self):
        path = self.name + "-" + str(self.iteration).zfill(2) + " igd+ " + str(np.around(self.performance[0], decimals=3)) + ".xlsx"
        self.recorder.save(path)
        return self.ps, self.performance[1]

