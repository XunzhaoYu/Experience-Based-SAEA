"""
MOEA Tester for DTLZ.
"""
import yaml
import xlrd
from time import time

from data.Benchmark.DTLZ_variants import *
from comparisons.ParEGO.ParEGO import *
#from comparisons.MOEADEGO.MOEADEGO import *
#from comparisons.OREA.OREA import *
#from comparisons.MOEADEGO.MOEADEGO_EB import *

desired_width = 160
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(precision=4, suppress=True)

cfg_filename = 'configs/DTLZ-config.yml'
with open(cfg_filename,'r') as ymlfile:
    config = yaml.load(ymlfile)

#""" # for meta-learning
config['evaluation_init'] = 10
config['evaluation_max'] = 60
config['model_save_path'] = './saved_surrogate/DTLZ/'
#"""

name = 'DTLZ1'
dataset = DTLZ1(config)

# get the Pareto Front of DTLZ
src_path = "data/Benchmark/"+name+" PF "+str(3)+"d "+str(5000)+".xlsx"
pf_data = xlrd.open_workbook(src_path).sheets()[0]
n_rows = pf_data.nrows
pf = np.zeros((n_rows, 3))
for index in range(n_rows):
    pf[index] = pf_data.row_values(index)

iteration_max = 30
for iteration in range(0, iteration_max):
    time1 = time()
    current_iteration = str(iteration + 1)#.zfill(2)
    alg = ParEGO(config, name, dataset, pf, init_path="results/")
    # alg = MOEADEGO(config, name, dataset, pf)  #, init_path="results/")
    # alg = OREA(config, name, dataset, pf, init_path="results/")
    alg.run(current_iteration)
    t = time() - time1
    print('run time:', t // 60, " mins, ", t % 60, " secs.")
    solution, minimum = alg.get_result()
    print("solution: ", type(solution))
    print(solution)
    print("minimum: ", type(minimum))
    print(minimum)

