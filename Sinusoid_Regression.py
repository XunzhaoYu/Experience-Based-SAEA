# --- basic libraries ---
import numpy as np
import yaml
import xlwt
# --- data ---
from data.dataset import *
from data.dataViz import *
# --- MDKL variants ---
from models.mdkl.MDKL import *
from models.mdkl.DKL import *
from models.mdkl.MDKL_NN import *
# --- comparison modeling methods ---
from models.kriging.pydacefit.dace import *
from models.kriging.pydacefit.regr import *
from models.kriging.pydacefit.corr import *
from models.kriging.Kriging_Adam import *
from models.dkt.DKT import *
from models.maml.MAML_Agent import MAMLAgent, register_flags
register_flags()
from models.alpaca.ALPaCA import *


""" Last update: 2023-03-26
Code for Sinusoid Regression Experiment:
1. set config to do experiments.
2. dataViz.py have MSE code. 
3. Kriging_Adam, DKL, MDKL_NN, and MDKL need to be reset when testing under different support set.
"""


cfg_filename = 'configs/sinusoid-config.yml'
with open(cfg_filename,'r') as ymlfile:
    config = yaml.load(ymlfile)

dataset = SinusoidDataset(config)
comparisons = ['GP', 'GP_Adam', 'DKL', 'MDKL_NN', 'MDKL', 'DKT', 'MAML', 'ALPaCA']

n_com = len(comparisons)
n_updates = config['n_update']
sample_size_list = [2, 3, 5, 10, 20, 30]
n_sample_list = len(sample_size_list)
n_vars = config['x_dim']

result_file = xlwt.Workbook()
result_sheet = result_file.add_sheet('MDKL')
write_style = xlwt.XFStyle()
write_style.num_format_str = '0.0000'
# fill the table headlines (columns).
for i in range(n_sample_list):
    section_length = 2 * n_com + 1  # (result + time_cost) * n_com + space
    result_sheet.write(0, 2+i*section_length, 'Sinusoid'+str(sample_size_list[i]))
    for j in range(n_com):
        result_sheet.write(1, 2+i*section_length+2*j, comparisons[j])

max_iteration = 1
for iteration in range(0, max_iteration):
    # fill the table headlines (rows).
    result_sheet.write(iteration + 2, 0, iteration + 1)
    print('--- --- Training with related tasks. --- ---')
    # """ Gaussian Process Regression with kernel trained by Adam Optimizer.  # see models.kriging.Kriging_Adam.py
    if 'GP_Adam' in comparisons:
        g_GP_Adam = tf.Graph()
        sess_GP_Adam = tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=g_GP_Adam)
        agent_GP_Adam = Kriging_Adam(config, sess_GP_Adam, g_GP_Adam)
        agent_GP_Adam.construct_model()
        print("--- --- GP_Adam initialized. --- ---")
    # """

    # """ Deep Kernel + Gaussian Process (no meta on both GP and DK)  # see models.mdkl.DKL.py
    if 'DKL' in comparisons:
        g_DKL = tf.Graph()
        sess_DKL = tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=g_DKL)
        agent_DKL = DKL(config, sess_DKL, g_DKL)
        agent_DKL.construct_model()
        print("--- --- DKL initialized. --- ---")
    # """

    # """ Deep Kernel + Gaussian Process (no meta on GP)  # see models.mdkl.MDKL_NN.py
    if 'MDKL_NN' in comparisons:
        g_MDKL_NN = tf.Graph()
        sess_MDKL_NN = tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=g_MDKL_NN)
        agent_MDKL_NN = MDKL_NN(config, sess_MDKL_NN, g_MDKL_NN)
        agent_MDKL_NN.construct_model()
        agent_MDKL_NN.train(dataset, n_updates)
        #agent_MDKL_NN.save('MDKL_NN')
        print("--- --- MDKL_NN initialized. --- ---")
    # """

    # """ Deep Kernel + Gaussian Process  # see models.mdkl.MDKL.py
    if 'MDKL' in comparisons:
        g_MDKL = tf.Graph()
        sess_MDKL = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False), graph=g_MDKL)
        agent_MDKL = MDKL(config, sess_MDKL, g_MDKL)
        agent_MDKL.construct_model()
        agent_MDKL.train(dataset, n_updates)
        #agent_MDKL.save('MDKL')
        print("--- --- MDKL initialized. --- ---")
    # """

    # """ DKT
    if 'DKT' in comparisons:
        g_DKT = tf.Graph()
        sess_DKT = tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=g_DKT)
        agent_DKT = DKT(config, sess_DKT, g_DKT)
        agent_DKT.construct_model()
        agent_DKT.train(dataset, n_updates)
        print("--- --- DKT initialized. --- ---")
    # """

    # """ MAML
    if 'MAML' in comparisons:
        g_MAML = tf.Graph()
        sess_MAML = tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=g_MAML)
        agent_MAML = MAMLAgent(config, sess_MAML, g_MAML, exp_string='Sinusoid')
        agent_MAML.construct_model()
        agent_MAML.train(dataset)
        # save meta models automatically, to set name of models, see the folder 'maml' or visit the Github pages of MAML project.
        print("--- --- MAML initialized. --- ---")
    # """

    # """ ALPaCA
    if 'ALPaCA' in comparisons:
        g_ALPaCA = tf.Graph()
        sess_ALPaCA = tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=g_ALPaCA)
        agent_ALPaCA = ALPaCA(config, sess_ALPaCA, g_ALPaCA)
        agent_ALPaCA.config['nn_layers'] = [128,128,32]
        agent_ALPaCA.config['activation'] = 'tanh'
        agent_ALPaCA.construct_model()
        agent_ALPaCA.train(dataset, n_updates)
        #agent_ALPaCA.save('ALPaCA')
        print("--- --- ALPaCA initialized. --- ---")
    # """

    print('--- --- Training with the target task. --- ---')
    N_test = 1
    test_horz = 30
    X_test, Y_test, freq_list_test, amp_list_test, phase_list_test = dataset.sample(N_test,test_horz, return_lists=True)
    ind = 0

    print('test function: amp: {}, phase: {}, freq: {}'.format(amp_list_test[ind], phase_list_test[ind], freq_list_test[ind]))
    result_sheet.write(iteration + 2, 1, amp_list_test[ind], write_style)
    plt.figure(figsize=(9, len(sample_size_list) * 1))
    for i, num_pts in enumerate(sample_size_list):
        print(" --- current sample size: ", num_pts, " --- ")
        X_update = X_test[ind:(ind + 1), :num_pts, :]
        Y_update = Y_test[ind:(ind + 1), :num_pts, :]

        title = None
        legend = False
        if i == 0:
            legend = True
            title = True

        current_ax = 1
        # """ Gaussian Process Regression with kernel trained.
        ax_GP = plt.subplot(len(sample_size_list), n_com, n_com * i + current_ax)
        start_time_GP = time.time()

        theta = np.ones(2*n_vars)
        agent_GP = DACE(regr=regr_constant, corr=corr_gauss2, theta=theta,
                        thetaL=np.append(np.ones(n_vars) * config['coe_range'][0], np.ones(n_vars) * config['exp_range'][0]),
                        thetaU=np.append(np.ones(n_vars) * config['coe_range'][1], np.ones(n_vars) * config['exp_range'][1]))
        agent_GP.fit(X_update[0], Y_update[0])
        NMSE_GP = gen_sin_fig_Kriging(agent_GP, X_update, Y_update, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
        time_cost_GP = time.time() - start_time_GP

        plt.setp(ax_GP.get_yticklabels(), visible=False)
        if i == 0:
            plt.title('GP')
        if i < len(sample_size_list) - 1:
            plt.setp(ax_GP.get_xticklabels(), visible=False)

        print('GP time cost: {} mins, {:.4f} secs \n'.format(time_cost_GP // 60, time_cost_GP % 60))
        result_sheet.write(iteration + 2, i * section_length + 2 * current_ax, NMSE_GP, write_style)
        result_sheet.write(iteration + 2, i * section_length + 2 * current_ax + 1, time_cost_GP, write_style)
        current_ax += 1
        # """

        # """ Gaussian Process Regression with kernel trained by Adam.  # see models.kriging.Kriging_Adam.py
        if 'GP_Adam' in comparisons:
            ax_GP_Adam = plt.subplot(len(sample_size_list), n_com, n_com * i + current_ax)
            start_time_GP_Adam = time.time()
            agent_GP_Adam.parameter_init()  # initialize GP parameters.
            agent_GP_Adam.train(X_update, Y_update)
            NMSE_GP_Adam = gen_sin_fig(agent_GP_Adam, X_update, Y_update, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
            time_cost_GP_Adam = time.time() - start_time_GP_Adam

            plt.setp(ax_GP_Adam.get_yticklabels(), visible=False)
            if i == 0:
                plt.title('GP (Adam)')
            if i < len(sample_size_list) - 1:
                plt.setp(ax_GP_Adam.get_xticklabels(), visible=False)

            print('GP_Adam time cost: {} mins, {:.4f} secs \n'.format(time_cost_GP_Adam // 60, time_cost_GP_Adam % 60))
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax, NMSE_GP_Adam, write_style)
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax + 1, time_cost_GP_Adam, write_style)
            current_ax += 1
        # """

        # """ Deep Kernel + Gaussian Process (no meta on both GP and DK)  # see models.mdkl.DKL.py
        if 'DKL' in comparisons:
            ax_DKL = plt.subplot(len(sample_size_list), n_com, n_com * i + current_ax)
            start_time_DKL = time.time()
            agent_DKL.parameter_init()  # initialize all parameters.
            agent_DKL.train(X_update, Y_update)
            NMSE_DKL = gen_sin_fig(agent_DKL, X_update, Y_update, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
            time_cost_DKL = time.time() - start_time_DKL

            plt.setp(ax_DKL.get_yticklabels(), visible=False)
            if i == 0:
                plt.title('DKL')
            if i < len(sample_size_list) - 1:
                plt.setp(ax_DKL.get_xticklabels(), visible=False)

            print('DKL time cost: {} mins, {:.4f} secs \n'.format(time_cost_DKL // 60, time_cost_DKL % 60))
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax, NMSE_DKL, write_style)
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax + 1, time_cost_DKL, write_style)
            current_ax += 1
        # """

        # """ Deep Kernel + Gaussian Process (no meta on GP)  # see models.mdkl.MDKL_NN.py
        if 'MDKL_NN' in comparisons:
            ax_MDKL_NN = plt.subplot(len(sample_size_list), n_com, n_com * i + current_ax)
            start_time_MDKL_NN = time.time()
            agent_MDKL_NN.parameter_init()  # initialize GP base parameters.
            agent_MDKL_NN.adapt(X_update, Y_update)
            NMSE_MDKL_NN = gen_sin_fig(agent_MDKL_NN, X_update, Y_update, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
            time_cost_MDKL_NN = time.time() - start_time_MDKL_NN

            plt.setp(ax_MDKL_NN.get_yticklabels(), visible=False)
            if i == 0:
                plt.title('MDKL NN')
            if i < len(sample_size_list) - 1:
                plt.setp(ax_MDKL_NN.get_xticklabels(), visible=False)

            print('MDKL_NN time cost: {} mins, {:.4f} secs \n'.format(time_cost_MDKL_NN // 60, time_cost_MDKL_NN % 60))
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax, NMSE_MDKL_NN, write_style)
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax + 1, time_cost_MDKL_NN, write_style)
            current_ax += 1
        # """

        # """ Deep Kernel + Gaussian Process  # see models.mdkl.MDKL.py
        if 'MDKL' in comparisons:
            ax_MDKL = plt.subplot(len(sample_size_list), n_com, n_com * i + current_ax)
            start_time_MDKL = time.time()
            agent_MDKL.parameter_init()  # initialize task-specific parameters.
            agent_MDKL.adapt(X_update, Y_update)
            NMSE_MDKL = gen_sin_fig(agent_MDKL, X_update, Y_update, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
            time_cost_MDKL = time.time() - start_time_MDKL

            plt.setp(ax_MDKL.get_yticklabels(), visible=False)
            if i == 0:
                plt.title('MDKL')
            if i < len(sample_size_list) - 1:
                plt.setp(ax_MDKL.get_xticklabels(), visible=False)

            print('MDKL time cost: {} mins, {:.4f} secs \n'.format(time_cost_MDKL // 60, time_cost_MDKL % 60))
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax, NMSE_MDKL, write_style)
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax + 1, time_cost_MDKL, write_style)
            current_ax += 1
        # """

        # """ DKT
        if 'DKT' in comparisons:
            ax_DKT = plt.subplot(len(sample_size_list), n_com, n_com * i + current_ax, sharey=ax_GP)
            start_time_DKT = time.time()
            NMSE_DKT = gen_sin_fig(agent_DKT, X_update, Y_update, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
            time_cost_DKT = time.time() - start_time_DKT

            plt.setp(ax_DKT.get_yticklabels(), visible=False)
            if i == 0:
                plt.title('DKT')
            if i < len(sample_size_list) - 1:
                plt.setp(ax_DKT.get_xticklabels(), visible=False)

            print('DKT time cost: {} mins, {:.4f} secs \n'.format(time_cost_DKT // 60, time_cost_DKT % 60))
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax, NMSE_DKT, write_style)
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax + 1, time_cost_DKT, write_style)
            current_ax += 1
        # """

        # """ MAML
        if 'MAML' in comparisons:
            ax_MAML = plt.subplot(len(sample_size_list), n_com, n_com * i + current_ax, sharey=ax_GP)
            start_time_MAML = time.time()
            NMSE_MAML = gen_sin_fig(agent_MAML, X_update, Y_update, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
            time_cost_MAML = time.time() - start_time_MAML

            plt.setp(ax_MAML.get_yticklabels(), visible=False)
            if i == 0:
                plt.title('MAML')
            if i < len(sample_size_list) - 1:
                plt.setp(ax_MAML.get_xticklabels(), visible=False)

            print('MAML time cost: {} mins, {:.4f} secs \n'.format(time_cost_MAML // 60, time_cost_MAML % 60))
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax, NMSE_MAML, write_style)
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax + 1, time_cost_MAML, write_style)
            current_ax += 1
        # """

        # """ ALPaCA
        if 'ALPaCA' in comparisons:
            ax_ALPaCA = plt.subplot(len(sample_size_list), n_com, n_com * i + current_ax, sharey=ax_GP)
            start_time_ALPaCA = time.time()
            NMSE_ALPaCA = gen_sin_fig(agent_ALPaCA, X_update, Y_update, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
            time_cost_ALPaCA = time.time() - start_time_ALPaCA

            plt.setp(ax_ALPaCA.get_yticklabels(), visible=False)
            if i == 0:
                plt.title('ALPaCA')
            if i < len(sample_size_list) - 1:
                plt.setp(ax_ALPaCA.get_xticklabels(), visible=False)

            print('ALPaCA time cost: {} mins, {:.4f} secs \n'.format(time_cost_ALPaCA // 60, time_cost_ALPaCA % 60))
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax, NMSE_ALPaCA, write_style)
            result_sheet.write(iteration + 2, i * section_length + 2 * current_ax + 1, time_cost_ALPaCA, write_style)
            current_ax += 1
        # """

    plt.tight_layout(w_pad=0.0,h_pad=0.2)
    plt.savefig('figure/sinusoid_'+str(n_com)+' 2,3,5,10,20,30 ('+str(iteration+1)+').pdf')
    result_file.save("Sinusoid test result.xlsx")
    #plt.show()
