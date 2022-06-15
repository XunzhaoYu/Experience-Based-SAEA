# -*- coding: UTF-8 -*-
import numpy as np
from time import time
import tensorflow as tf
from copy import deepcopy
from scipy import spatial

np.set_printoptions(linewidth=200)

class MDKL:
    def __init__(self, config, sess, graph=None, preprocess=None, f_nom=None):
        # constants.
        self.config = deepcopy(config)
        self.x_dim = config['x_dim']  # 1
        self.y_dim = config['y_dim']  # 1
        self.phi_dim = config['nn_layers'][-1]  # 32
        self.batch_dim = config['meta_batch_size']  # B
        #self.n_support = config['data_horizon'] + config['test_horizon']
        self.lr = config['lr']  # 0.001
        self.sigma_eps = np.sqrt( config['sigma_eps'])  # sqrt(0.0001)
        self.sigma_eps_square = config['sigma_eps']
        self.coe_range = [-5., 2.]  # config['coe_range']
        #self.theta_lr = 1
        #self.coe_range = np.array([1e-5, 100.]) / self.theta_lr  # config['coe_range']
        #self.exp_range = config['exp_range']

        # counter.
        self.updates_so_far = 0
        # Tensorflow tools.
        self.sess = sess
        self.graph = graph if graph is not None else tf.get_default_graph()

    def construct_model(self, need_summary=True):
        with self.graph.as_default():
            # Parameters of GP kernel
            self.theta_0 = tf.get_variable(name='theta_0',
                                          shape=[self.phi_dim],
                                          initializer=tf.random_uniform_initializer(minval=-2., maxval=-1.),
                                          #initializer=tf.random_uniform_initializer(minval=0.1/self.theta_lr, maxval=10.0/self.theta_lr),
                                          constraint=lambda t: tf.clip_by_value(t, self.coe_range[0], self.coe_range[1]))
            self.theta = tf.get_variable(name='theta',
                                         shape=[self.phi_dim],
                                         initializer=tf.random_uniform_initializer(minval=0., maxval=0.),
                                         constraint=lambda t: tf.clip_by_value(t, self.coe_range[0], self.coe_range[1]))

            # X_support: x points collected from related tasks. Shape (M, n_sample, x_dim)
            # Y_support: y points collected from related tasks. Shape (M, n_sample, y_dim)
            self.X_support = tf.placeholder(tf.float32, shape=[None, None, self.x_dim], name="X_support")
            self.Y_support = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="Y_support")
            # X_query: query points from the target task. (M, n_query, x_dim)
            # Y_query: target points from the target task. (M, n_query, y_dim)
            self.X_query = tf.placeholder(tf.float32, shape=[None, None, self.x_dim], name="X_query")
            #self.Y_query = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="Y_query")
            self.n_batch = tf.shape(self.X_support)[0]  
            self.n_support = tf.shape(self.X_support)[1]

            # Map input to feature space
            # self.Phi_support shape (M, n_support, phi_dim)
            with tf.variable_scope('phi', reuse=True):
                self.Phi_support = tf.map_fn(lambda x: self.basis(x), elems=self.X_support, dtype=tf.float32)

            # self.Phi_query shape (M, n_query, phi_dim)
            with tf.variable_scope('phi', reuse=None):
                self.Phi_query = tf.map_fn(lambda x: self.basis(x), elems=self.X_query, dtype=tf.float32)

            # self.dis_support shape: (M, n_support, n_support, phi_dim)
            self.dis_support = tf.map_fn(lambda x: self.calc_pairwise_distance(*x),
                                         elems=(self.Phi_support, self.Phi_support),
                                         dtype=tf.float32)

            # self.R shape: (M, n_support, n_support)
            self.R = tf.map_fn(lambda x: self.calc_covariance(x),
                               elems=self.dis_support,
                               dtype=tf.float32)
            self.R = self.R + tf.eye(self.n_support) * self.sigma_eps_square

            self.chol_L = chol_L = tf.linalg.cholesky(self.R)  # chol_L is a lower triangular matrix, shape: (M, n_support, n_support).
            self.Ln_Det_R = Ln_Det_R = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L)), axis=-1)  # Ln_Det_R shape: (M)
            # LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(U))))  # !!! np.abs???

            Y_support_vector = self.Y_support  # Y_support_vector shape: (M, n_support, y_dim)
            one_vector = tf.ones(shape=[self.n_batch, self.n_support, 1], dtype=tf.float32)  # one_vector shape: (M, n_support, 1)
            mu = tf.math.divide((tf.linalg.transpose(one_vector) @ tf.linalg.cholesky_solve(chol_L, Y_support_vector)),
                                (tf.linalg.transpose(one_vector) @ tf.linalg.cholesky_solve(chol_L, one_vector)))  # mu shape: (M, 1, y_dim)
            self.mu = mu[:, 0, :]  # self.mu shape: (M, y_dim)
            # reasoning of mu: 1 @ R^-1 @ y  ->  1 @ A, where R @ A = y, R = chol_L @ chol_L.T

            Y_support_cen_vector = Y_support_vector - mu  # Y_support_cen_vector shape: (M, n_support, y_dim)
            self.temp_invR_ycen = temp_invR_ycen = tf.linalg.cholesky_solve(chol_L, Y_support_cen_vector)  # (M, n_support, y_dim)
            sigma2 = tf.math.divide(tf.matmul(tf.linalg.transpose(Y_support_cen_vector), temp_invR_ycen), tf.cast(self.n_support, dtype=tf.float32))  # sigma2 shape: (M, y_dim, y_dim)
            self.sigma2 = tf.linalg.diag_part(sigma2)  # self.sigma^2 shape: (M, y_dim)

            # The loss function is expectation of this predictive nll.
            self.nll = .5 * tf.cast(self.n_support, dtype=tf.float32) * tf.math.log(self.sigma2) + .5 * tf.reshape(Ln_Det_R, [self.n_batch, 1])  
            # !!! this is same as GEO98 but diffs from NIPS20
            self.total_loss = tf.reduce_mean(self.nll)
            #tf.summary.scalar('0.total_loss', self.total_loss)

            """
            Prediction:
            for quick prediction, we need feed the following variables:
                self.Phi_support: (M, n_support, phi_dim)
                self.mu: (M, y_dim) -> (M, 1, y_dim)
                temp_invR_ycen: (M, n_support, y_dim)
                chol_L: (M, n_support, n_support).
                self.sigma: (M, y_dim) -> (M, 1, y_dim)
            """
            self.q_phi_train = tf.placeholder(tf.float32, shape=[None, None, self.phi_dim], name="q_phi_train")
            self.q_mu = tf.placeholder(tf.float32, shape=[None, 1, self.y_dim], name="q_mu")
            self.q_sigma2 = tf.placeholder(tf.float32, shape=[None, 1, self.y_dim], name="q_sigma2")
            self.q_invR_ycen = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="q_invR_ycen")
            self.q_chol_L = tf.placeholder(tf.float32, shape=[None, None, None], name="q_chol_L")

            self.dis_query = tf.abs(self.Phi_query - self.q_phi_train)
            self.r = tf.expand_dims(self.calc_covariance(self.dis_query), axis=-1)
            self.mu_hat = self.q_mu + tf.matmul(tf.linalg.transpose(self.r), self.q_invR_ycen)
            self.rRr = one_rRr = tf.expand_dims(1. - tf.linalg.transpose(self.r) @ tf.linalg.cholesky_solve(self.q_chol_L, self.r), axis=-1)
            def batch_sigma2(rRr, s2_hat):
                return tf.tensordot(rRr, s2_hat, axes=[[-1], [0]])
            self.sigma2_hat = tf.map_fn(lambda x: batch_sigma2(*x), elems=(one_rRr, self.q_sigma2), dtype=tf.float32)

            """
            Setups
            """
            # Optimizer setups.
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # self.lr_decay = tf.train.exponential_decay(self.lr, global_step, 1, 0.995)
            self.optimizer = tf.train.AdamOptimizer(self.lr)  # _decay
            self.optimizer2 = tf.train.AdamOptimizer(self.lr*10)  # _decay

            self.train_op = self.optimizer.minimize(self.total_loss, global_step=global_step)  ### v3
            self.train_kernel = self.optimizer2.minimize(self.total_loss, var_list=[self.theta])

            self.para_initializer = tf.variables_initializer(var_list=[self.theta])

            # Saver and initializer.
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    # ----  TF operations ---- #
    def basis(self, inp, name='basis'):
        layer_sizes = self.config['nn_layers']
        activations = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'leakyrelu': tf.nn.leaky_relu,
            'relu6': tf.nn.relu6
        }
        activation = activations[self.config['activation']]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for units in layer_sizes:
                inp = tf.layers.dense(inputs=inp, units=units, activation=activation)
        return inp

    def calc_pairwise_distance(self, Phi_query, Phi_support):
        # compute distance between Phi_query and Phi_supoort 
        dis = tf.abs(tf.expand_dims(Phi_query, 1) - tf.expand_dims(Phi_support, 0))
        return dis  # (n_query, n_support, n_phi) 

    def calc_covariance(self, X):  # dis_train shape [n_support, n_support, phi_dim], dis_test shape [M=1, n_support, phi_dim]
        theta = tf.clip_by_value((self.theta_0 + self.theta), self.coe_range[0], self.coe_range[1])
        theta = tf.math.pow(10.0, theta)
        #theta = self.theta_lr * theta
        return tf.exp(-tf.math.reduce_sum(theta * tf.math.pow(X, 2.0), axis=-1))  # R shape [n_support, n_support]; r shape = [M=1, n_support]

    # ---- Train and Test functions ------ #
    def train(self, dataset, num_train_updates, b_save=False, obj=None, b_track=False, track_frq=100):
        batch_size = self.config['meta_batch_size']
        support_horizon = self.config['data_horizon']
        query_horizon = self.config['test_horizon']

        if obj is None:
            print('training model for all objs')
        else:
            print('training model for obj:', obj)

        n_supporting = support_horizon + query_horizon

        for i in range(num_train_updates):
            self.sess.run(self.para_initializer)
            x, y = dataset.sample(n_funcs=batch_size, n_samples=n_supporting)

            if obj is None:
                feed_dict = {
                    self.X_support: x,
                    self.Y_support: y
                }
            else:
                feed_dict = {
                    self.X_support: x,
                    self.Y_support: y[:, :, obj].reshape(batch_size, n_supporting, 1)
                }
            loss, _ = self.sess.run([self.total_loss, self.train_op], feed_dict)

            if (i + 1) % 5 == 0:
                theta, phi_X = self.sess.run([self.theta_0, self.Phi_support], feed_dict)
                print('loss of the update iteration {}: {:.6f}; theta_0[0]: {:.6f}'.format(i + 1, loss, theta[0]))
                #print(phi_X[0, 0, :])
                #self.theta_lr*theta[0])) #np.power(10, theta[0])))
                """
                if b_track and (i + 1) % track_frq == 0:
                    model_name = str(i + 1) + '-' + str(loss)
                    self.save(name=model_name, display=False)
                """
            self.updates_so_far += 1
        print(" --- --- meta training done --- ---")
        if b_save:
            self.save()

    def parameter_init(self):
        print("theta. before init", self.sess.run(self.theta)[0])
        self.sess.run(self.para_initializer)
        print("after init", self.sess.run(self.theta)[0])

    def no_adapt(self, X_support, Y_support):
        feed_dict = {
            self.X_support: X_support,
            self.Y_support: Y_support
        }
        theta, phi_train, mu, sigma2, invR_ycen, chol_L = self.sess.run([self.theta, self.Phi_support, self.mu, self.sigma2, self.temp_invR_ycen, self.chol_L], feed_dict)
        #print("theta: {:.10f}".format(theta[0, 0] + delta_theta[0, 0]))
        #print("no adapt theta[0]:", np.log10(self.theta_lr*theta[:5]))
        #print("no adapt theta[0]:", np.power(10, theta[:5]))
        print("no adapt theta[0]:", theta[:5])
        return phi_train, mu.reshape(1, 1, self.y_dim), sigma2.reshape(1, 1, self.y_dim), invR_ycen, chol_L

    # X_support shape: [1, n_query, x_dim]
    # Y_support shape: (1, n_query, y_dim)
    def adapt(self, X_support, Y_support, b_speed_up=False):
        feed_dict = {
            self.X_support: X_support,
            self.Y_support: Y_support
        }

        if b_speed_up:
            """
                self.Phi_support: (M=1, n_support, phi_dim)
                self.mu: (M=1, y_dim) -> (M=1, 1, y_dim)
                self.sigma: (M=1, y_dim) -> (M=1, 1, y_dim)
                temp_invR_ycen: (M=1, n_support, y_dim)
                chol_L: (M=1, n_support, n_support).
            """
            theta, _, phi_train, mu, sigma2, invR_ycen, chol_L = \
                self.sess.run([self.theta, self.train_kernel, self.Phi_support, self.mu, self.sigma2, self.temp_invR_ycen, self.chol_L], feed_dict)
            #print("theta: {:.10f}".format(theta[0, 0] + delta_theta[0, 0]))
            #print("adapt theta[0]:", np.log10(self.theta_lr * theta[:5]))
            #print("adapt theta[0]:", np.power(10, theta[:5]))
            print("adapt theta[0]:", theta[:5])
            return phi_train, mu.reshape(1, 1, self.y_dim), sigma2.reshape(1, 1, self.y_dim), invR_ycen, chol_L
        else:
            theta, _ = self.sess.run([self.theta, self.train_kernel], feed_dict)
            return

    # a more efficient version of def test(), inputs are obtained from the method 'adapt(b_speed_up=True)':
    # phi_train, tf.expand_dims(mu, axis=1), tf.expand_dims(sigma2, axis=1), invR_ycen, chol_L
    def predict(self, phi_train, mu, sigma2, invR_ycen, chol_L, X_query, b_sigma=True):
        n_query = np.shape(X_query)[1]
        feed_dict = {
            self.X_query: X_query,
            self.q_phi_train: phi_train,
            self.q_mu: mu,
            self.q_sigma2: sigma2,
            self.q_invR_ycen: invR_ycen,
            self.q_chol_L: chol_L
        }
        if b_sigma:
            mu_hat, sigma2_hat = self.sess.run([self.mu_hat, self.sigma2_hat], feed_dict)
            return np.reshape(mu_hat, (1, n_query, self.y_dim)), np.reshape(sigma2_hat, (1, n_query, n_query, self.y_dim))
        else:
            mu_hat = self.sess.run(self.mu_hat, feed_dict)
            return np.reshape(mu_hat, (1, n_query, self.y_dim))

    # ---- Save and Restore ------
    def save(self, name=None, display=False):
        if display:
            for i, var in enumerate(self.saver._var_list):
                print('Var {}: {}'.format(i, var))
        if name:
            save_path = self.saver.save(self.sess, self.config['model_save_path'] + str(name))
        else:
            save_path = self.saver.save(self.sess, self.config['model_save_path'])
        #print('Saved to:', save_path)

    def restore(self, name=None, display=False):
        if display:
            for i, var in enumerate(self.saver._var_list):
                print('Var {}: {}'.format(i, var))
        if name:
            model_path = self.saver.restore(self.sess, (self.config['model_save_path'] + str(name)))
        else:
            model_path = self.saver.restore(self.sess, self.config['model_save_path'])
        #print('Restored model from:', model_path)

    def close(self):
        self.sess.close()

