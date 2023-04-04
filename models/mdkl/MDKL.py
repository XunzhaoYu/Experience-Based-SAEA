# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from copy import deepcopy

np.set_printoptions(linewidth=200)

""" Last update: 2023-03-26
Meta Deep Kernel Learning. 
"""

class MDKL:
    def __init__(self, config, sess, graph=None):
        # constants.
        self.config = deepcopy(config)
        self.x_dim = config['x_dim']
        self.y_dim = config['y_dim']
        self.phi_dim = config['nn_layers'][-1]

        self.lr = config['lr']
        self.sigma_eps = np.sqrt(config['sigma_eps'])
        self.sigma_eps_square = config['sigma_eps']
        self.coe_range = np.log10(config['coe_range'])  # [-5, 1]
        self.exp_range = config['exp_range']  # [1, 2]
        # counter.
        self.updates_so_far = 0
        # Tensorflow tools.
        self.sess = sess
        self.graph = graph if graph is not None else tf.get_default_graph()

    def construct_model(self, need_summary=True):
        with self.graph.as_default():
            # Parameters of GP kernel
            self.coeff_0 = tf.get_variable(name='coeff_0',
                                           shape=[self.phi_dim],
                                           initializer=tf.random_uniform_initializer(minval=-2., maxval=-1.),
                                           constraint=lambda t: tf.clip_by_value(t, self.coe_range[0], self.coe_range[1]))
            self.exponent_0 = tf.get_variable(name='exponent_0',
                                             shape=[self.phi_dim],
                                             initializer=tf.random_uniform_initializer(minval=1.3, maxval=1.7),
                                             constraint=lambda t: tf.clip_by_value(t, self.exp_range[0], self.exp_range[1]))
            self.coeff = tf.get_variable(name='coeff',
                                         shape=[self.phi_dim],
                                         initializer=tf.random_uniform_initializer(minval=0., maxval=0.),
                                         constraint=lambda t: tf.clip_by_value(t, self.coe_range[0], self.coe_range[1]))
            self.exponent = tf.get_variable(name='exponent',
                                            shape=[self.phi_dim],
                                            initializer=tf.random_uniform_initializer(minval=0., maxval=0.),
                                            constraint=lambda t: tf.clip_by_value(t, self.exp_range[0], self.exp_range[1]))

            # X_support: support variables. Shape (M, n_support, x_dim)
            # Y_support: support objectives. Shape (M, n_support, y_dim)
            self.X_support = tf.placeholder(tf.float32, shape=[None, None, self.x_dim], name="X_support")
            self.Y_support = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="Y_support")
            # X_query: query variables. (M, n_query, x_dim)
            # Y_query: query objectives. (M, n_query, y_dim)
            self.X_query = tf.placeholder(tf.float32, shape=[None, None, self.x_dim], name="X_query")
            #self.Y_query = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="Y_query")
            self.n_batch = tf.shape(self.X_support)[0]  
            self.n_support = tf.shape(self.X_support)[1]

            # Map input to feature space
            # self.Phi_support shape (M, n_support, phi_dim)
            with tf.variable_scope('phi', reuse=True):
                self.Phi_support = tf.map_fn(lambda x: self.basis(x), elems=self.X_support, dtype=tf.float32)

            # self.Phi_query shape (M, n_query, phi_dim)
            with tf.variable_scope('phi', reuse=True):
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
            # LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(U))))

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
            self.total_loss = tf.reduce_mean(self.nll)

            """
            Prediction:
            for quick prediction, we need feed the following variables:
                self.Phi_support: (M, n_support, phi_dim)
                self.mu: (M, y_dim) -> (M, 1, y_dim)
                temp_invR_ycen: (M, n_support, y_dim)
                chol_L: (M, n_support, n_support).
                self.sigma: (M, y_dim) -> (M, 1, y_dim)
            """
            self.b_quick = tf.placeholder(tf.bool, name="b_quick")
            if self.b_quick is True:
                self.q_Phi_support = tf.placeholder(tf.float32, shape=[None, None, self.phi_dim], name="q_Phi_support")
                self.q_mu = tf.placeholder(tf.float32, shape=[None, 1, self.y_dim], name="q_mu")
                self.q_sigma2 = tf.placeholder(tf.float32, shape=[None, 1, self.y_dim], name="q_sigma2")
                self.q_invR_ycen = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="q_invR_ycen")
                self.q_chol_L = tf.placeholder(tf.float32, shape=[None, None, None], name="q_chol_L")

                self.dis_query = tf.abs(self.Phi_query - self.q_Phi_support)
                self.r = tf.expand_dims(self.calc_covariance(self.dis_query), axis=-1)
                self.mu_hat = self.q_mu + tf.matmul(tf.linalg.transpose(self.r), self.q_invR_ycen)
                self.rRr = one_rRr = tf.expand_dims(1. - tf.linalg.transpose(self.r) @ tf.linalg.cholesky_solve(self.q_chol_L, self.r), axis=-1)
                def batch_sigma2(rRr, s2_hat):
                    return tf.tensordot(rRr, s2_hat, axes=[[-1], [0]])
                self.sigma2_hat = tf.map_fn(lambda x: batch_sigma2(*x), elems=(one_rRr, self.q_sigma2), dtype=tf.float32)
            else:
                # self.dis_query shape: (M=1, n_support, phi_dim)
                self.dis_query = tf.abs(self.Phi_query - self.Phi_support)
                # self.r shape: (M=1, n_support, 1)
                self.r = tf.expand_dims(self.calc_covariance(self.dis_query), axis=-1)
                # self.mu_hat shape (M, 1, y_dim): (M, 1, y_dim) + (M, 1 ,n_support) @ (M, n_support, y_dim)
                self.mu_hat = mu + tf.matmul(tf.linalg.transpose(self.r), temp_invR_ycen)
                # self.sigma2_hat shape: (M, 1, 1, y_dim):  (M, 1, n_support) @ (M, n_support, 1)
                self.rRr = one_rRr = tf.expand_dims(1. - tf.linalg.transpose(self.r) @ tf.linalg.cholesky_solve(chol_L, self.r), axis=-1)  # (M, n_query=1, n_query=1, 1)
                sigma2_hat = tf.expand_dims(self.sigma2, axis=1)
                def batch_sigma2(rRr, s2_hat):
                    return tf.tensordot(rRr, s2_hat, axes=[[-1], [0]])
                self.sigma2_hat = tf.map_fn(lambda x: batch_sigma2(*x), elems=(one_rRr, sigma2_hat), dtype=tf.float32)

            """
            Setups
            """
            # Optimizer setups.
            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=global_step)
            self.train_kernel = self.optimizer.minimize(self.total_loss, var_list=[self.coeff, self.exponent])

            self.para_initializer = tf.variables_initializer(var_list=[self.coeff, self.exponent])

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
        # compute distance between Phi_query and Phi_support
        dis = tf.abs(tf.expand_dims(Phi_query, 1) - tf.expand_dims(Phi_support, 0))
        return dis  # (n_query, n_support, n_phi) 

    def calc_covariance(self, X):  # dis_support shape (n_support, n_support, phi_dim), dis_query shape (M=1, n_support, phi_dim)
        coeff = tf.clip_by_value((self.coeff_0 + self.coeff), self.coe_range[0], self.coe_range[1])
        coeff = tf.math.pow(10.0, coeff)
        exponent = tf.clip_by_value((self.exponent_0 + self.exponent), self.exp_range[0], self.exp_range[1])
        return tf.exp(-tf.math.reduce_sum(coeff * tf.math.pow(X, exponent), axis=-1))  # R shape (n_support, n_support); r shape = (M=1, n_support)

    # ---- Train and Test functions ------ #
    def parameter_init(self, b_print=False):
        if b_print:
            print("coeff and exponent. before init", self.sess.run([self.coeff[0], self.exponent[0]]))
        self.sess.run(self.para_initializer)
        if b_print:
            print("after init", self.sess.run([self.coeff[0], self.exponent[0]]))

    def train(self, dataset, num_train_updates, b_save=False, obj=None, b_track=False, track_frq=100):
        batch_size = self.config['meta_batch_size']
        support_horizon = self.config['data_horizon']
        query_horizon = self.config['test_horizon']

        if obj is None:
            print('training model for all objs')
        else:
            print('training model for obj:', obj)

        n_samples = support_horizon + query_horizon

        for i in range(num_train_updates):
            self.sess.run(self.para_initializer)
            x, y = dataset.sample(n_funcs=batch_size, n_samples=n_samples)

            if obj is None:
                feed_dict = {
                    self.X_support: x,
                    self.Y_support: y
                }
            else:
                feed_dict = {
                    self.X_support: x,
                    self.Y_support: y[:, :, obj].reshape(batch_size, n_samples, 1)
                }
            loss, _ = self.sess.run([self.total_loss, self.train_op], feed_dict)

            if (i + 1) % 5 == 0:
                coeff, exponent, phi_X = self.sess.run([self.coeff_0, self.exponent_0, self.Phi_support], feed_dict)
                print('loss of the update iteration {}: {:.6f}; coeff_0[0]: {:.6f}; exponent_0[0]: {:.6f}; '.format(i + 1, loss, coeff[0], exponent[0]))
                """
                if b_track and (i + 1) % track_frq == 0:
                    model_name = str(i + 1) + '-' + str(loss)
                    self.save(name=model_name, display=False)
                """
            self.updates_so_far += 1
        print(" --- --- meta training done --- ---")
        if b_save:
            self.save()

    def test(self, x_train, y_train, x_test):
        n_test = np.shape(x_test)[1]
        feed_dict = {
            self.b_quick: False,
            self.X_support: x_train,
            self.Y_support: y_train,
            self.X_query: x_test
        }
        mu_hat, sigma2_hat = self.sess.run([self.mu_hat, self.sigma2_hat], feed_dict)
        return np.reshape(mu_hat, (1, n_test, self.y_dim)), np.reshape(sigma2_hat, (1, n_test, n_test, self.y_dim))

    def no_adapt(self, X_support, Y_support):
        feed_dict = {
            self.X_support: X_support,
            self.Y_support: Y_support
        }
        coeff, exponent, Phi_support, mu, sigma2, invR_ycen, chol_L = self.sess.run([self.coeff, self.exponent, self.Phi_support, self.mu, self.sigma2, self.temp_invR_ycen, self.chol_L], feed_dict)
        print("no adapt coeff[0]:", coeff[:5])
        return Phi_support, mu.reshape(1, 1, self.y_dim), sigma2.reshape(1, 1, self.y_dim), invR_ycen, chol_L

    # X_support shape: (1, n_support, x_dim)
    # Y_support shape: (1, n_support, y_dim)
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
            coeff, exponent, _, Phi_support, mu, sigma2, invR_ycen, chol_L = \
                self.sess.run([self.coeff, self.exponent, self.train_kernel, self.Phi_support, self.mu, self.sigma2, self.temp_invR_ycen, self.chol_L], feed_dict)
            print("adapt coeff[0]:", coeff[:5])
            return Phi_support, mu.reshape(1, 1, self.y_dim), sigma2.reshape(1, 1, self.y_dim), invR_ycen, chol_L
        else:
            coeff, exponent, _ = self.sess.run([self.coeff, self.exponent, self.train_kernel], feed_dict)
            return

    # a more efficient version of def test(), inputs are obtained from the method 'adapt(b_speed_up=True)':
    # Phi_support, tf.expand_dims(mu, axis=1), tf.expand_dims(sigma2, axis=1), invR_ycen, chol_L
    def predict(self, Phi_support, mu, sigma2, invR_ycen, chol_L, X_query, b_sigma=True):
        n_query = np.shape(X_query)[1]
        feed_dict = {
            self.X_query: X_query,
            self.q_Phi_support: Phi_support,
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

