# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from copy import deepcopy

np.set_printoptions(linewidth=200)

""" Last update: 2023-03-26
A variant of MDKL, meta-learning only deep neural network.
"""

class MDKL_NN:
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

    def construct_model(self):
        with self.graph.as_default():
            # Parameters of GP kernel
            mid_coe_range = (self.coe_range[0] + self.coe_range[1]) / 2
            mid_exp_range = (self.exp_range[0] + self.exp_range[1]) / 2
            self.coeff = tf.get_variable(name='coeff',
                                         shape=[self.phi_dim],
                                         initializer=tf.random_uniform_initializer(minval=mid_coe_range, maxval=mid_coe_range),
                                         constraint=lambda t: tf.clip_by_value(t, self.coe_range[0], self.coe_range[1]))
            self.exponent = tf.get_variable(name='exponent',
                                            shape=[self.phi_dim],
                                            initializer=tf.random_uniform_initializer(minval=mid_exp_range, maxval=mid_exp_range),
                                            constraint=lambda t: tf.clip_by_value(t, self.exp_range[0], self.exp_range[1]))

            # X_support: support variables. Shape (M, n_support, x_dim)
            # Y_support: support objectives. Shape (M, n_support, y_dim)
            self.X_support = tf.placeholder(tf.float32, shape=[None, None, self.x_dim], name="X_support")
            self.Y_support = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="Y_support")
            # X_query: query variables. (M, n_query, x_dim)
            # Y_query: query objectives. (M, n_query, y_dim)
            self.X_query = tf.placeholder(tf.float32, shape=[None, None, self.x_dim], name="X_query")
            # self.Y_query = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="Y_query")
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

            chol_L = tf.linalg.cholesky(self.R)  # chol_L is a lower triangular matrix, shape: (M, n_train, n_train).
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
            sigma2 = tf.math.divide(tf.matmul(tf.linalg.transpose(Y_support_cen_vector), temp_invR_ycen),
                                    tf.cast(self.n_support, dtype=tf.float32))  # sigma2 shape: (M, y_dim, y_dim)
            self.sigma2 = tf.linalg.diag_part(sigma2)  # self.sigma^2 shape: (M, y_dim)

            # The loss function is expectation of this predictive nll.
            self.nll = .5 * tf.cast(self.n_support, dtype=tf.float32) * tf.math.log(self.sigma2) + .5 * tf.reshape(Ln_Det_R, [self.n_batch, 1])
            self.total_loss = tf.reduce_mean(self.nll)

            """
            Prediction:
            """
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
        coe = self.coeff
        coeff = tf.clip_by_value(coe, self.coe_range[0], self.coe_range[1])
        coeff = tf.math.pow(10.0, coeff)
        exp = self.exponent
        exponent = tf.clip_by_value(exp, self.exp_range[0], self.exp_range[1])
        return tf.exp(-tf.math.reduce_sum(coeff * tf.math.pow(X, exponent), axis=-1))  # R shape (n_support, n_support); r shape (M=1, n_support)

    # ---- Train and Test functions ------ #
    def parameter_init(self):
        # re-initialize parameters.
        self.sess.run(self.para_initializer)

    def train(self, dataset, num_train_updates):
        """
        :param dataset: A dataset to generate samples from related tasks.
        :param num_train_updates: the number of updates U.
        :return:
        """
        batch_size = self.config['meta_batch_size']
        support_horizon = self.config['data_horizon']
        query_horizon = self.config['test_horizon']

        n_samples = support_horizon + query_horizon

        for i in range(num_train_updates):
            self.sess.run(self.para_initializer)
            x, y = dataset.sample(n_funcs=batch_size, n_samples=n_samples)

            feed_dict = {
                self.X_support: x,
                self.Y_support: y
            }
            loss, _ = self.sess.run([self.total_loss, self.train_op], feed_dict)
            self.updates_so_far += 1

    def adapt(self, X_support, Y_support):
        """
        :param X_support: shape (1, n_support, x_dim)
        :param Y_support: shape (1, n_support, y_dim)
        :return:
        """
        feed_dict = {
            self.X_support: X_support,
            self.Y_support: Y_support
        }
        delta_coeff, delta_exponent, _ = self.sess.run([self.coeff, self.exponent, self.train_kernel], feed_dict)

    def test(self, X_support, Y_support, X_query):
        """
        :param X_support: shape (1, n_support, x_dim)
        :param Y_support: shape (1, n_support, y_dim)
        :param X_query: shape (1, 1, x_dim)
        :return: mu_hat, sigma2_hat
        """
        feed_dict = {
            self.X_support: X_support,
            self.Y_support: Y_support,
            self.X_query: X_query
        }

        mu_hat, sigma2_hat = self.sess.run([self.mu_hat, self.sigma2_hat], feed_dict)
        return np.reshape(mu_hat, (1, 1, 1)), np.reshape(sigma2_hat, (1, 1, 1, 1))

    # convenience function to use just the encoder on numpy input
    def encode(self, x):
        feed_dict = {
            self.X_query: x
        }
        return self.sess.run(self.Phi_query, feed_dict)

    # ---- Save and Restore ------
    def save(self, name=None, display=False):
        if display:
            for i, var in enumerate(self.saver._var_list):
                print('Var {}: {}'.format(i, var))
        if name:
            save_path = self.saver.save(self.sess, self.config['event_file_name'] + str(name))
        else:
            save_path = self.saver.save(self.sess, self.config['event_file_name'])
        print('Saved to:', save_path)

    def restore(self, name=None, display=False):
        if display:
            for i, var in enumerate(self.saver._var_list):
                print('Var {}: {}'.format(i, var))
        if name:
            model_path = self.saver.restore(self.sess, (self.config['event_file_name'] + str(name)))
        else:
            model_path = self.saver.restore(self.sess, self.config['event_file_name'])
        print('Restored model from:', model_path)

    def close(self):
        self.sess.close()



