# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from copy import deepcopy

np.set_printoptions(linewidth=200)

""" Last update: 2023-04-04
Deep Kernel Learning.
"""

class DKL:
    def __init__(self, config, sess, graph=None):
        # constants.
        self.config = deepcopy(config)
        self.x_dim = config['x_dim']
        self.y_dim = config['y_dim']
        self.phi_dim = config['nn_layers'][-1]
        # algorithm setups.
        self.lr = config['lr']
        self.sigma_eps = np.sqrt(config['sigma_eps'])
        self.sigma_eps_square = config['sigma_eps']
        self.coe_range = np.log10(config['coe_range'])
        self.exp_range = config['exp_range']
        # counter.
        self.updates_so_far = 0
        # Tensorflow tools.
        self.sess = sess
        self.graph = graph if graph is not None else tf.get_default_graph()

    def construct_model(self):
        with self.graph.as_default():
            # Parameters of GP kernel
            self.coeff = tf.get_variable(name='coeff',
                                         shape=[self.phi_dim],
                                         initializer=tf.random_uniform_initializer(minval=self.coe_range[0], maxval=self.coe_range[1]),
                                         constraint=lambda t: tf.clip_by_value(t, self.coe_range[0], self.coe_range[1]))
            self.exponent = tf.get_variable(name='exponent',
                                            shape=[self.phi_dim],
                                            initializer=tf.random_uniform_initializer(minval=self.exp_range[0], maxval=self.exp_range[1]),
                                            constraint=lambda t: tf.clip_by_value(t, self.exp_range[0], self.exp_range[1]))

            self.X_train = tf.placeholder(tf.float32, shape=[None, self.x_dim], name="X_train")
            self.Y_train = tf.placeholder(tf.float32, shape=[None, self.y_dim], name="Y_train")
            self.X_test = tf.placeholder(tf.float32, shape=[None, self.x_dim], name="X_test")
            self.n_train = tf.shape(self.X_train)[0]

            # Map input to feature space
            # self.Phi_support shape (M, n_train, phi_dim)
            with tf.variable_scope('phi', reuse=True):
                self.Phi_train = self.basis(self.X_train)

            # self.Phi_query shape (M, n_test, phi_dim)
            with tf.variable_scope('phi', reuse=True):
                self.Phi_test = self.basis(self.X_test)

            # self.dis_train shape: (n_train, n_train, x_dim)
            self.dis_train = self.calc_pairwise_distance(self.X_train)

            # self.R shape: (n_train, n_train)
            self.R = self.calc_covariance(self.dis_train)
            self.R = self.R + tf.eye(self.n_train) * self.sigma_eps_square

            # chol_L is a lower triangular matrix, shape: (n_train, n_train).
            chol_L = tf.linalg.cholesky(self.R)
            self.Ln_Det_R = Ln_Det_R = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L)), axis=-1)
            # LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(U))))

            # y_train_vector shape: (n_train, y_dim)
            y_train_vector = self.Y_train
            # one_vector shape: (n_train, 1)
            one_vector = tf.ones(shape=[self.n_train, 1], dtype=tf.float32)
            # mu shape: (1, y_dim)
            mu = tf.math.divide((tf.linalg.transpose(one_vector) @ tf.linalg.cholesky_solve(chol_L, y_train_vector)),
                                (tf.linalg.transpose(one_vector) @ tf.linalg.cholesky_solve(chol_L, one_vector)))
            self.mu = tf.squeeze(mu)
            # reasoning of mu: 1 @ R^-1 @ y  ->  1 @ A, where R @ A = y, R = chol_L @ chol_L.T

            # y_train_cen_vector shape: (n_train, y_dim)
            y_train_cen_vector = y_train_vector - mu
            # (n_train, y_dim)
            self.temp_invR_ycen = temp_invR_ycen = tf.linalg.cholesky_solve(chol_L, y_train_cen_vector)
            # sigma2 shape: (y_dim, y_dim)
            sigma2 = tf.math.divide(tf.matmul(tf.linalg.transpose(y_train_cen_vector), temp_invR_ycen), tf.cast(self.n_train, dtype=tf.float32))
            self.sigma2 = tf.linalg.diag_part(sigma2)

            # The loss function is expectation of this predictive nll.
            self.nll = .5 * tf.cast(self.n_train, dtype=tf.float32) * tf.math.log(self.sigma2) + .5 * Ln_Det_R
            self.total_loss = tf.reduce_mean(self.nll)

            """
            Prediction:
            """
            # self.dis_test shape: (n_train, phi_dim)
            self.dis_test = tf.abs(self.Phi_test - self.Phi_train)
            # self.r shape: (n_train, 1)
            self.r = tf.expand_dims(self.calc_covariance(self.dis_test), axis=-1)
            # self.mu_hat shape (1, y_dim): (1, y_dim) + (1 ,n_train) @ (n_train, y_dim)
            self.mu_hat = mu + tf.squeeze(tf.linalg.transpose(self.r) @ temp_invR_ycen)
            # self.sigma2_hat shape: (1, 1, y_dim):  (1, n_train) @ (n_train, 1)
            self.rRr = one_rRr = tf.expand_dims(1. - tf.linalg.transpose(self.r) @ tf.linalg.cholesky_solve(chol_L, self.r), axis=-1)  # (n_test=1, n_test=1, 1)
            sigma2_hat = tf.expand_dims(self.sigma2, axis=0)  # (1, y_dim)
            self.sigma2_hat = tf.tensordot(one_rRr, sigma2_hat, axes=[[-1], [0]])
            """
            Setups
            """
            # Optimizer setups.
            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=global_step)
            self.global_para_initializer = tf.global_variables_initializer()

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


    def calc_pairwise_distance(self, X_train):
        # compute distance between X_train.
        dis = tf.abs(tf.expand_dims(X_train, 1) - tf.expand_dims(X_train, 0))
        return dis  # (n_train, n_train, x_dim)

    def calc_covariance(self, X):  # dis_train shape (n_train, n_train, x_dim), dis_test shape (n_train, x_dim)
        coe = self.coeff
        coeff = tf.clip_by_value(coe, self.coe_range[0], self.coe_range[1])
        coeff = tf.math.pow(10.0, coeff)
        exp = self.exponent
        exponent = tf.clip_by_value(exp, self.exp_range[0], self.exp_range[1])
        return tf.exp(-tf.math.reduce_sum(coeff * tf.math.pow(X, exponent), axis=-1))  # R shape (n_train, n_train); r shape (1, n_train)

    # ---- Train and Test functions ------ #
    def parameter_init(self):
        # re-initialize all parameters, including GP's and neural network's parameters.
        self.sess.run(self.global_para_initializer)

    def train(self, X_train, Y_train):
        """
        :param X_train: shape (1, n_train, x_dim)
        :param Y_train: shape (1, n_train, y_dim=1)
        :return:
        """
        feed_dict = {
            self.X_train: X_train[0],
            self.Y_train: Y_train[0]
        }
        loss, _ = self.sess.run([self.total_loss, self.train_op], feed_dict)

    def test(self, X_train, Y_train, X_test):
        """
        :param X_train: shape (1, n_train, x_dim)
        :param Y_train: shape (1, n_train, y_dim=1)
        :param X_test: shape (1, n_test=1, x_dim)
        :return: mu_hat, sigma2_hat
        """
        feed_dict = {
            self.X_train: X_train[0],
            self.Y_train: Y_train[0],
            self.X_test: X_test[0]
        }
        mu_hat, sigma2_hat = self.sess.run([self.mu_hat, self.sigma2_hat], feed_dict)
        return np.reshape(mu_hat, (1, 1, 1)), np.reshape(sigma2_hat, (1, 1, 1, 1))

    # convenience function to use just the encoder on numpy input
    def encode(self, x):
        feed_dict = {
            self.X_test: x[0]
        }
        return self.sess.run(self.Phi_test, feed_dict)

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



