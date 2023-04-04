import tensorflow as tf
import numpy as np
import time
from copy import deepcopy
from tensorflow.python.ops.parallel_for import gradients


class ALPaCA:
    def __init__(self, config, sess, graph=None, preprocess=None, f_nom=None):
        self.config = deepcopy(config)
        self.lr = config['lr']
        self.x_dim = config['x_dim']
        self.phi_dim = config['nn_layers'][-1]
        self.y_dim = config['y_dim']
        self.sigma_eps = self.config['sigma_eps']

        self.updates_so_far = 0
        self.sess = sess
        self.graph = graph if graph is not None else tf.get_default_graph()

        # y = K^T phi( preprocess(x) ) + f_nom(x)
        self.preprocess = preprocess
        self.f_nom = f_nom

    def construct_model(self):
        with self.graph.as_default():
            last_layer = self.config['nn_layers'][-1]

            if self.sigma_eps is list:
                self.SigEps = tf.diag(np.array(self.sigma_eps))
            else:
                self.SigEps = self.sigma_eps * tf.eye(self.y_dim)
            self.SigEps = tf.reshape(self.SigEps, (1, 1, self.y_dim, self.y_dim))

            # try making it learnable
            # self.SigEps = tf.get_variable('sigma_eps', initializer=self.SigEps )

            # Prior Parameters of last layer
            self.K = tf.get_variable('K_init', shape=[last_layer, self.y_dim])  # \bar{K}_0

            self.L_asym = tf.get_variable('L_asym', shape=[last_layer, last_layer])  # cholesky decomp of \Lambda_0
            self.L = self.L_asym @ tf.transpose(self.L_asym)  # \Lambda_0

            # x: query points (M, N_test, x_dim)
            # y: target points (M, N_test, y_dim) ( what K^T phi(x) should approximate )
            self.x = tf.placeholder(tf.float32, shape=[None, None, self.x_dim], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="y")

            # Points used to compute posterior using BLR
            # context_x: x points available for context (M, N_context, x_dim)
            # context_y: y points available for context (M, N_context, y_dim)
            self.context_x = tf.placeholder(tf.float32, shape=[None, None, self.x_dim], name="cx")
            self.context_y = tf.placeholder(tf.float32, shape=[None, None, self.y_dim], name="cy")

            # num_updates: number of context points from context_x,y to use when computing posterior. size (M,)
            self.num_models = tf.shape(self.context_x)[0]
            self.max_num_context = tf.shape(self.context_x)[1] * tf.ones((self.num_models,), dtype=tf.int32)
            self.num_context = tf.placeholder_with_default(self.max_num_context, shape=(None,))

            # Map input to feature space
            with tf.variable_scope('phi', reuse=None):
                # self.phi is (M, N_test, phi_dim)
                self.phi = tf.map_fn(lambda x: self.basis(x),
                                     elems=self.x,
                                     dtype=tf.float32)

            # Map context input to feature space
            with tf.variable_scope('phi', reuse=True):
                # self.context_phi is (M, N_context, phi_dim)
                self.context_phi = tf.map_fn(lambda x: self.basis(x),
                                             elems=self.context_x,
                                             dtype=tf.float32)

            # Evaluate f_nom if given, else use 0
            self.f_nom_cx = tf.zeros_like(self.context_y)
            self.f_nom_x = 0  # tf.zeros_like(self.y)
            if self.f_nom is not None:
                self.f_nom_cx = self.f_nom(self.context_x)
                self.f_nom_x = self.f_nom(self.x)

            # Subtract f_nom from context points before BLR
            self.context_y_blr = self.context_y - self.f_nom_cx

            # Compute posterior weights from context data
            with tf.variable_scope('blr', reuse=None):
                # posterior_K is (M, phi_dim, y_dim), posterior_L_inv is (M, phi_dim, phi_dim)
                self.posterior_K, self.posterior_L_inv = tf.map_fn(lambda x: self.batch_blr(*x),
                                                                   elems=(self.context_phi, self.context_y_blr, self.num_context),
                                                                   dtype=(tf.float32, tf.float32))

            # Compute posterior predictive distribution, and evaluate targets self.y under this distribution
            self.mu_pred, self.Sig_pred, self.predictive_nll = self.compute_pred_and_nll()

            # The loss function is expectation of this predictive nll.
            self.total_loss = tf.reduce_mean(self.predictive_nll)
            tf.summary.scalar('total_loss', self.total_loss)

            self.optimizer = tf.train.AdamOptimizer(self.lr)

            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=global_step)

            self.train_writer = tf.summary.FileWriter('summaries/' + str(time.time()), self.sess.graph, flush_secs=10)
            self.merged = tf.summary.merge_all()

            self.saver = tf.train.Saver()

            self.sess.run(tf.global_variables_initializer())

    # ----  TF operations ---- #
    def basis(self, x, name='basis'):
        layer_sizes = self.config['nn_layers']
        activations = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid
        }
        activation = activations[self.config['activation']]

        if self.preprocess is None:
            inp = x
        else:
            inp = self.preprocess(x)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for units in layer_sizes:
                inp = tf.layers.dense(inputs=inp, units=units, activation=activation)

        return inp

    def batch_blr(self, X, Y, num):
        X = X[:num, :]
        Y = Y[:num, :]
        Ln_inv = tf.matrix_inverse(tf.transpose(X) @ X + self.L)
        Kn = Ln_inv @ (tf.transpose(X) @ Y + self.L @ self.K)
        return tf.cond(num > 0, lambda: (Kn, Ln_inv), lambda: (self.K, tf.linalg.inv(self.L)))

    def compute_pred_and_nll(self):
        """
        Uses self.posterior_K and self.posterior_L_inv and self.f_nom_x to generate the posterior predictive.
        Returns:
            mu_pred = posterior predictive mean at query points self.x
                        shape (M, T, y_dim)
            Sig_pred = posterior predictive variance at query points self.x
                        shape (M, T, y_dim, y_dim)
            predictive_nll = negative log likelihood of self.y under the posterior predictive density
                        shape (M, T)
        """
        mu_pred = batch_matmul(tf.matrix_transpose(self.posterior_K), self.phi) + self.f_nom_x
        spread_fac = 1 + batch_quadform(self.posterior_L_inv, self.phi)
        Sig_pred = tf.expand_dims(spread_fac, axis=-1) * tf.reshape(self.SigEps, (1, 1, self.y_dim, self.y_dim))

        # Score self.y under predictive distribution to obtain loss
        with tf.variable_scope('loss', reuse=None):
            logdet = self.y_dim * tf.log(spread_fac) + tf.linalg.logdet(self.SigEps)
            Sig_pred_inv = tf.linalg.inv(Sig_pred)
            quadf = batch_quadform(Sig_pred_inv, (self.y - mu_pred))

        predictive_nll = tf.squeeze(logdet + quadf, axis=-1)

        # log stuff for summaries
        self.rmse_1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(mu_pred - self.y)[:, 0, :], axis=-1)))
        self.mpv_1 = tf.reduce_mean(tf.matrix_determinant(Sig_pred[:, 0, :, :]))
        tf.summary.scalar('RMSE_1step', self.rmse_1)
        tf.summary.scalar('MPV_1step', self.mpv_1)

        return mu_pred, Sig_pred, predictive_nll

    # ---- Train and Test functions ------ #
    def train(self, dataset, num_train_updates):
        batch_size = self.config['meta_batch_size']
        horizon = self.config['data_horizon']
        test_horizon = self.config['test_horizon']

        # minimize loss
        for i in range(num_train_updates):
            x, y = dataset.sample(n_funcs=batch_size, n_samples=horizon + test_horizon)

            feed_dict = {
                self.context_y: y[:, :horizon, :],
                self.context_x: x[:, :horizon, :],
                self.y: y[:, horizon:, :],
                self.x: x[:, horizon:, :],
                self.num_context: np.random.randint(horizon + 1, size=batch_size)
            }

            summary, loss, _ = self.sess.run([self.merged, self.total_loss, self.train_op], feed_dict)

            if i % 50 == 0:
                print('loss:', loss)

            self.train_writer.add_summary(summary, self.updates_so_far)
            self.updates_so_far += 1

    # x_c, y_c, x are all [N, n]
    # returns mu_pred, Sig_pred
    def test(self, x_c, y_c, x):
        feed_dict = {
            self.context_y: y_c,
            self.context_x: x_c,
            self.x: x
        }
        mu_pred, Sig_pred = self.sess.run([self.mu_pred, self.Sig_pred], feed_dict)
        return mu_pred, Sig_pred

    # convenience function to use just the encoder on numpy input
    def encode(self, x):
        feed_dict = {
            self.x: x
        }
        return self.sess.run(self.phi, feed_dict)

    # ---- Save and Restore ------
    def save(self, name=None):
        if name:
            save_path = self.saver.save(self.sess, self.config['event_file_name'] + str(name))
        else:
            save_path = self.saver.save(self.sess, self.config['event_file_name'])
        print('Saved to:', save_path)

    def restore(self, name=None):
        if name:
            model_path = self.saver.restore(self.sess, (self.config['event_file_name'] + str(name)))
        else:
            model_path = self.saver.restore(self.sess, self.config['event_file_name'])
        print('Restored model from:', model_path)


# given mat [a,b,c,...,N,N] and batch_v [a,b,c,...,M,N], returns [a,b,c,...,M,N]
def batch_matmul(mat, batch_v, name='batch_matmul'):
    with tf.name_scope(name):
        return tf.matrix_transpose(tf.matmul(mat, tf.matrix_transpose(batch_v)))


# works for A = [...,n,n] or [...,N,n,n]
# (uses the same matrix A for all N b vectors in the first case)
# assumes b = [...,N,n]
# returns  [...,N,1]
def batch_quadform(A, b):
    A_dims = A.get_shape().ndims
    b_dims = b.get_shape().ndims
    b_vec = tf.expand_dims(b, axis=-1)
    if A_dims == b_dims + 1:
        return tf.squeeze(tf.matrix_transpose(b_vec) @ A @ b_vec, axis=-1)
    elif A_dims == b_dims:
        Ab = tf.expand_dims(tf.matrix_transpose(A @ tf.matrix_transpose(b)), axis=-1)  # ... x N x n x 1
        return tf.squeeze(tf.matrix_transpose(b_vec) @ Ab, axis=-1)  # ... x N x 1
    else:
        raise ValueError('Matrix size of %d is not supported.' % (A_dims))


# takes in y = (..., y_dim)
#          x = (..., x_dim)
# returns dydx = (..., y_dim, x_dim), the jacobian of y wrt x
def batch_2d_jacobian(y, x):
    shape = tf.shape(y)
    y_dim = y.get_shape().as_list()[-1]
    x_dim = x.get_shape().as_list()[-1]
    batched_y = tf.reshape(y, (-1, y_dim))
    batched_x = tf.reshape(x, (-1, x_dim))

    batched_dydx = gradients.batch_jacobian(y, x)

    dydx = tf.reshape(batched_dydx, tf.concat((shape, [x_dim]), axis=0))
    return dydx


# ------------ END General ALPaCA -------------#

def blr_update_np(K, L, X, Y):
    Ln_inv = np.linalg.inv(X.T @ X + L)
    Kn = Ln_inv @ (X.T @ Y + L @ K)
    return Kn, Ln_inv


def sampleMN(K, L_inv, Sig):
    mean = np.reshape(K.T, [-1])
    cov = np.kron(Sig, L_inv)
    K_vec = np.random.multivariate_normal(mean, cov)
    return np.reshape(K_vec, K.T.shape).T


def tp(x):
    return np.swapaxes(x, -1, -2)


def extract_x(xu, x_dim):
    xu_shape = tf.shape(xu)
    begin = tf.zeros_like(xu_shape)
    size = tf.concat([-1 * tf.ones_like(xu_shape, dtype=tf.int32)[:-1], [x_dim]], axis=0)
    x = tf.slice(xu, begin, size)
    return x
