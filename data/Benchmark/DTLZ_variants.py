import numpy as np
from pyDOE import lhs

""" Version: 2022-June-15
Generate DTLZ functions and their variants.
To generate different variants for DTLZ 2-6, rewrite the method 'obj_func_on_variants' (in class DTLZ)
To generate different variants for DTLZ 1 or DTLZ 7, rewrite the method '_evaluate_on_variants' (in subclass DTLZ1 or DTLZ 7)

DTLZ 2-6 generate variants by using add_list and mul_list
DTLZ 1 and 7 generate variants by using add_list. No mul_list currently.
"""


class DTLZ:
    def __init__(self, config, k=None):
        self.n_vars = config['x_dim']
        self.n_objs = config['y_dim']
        self.add_range = config['add_range']
        self.mul_range = config['mul_range']

        if self.n_vars:
            self.k = self.n_vars - self.n_objs + 1
        elif k:
            self.k = k
            self.n_var = k + self.n_objs - 1
        else:
            raise Exception("Either provide number of variables or k!")

    def g1(self, X_M):  # for DTLZ 1 and 3.
        return 100 * (self.k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):  # for DTLZ 2, 4, and 5.
        return np.sum(np.square(X_M - 0.5), axis=1)

    def g3(self, X_M):  # for DTLZ 6.
        return np.sum(np.power(X_M, 0.1), axis=1)

    def _evaluate_on_variants(self, x, add_list, mul_list):
        return

    def evaluate(self, x):
        return

    def sample(self, n_funcs, n_samples, b_variant=True, b_return_lists=False):
        """
        :param n_funcs: The number of function variants in a sampling batch. Type: int.
        :param n_samples: The number of samples collected from a function variant. Type: int
        :param b_variant: Produce function variant or original function. Type: bool
        :param b_return_lists: Return variant parameters (add_list and mul_list) or not. Type: bool
        :return: The sampled data x and evaluated fitness y.
        """
        x = np.zeros((n_funcs, n_samples, self.n_vars))
        y = np.zeros((n_funcs, n_samples, self.n_objs))

        if b_variant:
            y = np.zeros((n_funcs, n_samples, self.n_objs))  # we assume variants for meta-learning will be used in OREA.
            add_list = self.add_range[0] + np.random.rand(n_funcs * self.n_objs) * (self.add_range[1] - self.add_range[0])
            add_list = add_list.reshape((n_funcs, self.n_objs))
            mul_list = self.mul_range[0] + np.random.rand(n_funcs * self.n_objs) * (self.mul_range[1] - self.mul_range[0])
            mul_list = mul_list.reshape((n_funcs, self.n_objs))
            for i in range(n_funcs):
                x_samp = lhs(self.n_vars, n_samples)
                y_samp = self._evaluate_on_variants(x_samp, add_list[i], mul_list[i])
                x[i, :, :] = x_samp
                y[i, :, :] = y_samp
            if b_return_lists:
                return x, y, add_list, mul_list
        else:
            for i in range(n_funcs):
                x_samp = lhs(self.n_vars, n_samples)
                y_samp = self.evaluate(x_samp)
                x[i, :, :] = x_samp
                y[i, :, :] = y_samp
        return x, y

    def obj_func_on_variants(self, X_, g, add_list=None, mul_list=None, alpha=1):
        """
        # original
        f = (1 + g)
        # with only add_list
        f = add_list[0] + g
        f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1]], alpha) * np.pi / 2.0), axis=1)
        """
        f = add_list[0] + g
        f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1]], alpha) * np.pi / mul_list[0]), axis=1)
        for i in range(1, self.n_objs):
            """
            # original
            _f = (1 + g)
            # with only add_list
            _f = add_list[i] + g
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)
            """
            # with mul_list:
            _f = add_list[i] + g
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / mul_list[i]), axis=1)
            _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / mul_list[i])

            f = np.concatenate((f, _f), axis=0)
        return np.reshape(f, (-1, self.n_objs), order='F')

    def obj_func(self, X_, g, alpha=1):
        f = (1 + g)
        f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1]], alpha) * np.pi / 2.0), axis=1)
        for i in range(1, self.n_objs):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)
            f = np.concatenate((f, _f), axis=0)
        return np.reshape(f, (-1, self.n_objs), order='F')


class DTLZ1(DTLZ):
    def __init__(self, config, k=None):
        super().__init__(config, k)

    def _evaluate_on_variants(self, x, add_list, mul_list):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :param add_list: The addition parameter for each objective. Type: array. Shape: (n_objs).
        :param mul_list: The multiply parameter for each objective. Type: array. Shape: (n_objs).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]  # y, z
        g = self.g1(X_M)

        f = add_list[0] + g
        f *= np.prod(X_[:, :X_.shape[1]], axis=1)
        for i in range(1, self.n_objs):

            _f = add_list[i] + g
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            _f *= 1 - X_[:, X_.shape[1] - i]

            f = np.concatenate((f, _f), axis=0)
        return np.reshape(f, (-1, self.n_objs), order='F')

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]  # y, z
        g = self.g1(X_M)

        f = 0.5 * (1 + g)
        f *= np.prod(X_[:, :X_.shape[1]], axis=1)
        for i in range(1, self.n_objs):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            _f *= 1 - X_[:, X_.shape[1] - i]
            f = np.concatenate((f, _f), axis=0)
        return np.reshape(f, (-1, self.n_objs), order='F')


class DTLZ2(DTLZ):
    def __init__(self, config, k=None):
        super().__init__(config, k)

    def _evaluate_on_variants(self, x, add_list, mul_list):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :param add_list: The addition parameter for each objective. Type: array. Shape: (n_objs).
        :param mul_list: The multiply parameter for each objective. Type: array. Shape: (n_objs).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g2(X_M)
        return self.obj_func_on_variants(X_, g, add_list, mul_list)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g2(X_M)
        return self.obj_func(X_, g)


class DTLZ3(DTLZ):
    def __init__(self, config, k=None):
        super().__init__(config, k)

    def _evaluate_on_variants(self, x, add_list, mul_list):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :param add_list: The addition parameter for each objective. Type: array. Shape: (n_objs).
        :param mul_list: The multiply parameter for each objective. Type: array. Shape: (n_objs).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g1(X_M)
        return self.obj_func_on_variants(X_, g, add_list, mul_list, alpha=1)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g1(X_M)
        return self.obj_func(X_, g, alpha=1)


class DTLZ4(DTLZ):
    def __init__(self, config, k=None):
        super().__init__(config, k)

    def _evaluate_on_variants(self, x, add_list, mul_list):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :param add_list: The addition parameter for each objective. Type: array. Shape: (n_objs).
        :param mul_list: The multiply parameter for each objective. Type: array. Shape: (n_objs).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g2(X_M)
        return self.obj_func_on_variants(X_, g, add_list, mul_list, alpha=100)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g2(X_M)
        return self.obj_func(X_, g, alpha=100)


class DTLZ5(DTLZ):
    def __init__(self, config, k=None):
        super().__init__(config, k)

    def _evaluate_on_variants(self, x, add_list, mul_list):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :param add_list: The addition parameter for each objective. Type: array. Shape: (n_objs).
        :param mul_list: The multiply parameter for each objective. Type: array. Shape: (n_objs).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g2(X_M)
        theta = 1. / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = np.column_stack((x[:, 0], theta[:, 1:]))
        return self.obj_func_on_variants(theta, g, add_list, mul_list)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g2(X_M)
        theta = 1. / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = np.column_stack((x[:, 0], theta[:, 1:]))
        return self.obj_func(theta, g)


class DTLZ6(DTLZ):
    def __init__(self, config, k=None):
        super().__init__(config, k)

    def _evaluate_on_variants(self, x, add_list, mul_list):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :param add_list: The addition parameter for each objective. Type: array. Shape: (n_objs).
        :param mul_list: The multiply parameter for each objective. Type: array. Shape: (n_objs).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g3(X_M)
        theta = 1. / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = np.column_stack((x[:, 0], theta[:, 1:]))
        return self.obj_func_on_variants(theta, g, add_list, mul_list)

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g3(X_M)
        theta = 1. / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = np.column_stack((x[:, 0], theta[:, 1:]))
        return self.obj_func(theta, g)


class DTLZ7(DTLZ):
    def __init__(self, config, k=None):
        super().__init__(config, k)

    def _evaluate_on_variants(self, x, add_list, mul_list):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :param add_list: The addition parameter for each objective. Type: array. Shape: (n_objs).
        :param mul_list: The multiply parameter for each objective. Type: array. Shape: (n_objs).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        f = x[:, 0] + add_list[0]
        for i in range(1, self.n_objs - 1):
            f = np.column_stack((f, x[:, i] + add_list[i]))

        g = add_list[-1] + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_objs - np.sum(f / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f)), axis=1)
        return np.column_stack((f, (1 + g) * h))

    def evaluate(self, x):
        """
        :param x: The set of samples to be evaluated. Type: 2darray. Shape: (n_samples, n_vars).
        :return: The set of evaluated samples. Type: 2darray. Shape: (n_samples, n_objs).
        """
        f = x[:, 0]
        for i in range(1, self.n_objs - 1):
            f = np.column_stack((f, x[:, i]))

        g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_objs - np.sum(f / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f)), axis=1)
        return np.column_stack((f, (1 + g) * h))

