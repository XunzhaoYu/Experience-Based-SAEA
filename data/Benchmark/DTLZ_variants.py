import numpy as np
from pyDOE import lhs
import yaml
#from comparisons.OREA.labeling_operator import domination_based_ordinal_values

"""
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


"""  # --- format validation ---
Example:
dataset = DTLZ1(config)
x, y = dataset.sample(n_funcs=10, n_samples=100)

Return:
x shape: (n_funcs, n_samples, n_vars)
y shape: (n_funcs, n_samples, n_objs)
"""

def DTLZ_validation(DTLZ_list):
    cfg_filename = '../configs/DTLZ-config.yml'
    with open(cfg_filename, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    # first two test data from SSCI, next two from PlatEMO. Test on package pymoo.
    if 1 in DTLZ_list:
        print('--- DTLZ 1 ---')
        dataset = DTLZ1(config)
        X = np.array([[0.512959, 0.596456, 0.770258, 0.216552, 0.793923, 0.993508, 0.659377, 0.692331, 0.021113, 0.118613],
                      [0.980125, 0.126840, 0.140081, 0.617695, 0.144347, 0.023202, 0.364994, 0.668009, 0.034634, 0.000000],
                      [0.972225, 0.583072, 0.441523, 0.685895, 0.173976, 0.258414, 0.532849, 0.402957, 0.884779, 0.036785],
                      [0.224569, 0.508845, 0.733631, 0.100618, 0.324195, 0.817603, 0.401993, 0.813790, 0.058533, 0.414680]])
        Y = np.array([[94.6127, 64.0122, 150.6100],
                      [67.1100, 461.983, 10.7289],
                      [264.4896, 189.1241, 12.9590],
                      [35.9969, 34.7454, 244.2714]])
        print(dataset.evaluate(X) - Y)

    if 2 in DTLZ_list:
        print('--- DTLZ 2 ---')
        dataset = DTLZ2(config)
        X = np.array([[0.382749, 0.291370, 0.443688, 0.552776, 0.541516, 0.370885, 0.550639, 0.498626, 0.469731, 0.326058],
                      [0.065842, 0.561275, 0.902678, 0.499258, 0.598058, 0.191232, 0.827697, 0.752817, 0.957646, 0.688103],
                      [0.975853, 0.372516, 0.696148, 0.436635, 0.486575, 0.401810, 0.919337, 0.140981, 0.420372, 0.312439],
                      [0.120113, 0.265794, 0.186567, 0.300821, 0.180685, 0.926552, 0.929635, 0.314517, 0.270314, 0.343584]])
        Y = np.array([[0.7827, 0.3856, 0.5985],
                      [1.0646, 1.2922, 0.1738],
                      [0.0442, 0.0293, 1.3976],
                      [1.5426, 0.6843, 0.3222]])
        print(dataset.evaluate(X) - Y)

    if 3 in DTLZ_list:
        print('--- DTLZ 3 ---')
        dataset = DTLZ3(config)
        X = np.array([[0.140860, 0.682954, 0.022691, 0.681591, 0.600793, 0.140003, 0.451558, 0.505652, 0.889364, 0.601127],
                      [0.926638, 0.291040, 0.814231, 0.274048, 0.404366, 0.796824, 0.321152, 0.486872, 0.406081, 0.147895],
                      [0.260076, 0.512463, 0.197811, 0.992959, 0.674890, 0.089451, 0.755151, 0.891655, 0.996912, 0.025496],
                      [0.609167, 0.020493, 0.350248, 0.098540, 0.519699, 0.344261, 0.136217, 0.484039, 0.659442, 0.151438]])
        Y = np.array([[284.9065, 523.9879, 134.1656],
                      [52.1880, 25.6724, 502.4673],
                      [361.9823, 376.4401, 226.0694],
                      [635.9824, 20.4800, 902.8666]])
        print(dataset.evaluate(X) - Y)

    if 4 in DTLZ_list:
        print('--- DTLZ 4 ---')
        dataset = DTLZ4(config)
        X = np.array([[0.549126, 0.422192, 0.515167, 0.521123, 0.525460, 0.583072, 0.424680, 0.524401, 0.590122, 0.571802],
                      [1.000000, 0.704675, 0.866934, 0.133731, 0.722335, 0.106823, 0.579466, 0.973384, 0.659218, 0.110497],
                      [0.711339, 0.323274, 0.770273, 0.876539, 0.977335, 0.291232, 0.150363, 0.959252, 0.875550, 0.901045],
                      [0.207247, 0.893407, 0.726502, 0.789443, 0.398448, 0.874622, 0.096373, 0.630363, 0.628557, 0.420668]])
        Y = np.array([[1.0278, 0.0000, 0.0000],
                      [0.0000, 0.0000, 1.8803],
                      [2.1213, 0.0000, 0.0000],
                      [1.4885, 0.0000, 0.0000]])
        print(dataset.evaluate(X) - Y)

    if 5 in DTLZ_list:
        print('--- DTLZ 5 ---')
        dataset = DTLZ5(config)
        X = np.array([[0.316455, 0.279264, 0.364729, 0.457624, 0.480323, 0.312941, 0.413002, 0.494280, 0.566832, 0.602372],
                      [0.103372, 0.989196, 0.256802, 1.000000, 0.976045, 0.288681, 0.618567, 0.922362, 0.629060, 0.618497],
                      [0.454071, 0.004363, 0.956548, 0.782571, 0.321298, 0.627768, 0.972044, 0.516396, 0.326187, 0.869503],
                      [0.212684, 0.068062, 0.726681, 0.829166, 0.433933, 0.532616, 0.190854, 0.650385, 0.040057, 0.004328]])
        Y = np.array([[0.6866, 0.6530, 0.5141],
                      [0.7630, 1.6080, 0.2916],
                      [1.1711, 0.5770, 1.1296],
                      [1.4456, 0.7836, 0.5707]])
        print(dataset.evaluate(X) - Y)

    if 6 in DTLZ_list:
        print('--- DTLZ 6 ---')
        dataset = DTLZ6(config)
        X = np.array([[0.757265, 0.149726, 0.491342, 0.001750, 0.737462, 0.488713, 0.519289, 0.256784, 0.001699, 0.453535],
                      [0.824463, 0.602376, 0.021473, 0.615806, 0.688604, 0.093949, 0.318423, 0.188075, 0.632497, 0.891327],
                      [0.625555, 0.333175, 0.326321, 0.226470, 0.774451, 0.208393, 0.635977, 0.662197, 0.579967, 0.420893],
                      [0.231500, 0.220361, 0.825897, 0.542459, 0.658747, 0.956540, 0.657261, 0.526902, 0.651792, 0.003690]])
        Y = np.array([[2.7041, 0.8583, 7.0767],
                      [1.3197, 1.7559, 7.7635],
                      [3.9455, 2.4444, 6.9594],
                      [7.1499, 3.0149, 2.9530]])
        print(dataset.evaluate(X) - Y)

    if 7 in DTLZ_list:
        print('--- DTLZ 7 ---')
        dataset = DTLZ7(config)
        X = np.array([[0.163301, 0.437073, 0.088336, 0.176334, 0.262780, 0.671290, 0.433185, 0.212621, 0.543164, 0.370274],
                      [0.303077, 0.141184, 0.310639, 0.026352, 0.041034, 0.234215, 0.231674, 0.830885, 0.990395, 0.918357],
                      [0.387029, 0.080809, 0.074084, 0.483671, 0.072289, 0.512406, 0.027153, 0.679924, 0.058713, 0.663367],
                      [0.928351, 0.792308, 0.861759, 0.911727, 0.504722, 0.844599, 0.896352, 0.766103, 0.744975, 0.497280]])
        Y = np.array([[0.1633, 0.4371, 14.9070],
                      [0.3031, 0.1412, 17.4278],
                      [0.3870, 0.0808, 14.3432],
                      [0.9284, 0.7923, 23.3081]])
        print(dataset.evaluate(X) - Y)

"""
DTLZ_list = [1, 2, 3, 4, 5, 6, 7]

DTLZ_validation(DTLZ_list)
#"""

