# -*- coding: UTF-8 -*-
from .boxmin import start, explore, move
from .corr import *
from .fit import fit
from .regr import *

""" Modified by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-13.
Modifications made:
1. function 'predict': rt = np.linalg.solve(self.model['C'], _R.T)
    Note: For improving the computational efficiency of prediction.
2. function 'boxmin': add a parameter 'max_iter'.  
    Note: Ordinal regression is not stable as fitness regression, so we need a flexible fitting setup.
3. function 'fit': self.theta = self.model['theta']
    Note: Update self.theta after training.
    
pydacefit source: https://github.com/msu-coinlab/pydacefit
"""


class DACE:

    def __init__(self, regr=regr_constant, corr=corr_gauss, theta=1.0, thetaL=0.0, thetaU=100.0):
        """

        This is the main object of this framework. It can be initialized with different regression and correlation
        types. Also, it can be defined if hyper parameter optimization should be used or not.


        Parameters
        ----------
        regr : callable
            Type of regression that should be used: regr_constant, regr_linear or regr_quadratic

        corr : callable
            Type of correlation (kernel) that should be used. default: corr_gauss

        theta : float
            Initial value of theta. Can be a vector or a float

        thetaL : float
            The lower bound if theta should be optimized.

        thetaU : float
            The upper bound if theta should be optimized.

        """

        super().__init__()
        self.regr = regr
        self.kernel = corr

        # most of the model will be stored here
        self.model = None

        # the hyperparameter can be defined
        self.theta = theta

        # lower and upper bound if it should be optimized
        self.tl = np.array(thetaL) if type(thetaL) == list else thetaL
        self.tu = np.array(thetaU) if type(thetaU) == list else thetaU

        # intermediate steps saved during hyperparameter optimization
        self.itpar = None

    def set_theta(self, theta):
        self.theta = theta

    def fit(self, X, Y, max_iter=4):

        # the targets should be a 2d array
        if len(Y.shape) == 1:
            Y = Y[:, None]

        # check if for each observation a target values exist
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must have the same number of rows.")

        # save the mean and standard deviation of the input
        mX, sX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
        sX[sX == 0] += 2.220446049250313e-16
        mY, sY = np.mean(Y, axis=0), np.std(Y, axis=0, ddof=1)

        # standardize the input
        nX = (X - mX) / sX
        nY = (Y - mY) / sY

        # check the hyperparamters
        if self.tl is not None and self.tu is not None:
            self.model = {'nX': nX, 'nY': nY}
            self.boxmin(max_iter)
            self.model = self.itpar["best"]
        else:
            self.model = fit(nX, nY, self.regr, self.kernel, self.theta)

        self.model = {**self.model, 'mX': mX, 'sX': sX, 'mY': mY, 'sY': sY, 'nX': nX, 'nY': nY}
        self.model['sigma2'] = np.square(sY) @ self.model['_sigma2']

        # update theta.
        self.theta = self.model['theta']

    def predict(self, _X, return_mse=False, return_gradient=False, return_mse_gradient=False):

        mX, sX, nX = self.model['mX'], self.model['sX'], self.model['nX']
        mY, sY = self.model['mY'], self.model['sY']
        regr, corr, theta = self.regr, self.kernel, self.model["theta"]
        beta, gamma = self.model['beta'], self.model['gamma']

        # normalize the input given the mX and sX that was fitted before
        # NOTE: For the values to predict the _ is added to clarify its not the data fitted before
        _nX = (_X - mX) / sX

        # calculate regression and corr
        _F = regr(_nX)
        _R = calc_kernel_matrix(_nX, nX, corr, theta)

        # predict and destandardize
        _sY = _F @ beta + (gamma.T @ _R.T).T  # F @ beta + r @ gamma
        _Y = (_sY * sY) + mY

        ret = [_Y]

        if return_mse:

            #rt = np.linalg.lstsq(self.model['C'], _R.T, rcond=None)[0]
            rt = np.linalg.solve(self.model['C'], _R.T)
            u = (self.model["Ft"].T @ rt).T - _F
            v = u @ np.linalg.inv(self.model["G"])
            mse = self.model["sigma2"] * (1 + np.sum(v ** 2, axis=1) - np.sum(rt ** 2, axis=0))
            ret.append(mse[:, None])

        if return_gradient:

            # the final gradient matrix
            _grad = np.zeros(_X.shape)

            # the gradient must be calculated for each point at once
            for i, _x in enumerate(_nX):
                _dF = get_gradient_func(self.regr)(_x[None, :])
                _dR = calc_grad(_x[None, :], nX, get_gradient_func(corr), theta)

                dy = (_dF @ self.model["beta"]).T + self.model["gamma"].T @ _dR
                _grad[i] = dy * self.model["sY"] / self.model["sX"]

            ret.append(_grad)

        if return_mse_gradient:

            if not return_mse or not return_gradient:
                raise Exception("To evaluate the gradient of MSE, you must calculate the gradient and MSE as well.")

            # the final gradient matrix
            _mse_grad = np.zeros(_X.shape)

            # the gradient must be calculated for each point at once
            for i, _x in enumerate(_nX):

                # is not implemented yet - here precision problems started to occur and results did not match
                _mse_grad[i] = np.nan

            ret.append(_mse_grad)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def boxmin(self, max_iter=4):

        itpar = start(self.theta, self)
        model = itpar["models"][-1]
        p, f = itpar["p"], model["f"]

        kmax = 2 if p <= 2 else min(p, max_iter)

        # if the initial guess is feasible
        if not np.isinf(f):

            for k in range(kmax):
                # save the last theta before exploring
                last_t = itpar["best"]["theta"]

                # do the actual explore step
                explore(self, itpar)
                move(last_t, self, itpar)

        self.itpar = itpar


def get_gradient_func(func):
    try:

        return globals()[func.__name__ + "_grad"]
    except:
        return None
