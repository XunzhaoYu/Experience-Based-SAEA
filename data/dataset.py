import numpy as np


class Dataset:
    def __init__(self):
        pass

    # draw n_sample (x,y) pairs drawn from n_func functions
    # returns (x,y) where each has size [n_func, n_samples, x/y_dim]
    def sample(self, n_funcs, n_samples):
        raise NotImplementedError


class SinusoidDataset(Dataset):
    def __init__(self, config, noise_var=None, rng=None):
        self.amp_range = config['amp_range']
        self.phase_range = config['phase_range']
        self.freq_range = config['freq_range']
        self.x_range = config['x_range']
        if noise_var is None:
            self.noise_std = np.sqrt(config['sigma_eps'])
        else:
            self.noise_std = np.sqrt(noise_var)

        self.np_random = rng
        if rng is None:
            self.np_random = np.random

    # draw n_samples points from n_funcs different 1D sinusoid functions.
    def sample(self, n_funcs, n_samples, return_lists=False):
        x_dim = 1
        y_dim = 1
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        amp_list = self.amp_range[0] + self.np_random.rand(n_funcs) * (self.amp_range[1] - self.amp_range[0])
        phase_list = self.phase_range[0] + self.np_random.rand(n_funcs) * (self.phase_range[1] - self.phase_range[0])
        freq_list = self.freq_range[0] + self.np_random.rand(n_funcs) * (self.freq_range[1] - self.freq_range[0])
        for i in range(n_funcs):
            x_samp = self.x_range[0] + self.np_random.rand(n_samples)*(self.x_range[1] - self.x_range[0])
            y_samp = amp_list[i] * np.sin(freq_list[i] * x_samp + phase_list[i]) + self.noise_std * self.np_random.randn(n_samples)

            x[i, :, 0] = x_samp
            y[i, :, 0] = y_samp

        if return_lists:
            return x, y, freq_list, amp_list, phase_list

        return x, y

