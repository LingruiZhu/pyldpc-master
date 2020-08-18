import numpy.random as rnd
from scipy import stats
import numpy as np


def AWGN_IS(x, snr, seed=None):
    rng = rnd.default_rng(0)
    noise_sigma = 10 ** (-snr / 20)
    n, n_trials = x.shape

    mu, sigma = 0, noise_sigma
    mu_biased, sigma_biased = 0.5, noise_sigma

    noise = np.zeros(x.shape, dtype=float)

    # sample noise (from original pdf)
    noise = rng.normal(mu, sigma, x.shape)
    bias_bits = np.array([0, 5, 10, 15, 20, 25])

    for i in range(len(bias_bits)):
        for j in range(n_trials):
            noise[i, j] = rng.normal(mu_biased, sigma_biased, 1)

    weights = (-1) * np.ones([n_trials])
    for j in range(n_trials):
        weight_temp = 1
        for i in range(len(bias_bits)):
            lr = stats.norm.pdf(noise[i, j], mu, sigma) / stats.norm.pdf(noise[i, j], mu_biased, sigma_biased)
            weight_temp = weight_temp * lr
        weights[j] = weight_temp

    y = x + noise
    return y, weights