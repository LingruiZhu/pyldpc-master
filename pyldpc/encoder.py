import numpy as np
from . import utils
import numpy.random as rnd
from scipy import stats


def encode_random_message(tG, snr, seed=None):
    """Encode a random message given a generating matrix tG and a SNR.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.
    snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.

    Returns
    -------
    v: array (k,) random message generated.
    y: array (n,) coded message + noise.

    """
    rng = utils.check_random_state(seed)
    n, k = tG.shape

    v = rng.randint(2, size=k)

    d = utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (-snr / 20)

    e = rng.randn(n) * sigma

    y = x + e

    return v, y


def encode(tG, v, snr, add_noise = False,seed=None):
    """Encode a binary message and adds Gaussian noise.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.

    v: array (k, ) or (k, n_messages) binary messages to be encoded.

    snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.

    Returns
    -------
    y: array (n,) or (n, n_messages) coded messages + noise.

    """
    n, k = tG.shape

    rng = utils.check_random_state(seed)
    d = utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (- snr / 20)
    e = rng.randn(*x.shape) * sigma

    y = x + e

    if add_noise:
        return y
    else:
        return x


def channel_IS(x, snr, seed=None):
    rng = rnd.default_rng(0)
    noise_sigma = 10**(-snr/20)
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
