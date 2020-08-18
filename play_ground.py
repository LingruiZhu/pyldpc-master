import numpy as np
import numpy.random as rnd
from scipy import stats

rng = rnd.default_rng(0)
x = np.ones([30, 50])

mu, sigma = 0, 1
mu_biased, sigma_biased = 0.5, 1

noise = rng.normal(mu, sigma, x.shape)
print(noise)

bias_bits = np.array([0, 5, 10, 15, 20, 25])

n, n_trials = x.shape

for i in range(len(bias_bits)):
    for j in range(n_trials):
        noise[i, j] = rng.normal(mu_biased, sigma_biased, 1)

weights = (-1) * np.ones([n_trials])
for j in range(n_trials):
    weight_temp = 1
    for i in range(len(bias_bits)):
        lr = stats.norm.pdf(noise[i,j], mu, sigma) / stats.norm.pdf(noise[i,j], mu_biased, sigma_biased)
        weight_temp = weight_temp * lr
    weights[j] = weight_temp
print(weights)
