"""
================================================
Coding - Decoding simulation of a random message
================================================

This example shows a simulation of the transmission of a binary message
through a gaussian white noise channel with an LDPC coding and decoding system.
"""


import numpy as np
from pyldpc import make_ldpc, decode, get_message, encode, channel, evaluation
from pyldpc import code
from matplotlib import pyplot as plt

n = 30     # default 30
d_v = 2    # default 2
d_c = 3    # default 3
seed = np.random.RandomState(42)

##################################################################
# First we create an LDPC code i.e a pair of decoding and coding matrices
# H and G. H is a regular parity-check matrix with d_v ones per row
# and d_c ones per column

# construct H and G from make ldpc
H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
print(H)


# get H G from txt file
# H, G = code.get_matrix(N=155, K=93)

n, k = G.shape

print(G.shape)
print(H.shape)
print("Number of coded bits:", k)
input('press enter to continue')

##################################################################
# Now we simulate transmission for different levels of noise and
# compute the percentage of errors using the bit-error-rate score
# The coding and decoding can be done in parallel by column stacking.

bers = []
fers = []
snrs = np.linspace(-2, 20, 20)
# snrs = np.linspace(5,10,5)

# v = np.arange(k) % 2  # fixed k bits message
v = np.zeros(k)

n_trials = 50  # number of transmissions with different noise
V = np.tile(v, (n_trials, 1)).T  # stack v in columns

for snr in snrs:
    x = encode(G, V, snr, add_noise=True, seed=seed)
    y, weights = channel.AWGN_IS(x, snr, seed)
    D = decode(H, y, snr)

    ber = 0.
    # fer = 0.
    for i in range(n_trials):
        x_hat = get_message(G, D[:, i])
        ber += abs(v - x_hat).sum() / (k * n_trials)
    #    if abs(v - x).sum() != 0:
    #        fer += 1 / n_trials
    fer = evaluation.fer_IS(D, x, weights)
    bers.append(ber)
    fers.append(fer)

print(fers)
print(bers)

plt.figure()
plt.semilogy(snrs, bers, color="indianred")
plt.ylabel("Bit error rate")
plt.xlabel("SNR")
plt.show()

plt.figure()
plt.semilogy(snrs, fers, color="indianred")
plt.ylabel("frame error rate")
plt.xlabel("SNR")
plt.show()
