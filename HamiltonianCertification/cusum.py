import random
import os
import numpy as np
from math import log, inf, sqrt
from itertools import count, islice
from functools import partial
from multiprocessing import Pool
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from numpy.linalg import norm
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_hermitian, Statevector
from utils import get_rydberg_hamiltonian, get_random_clifford_product_state
import utils
from Certification import MyStateVector, cert_prob


def rydberg_sequence(n, nu, nt, d, **param):
    ''' Sequence of gradually changing Rydberg Hamiltonians.
    n:      number of qubits
    nu:     changepoint location
    param:  parameters for the Rydberg Hamiltonian
    '''
    H = get_rydberg_hamiltonian(n, **param)
    yield H
    for t in count():
        diff = d[1] if (t >= nu and t < nu+nt) else d[0]
        yield (H := H + diff * random_hermitian(1<<n, seed=utils.rng).data)

def cusum(seq, h, p, q, shots=1):
    ''' Run CUSUM algorithm.
    seq:    sequence of Booleans
    p, q:   rejection probabilities
    h:      termination threshold
    '''
    z = [log((1-q)/(1-p)),log(q/p)]
    Z = 0
    while Z < h:
        try:
            x = int(next(seq))
        except StopIteration:
            return
        yield (Z := max(0, Z + x*z[1]+(shots-x)*z[0]))

class Changepoint:
    def __init__(self, n, tau, H0):
        self.n = n
        self.tau = tau
        self.H0 = H0

    def test(self, H, shots=1):

        # Create hypothesis and lab states
        psi0 = get_random_clifford_product_state(self.n)
        hyp = expm_multiply(-1j * self.tau * self.H0, psi0)
        lab = expm_multiply(-1j * self.tau * H, psi0)

        return np.random.binomial(shots, 1-cert_prob(
            MyStateVector(sv=Statevector(hyp)),
            MyStateVector(sv=Statevector(lab))))

def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    utils.rng = np.random.default_rng(seed)

def _run_cusum(seed, num_tri, cpt, H_seq, xi, n, s):
    _set_random(seed)
    return list(
        list(cusum(map(partial(cpt.test, shots=s), H_seq), inf, xi/(2*n), xi/n, shots=s))
        for _ in range(num_tri))

def plot_example(ax, nu, cid):
    _set_random(42)

    col_nms = 'blue'
    col_val = 'red'

    seq_len = 200
    num_tri = 10
    j = 12
    tau = 0.1

    n = [1, 3, 5][cid]
    s = [1000, 100, 100][cid]
    xi = [5e-5, 2e-3, 2e-3][cid]
    d = [(2e-2, 2e-1), (1e-2, 1e-1), (5e-3, 5e-2)][cid]
    nmx = [1, 0.8, 0.8][cid]
    vmx = [14, 14, 14][cid]

    H_seq = islice(rydberg_sequence(n, nu, 10, d, Omega=1, Delta=2.5, rb=1.5, a=1), seq_len+1)
    H0 = next(H_seq)
    H_seq = list(H_seq)
    cpt = Changepoint(n, tau, H0)
    val = np.concatenate(Pool(j).starmap(_run_cusum, [(42+i, num_tri, cpt, H_seq, xi, n, s) for i in range(j)]))

    # X axes
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel('Total evolution time')
    ax.text(1.05*seq_len*tau, -0.1075*nmx, f'x{s}')

    # Y axes
    lim = lambda x: (-0.05*x, 1.05*x)
    ax.set_ylabel(r'$\|H_i - H_0\|_F$', color=col_nms)
    ax.tick_params(axis='y', labelcolor=col_nms)
    ax.set_ylim(lim(nmx))
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$s_i$', color=col_val)
    ax2.tick_params(axis='y', labelcolor=col_val)
    ax2.set_ylim(lim(vmx))
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))

    # Series
    nms = list(map(lambda H: norm(H-H0, 'fro').item() / sqrt(1<<n), H_seq))
    stp = [tau*(i+1) for i in range(seq_len)]
    plt_nms, = ax.plot(stp, nms, color=col_nms, linestyle='dotted')
    plt_val, = ax2.plot(stp, np.median(val,axis=0), color=col_val)
    ax2.fill_between(stp, np.percentile(val, 2.5, axis=0), np.percentile(val, 97.5, axis=0), color=col_val, alpha=0.1)
    ax.legend([plt_nms, plt_val], ['Frobenius norm', 'Cumulative value'], loc='upper left')

    # Additional details
    if nu < inf:
        ax.axvspan(tau*nu, tau*(nu+10), color='orange', alpha=0.25)
        ax.text(1.015*tau*nu, 11/14*nmx, 'Changepoint', rotation=90, color='chocolate')

if __name__ == "__main__":
    figs_dir = "figs"
    os.makedirs(figs_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.3)
    for ax, nu in zip(axes, [100, inf]):
        plot_example(ax, nu, 1)
    plt.savefig(os.path.join(figs_dir, "cusum_example.pdf"))
    plt.show()
