# Closed-form metrics when scores are integer multiples

import os
import random
from math import sqrt
from itertools import product
import numpy as np
from numpy.polynomial import Polynomial
from numpy.linalg import det, slogdet
from scipy.sparse.linalg import expm_multiply
from multiprocessing import Pool
from qiskit.quantum_info import random_hermitian, Statevector
import matplotlib.pyplot as plt
from coloraide import Color

import utils
from utils import MyStateVector, get_rydberg_hamiltonian, get_random_clifford_product_state
from Certification import cert_prob
from cusum import _set_random


def sprt_int(p, g1, g2, a, b):
    ''' Sequential Probability Ratio Test (integer multiple scores).
    Formulats from Sequential Analysis by Abraham Wald, Appendix A.4.
    p:      actual rejection probability
    g1,g2:  rescaled scores; -g1 on acceptance, +g2 on rejection
    a,b:    terminate when score <= a or >= b
    '''
    if g1 == 1 and g2 == 2: # shortcut for efficiency
        v = sqrt(-p*(3*p-4))
        u = [1,(-p-v)/(2*p),(-p+v)/(2*p)]
    else:
        u = Polynomial([1-p]+[0]*(g1-1)+[-1]+[0]*(g2-1)+[p]).roots()
    c = list(range(a-g1+1,a+1)) + list(range(b,b+g2))
    D = np.array(u)[:, np.newaxis] ** np.array(c)[np.newaxis, :]
    #detD = det(D)
    _, logdetD = slogdet(D)
    P = N = 0
    for j in range(g1+g2):
        Dj = D.copy()
        Dj[:,j] = 1
        #d = det(Dj) / detD
        d = np.exp(slogdet(Dj)[1] - logdetD)
        if j < g1: P += d 
        N += c[j]*d
    return P, N / (p*g2-(1-p)*g1)

def cusum_int(p, g1, g2, h):
    ''' CUSUM test (integer multiple scores).
    Formula from Eq. (9) of [Page, 1954].
    p:      actual rejection probability
    g1,g2:  rescaled scores; -g1 on acceptance, +g2 on rejection
    h:      terminate when score >= h
    '''
    P, N = sprt_int(p, g1, g2, -1, h)
    if P == 1: return np.nan
    L = N / (1-P)
    return L if L < 2e13 else np.nan

def plot_arl(ax):
    plim = [0.1, 0.6]
    num = 101
    hlim = [2, 52]
    hnum = 11
    hcol = [(1,0,0), (0,0,1)]
    htext = {2:(0.1,1.8), 7:(0.1,1.3e2), 52:(0.54,1.2e2)}

    # Series
    interp = Color.interpolate([Color('srgb', col) for col in hcol], space="lab")
    ps = [plim[0]+i*(plim[1]-plim[0])/(num-1) for i in range(num)]
    for i in range(hnum):
        h = int(hlim[0]+i*(hlim[1]-hlim[0])/(hnum-1))
        #col = tuple(((hnum-i-1)*h0+i*h1)/(hnum-1) for h0,h1 in zip(*hcol))
        col = interp(i/(hnum-1)).convert('srgb').fit('srgb').coords(precision=3)
        ax.plot(ps, list(map(lambda p: cusum_int(p, 1, 2, h), ps)), color=col)
        if h in htext.keys():
            ax.text(*htext[h], rf'$h={h}$', color=col)

    # Axes
    ax.set_xlabel('Rejection probability')
    ax.set_ylabel('Average run length')
    ax.set_yscale('log')

    # Details
    for c, v in [('p',(3-sqrt(5))/4), ('q',1/2)]:
        ax.axvline(x=v, color='k', linestyle='--')
        eq = r'\approx' if 100*v%1 else '='
        ax.text(v+0.01, 5e12, rf"${c}{eq}{v:.2}$", color='k')

def _run_arl_ham(seed, H0, n, num_tri, nms, h, tau):
    _set_random(seed)
    num = len(nms)
    d = 1<<n
    arl = np.zeros((num_tri, num))
    for i in range(num_tri):
        dH = random_hermitian(d, seed=utils.rng).data.astype(np.complex128)
        dH /= (np.linalg.norm(dH) / sqrt(d))

        # Series
        for j, nm in enumerate(nms): 
            for s in product('01+-lr', repeat=n):
                psi0 = Statevector.from_label(''.join(s)).data
                hyp = expm_multiply(-1j * tau * H0, psi0)
                lab = expm_multiply(-1j * tau * (H0+nm*dH), psi0)
                p = 1-cert_prob(
                    MyStateVector(sv=Statevector(hyp)),
                    MyStateVector(sv=Statevector(lab)))
                arl[i,j] += cusum_int(p, 1, 2, h)
    return arl

def plot_arl_ham(ax, n):
    _set_random(42)
 
    num = 8
    num_tri = 10
    j = 12
    h = 3
    tau = 0.1

    H0 = get_rydberg_hamiltonian(n, Omega=1, Delta=2.5, rb=1.5, a=1).astype(np.complex128)
    nms = np.linspace(0.15, 0.5, num)
    arl = np.concatenate(Pool(j).starmap(_run_arl_ham, [(42+i, H0, n, num_tri, nms, h, tau) for i in range(j)]))
    arl = tau*arl / 6**n
    med, l, u = np.median(arl,axis=0), np.percentile(arl, 2.5, axis=0), np.percentile(arl, 97.5, axis=0)
    eb = ax.errorbar(nms, med, yerr=(l, u), fmt='o-', capsize=3, color=plt.cm.viridis(0))
    
    # Axes
    ax.set_xlabel(r'$\|H-H_0\|_F$')
    ax.set_ylabel('Expected total evolution time')
    ax.set_yscale('log')
    ax.legend([eb[0]], [f"{n} qubits"])

if __name__ == "__main__":
    figs_dir = "figs"
    os.makedirs(figs_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_arl(axes[0])
    plot_arl_ham(axes[1], 3)
    plt.savefig(os.path.join(figs_dir, "cusum_arl.pdf"))
    plt.show()
