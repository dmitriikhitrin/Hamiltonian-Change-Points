import sys, time
import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
from qiskit.quantum_info import random_hermitian, Statevector, random_clifford
import matplotlib.pyplot as plt
from typing import Optional

from Certification import MyStateVector, certify

def get_rydberg_hamiltonian(num_qubits, Omega, Delta, rb, a=1):
    N, d = num_qubits, 1 << num_qubits
    I = np.eye(2); X = np.array([[0,1],[1,0]]); Z = np.array([[1,0],[0,-1]]); Nop = (I - Z)/2
    def one(op,i):
        M = [I]*N; M[i] = op; out = M[0]
        for m in M[1:]: out = np.kron(out, m)
        return out
    def two(opA,i,opB,j):
        M = [I]*N; M[i], M[j] = opA, opB; out = M[0]
        for m in M[1:]: out = np.kron(out, m)
        return out
    H = np.zeros((d,d))
    for i in range(N): H += (Omega/2)*one(X,i) - Delta*one(Nop,i)
    for i in range(N):
        for j in range(i+1,N):
            H += Omega*(rb/(a*abs(i-j)))**6*two(Nop,i,Nop,j)
    return H

def get_random_product_state(num_qubits):
    state = 1
    for _ in range(num_qubits):
        v = np.random.randn(2) + 1j*np.random.randn(2)
        v /= np.linalg.norm(v)
        state = np.kron(state, v)
    return state

def get_random_clifford_product_state(num_qubits):
    state = np.array([1.0+0.0j])
    for _ in range(num_qubits):
        cliff = random_clifford(1)
        psi = Statevector.from_label('0').evolve(cliff).data
        state = np.kron(state, psi.astype(np.complex128, copy=False))
    return state.astype(np.complex128, copy=False)

class Experiment:
    def __init__(self,
                 num_qubits: int,
                 num_runs: int,
                 num_norms: int,
                 perturbation_mode: str,            # "per_run" or "per_qubits"
                 psi0_type: str = "random",         # "random" or "clifford"
                 rng_seed: Optional[int] = None):
        if perturbation_mode not in {"per_run", "per_qubits"}:
            raise ValueError("perturbation_mode must be 'per_run' or 'per_qubits'")
        if psi0_type not in {"random", "clifford"}:
            raise ValueError("psi0_type must be 'random' or 'clifford'")

        self.num_qubits = num_qubits
        self.num_runs = num_runs
        self.num_norms = num_norms
        self.perturbation_mode = perturbation_mode
        self.psi0_type = psi0_type
        self.rng = np.random.default_rng(rng_seed)

        # these will be populated by run()
        self.norms = None
        self.results = None
        self.fidelities = None
        self.meta = {}

    def _draw_psi0(self):
        if self.psi0_type == "random":
            return get_random_product_state(self.num_qubits, rng=self.rng)
        else:
            return get_random_clifford_product_state(self.num_qubits, rng=self.rng)

    def run(self,
            Omega: float,
            Delta: float,
            rb: float,
            tau: float,
            a: floats):
        """
        Executes the experiment with current settings.
        - Keeps the original semantics.
        - For 'per_qubits', precomputes U=e^{-i tau H} and U_lab for each norm for speed.
        """
        # Build Hamiltonian and constants
        H = get_rydberg_hamiltonian(self.num_qubits, Omega, Delta, rb, a).astype(np.complex128)
        dim = 2 ** self.num_qubits
        H_norm = np.linalg.norm(H)
        A_hyp = (-1j * tau * H).astype(np.complex128)

        # One fixed perturbation if per_qubits
        if self.perturbation_mode == "per_qubits":
            P_fixed = random_hermitian(dim).data.astype(np.complex128)
            P_fixed /= np.linalg.norm(P_fixed)

        # Norm sweep
        self.norms = np.linspace(-0.25 * H_norm, 0.25 * H_norm, self.num_norms)
        self.results = {float(norm): [] for norm in self.norms}
        self.fidelities = {float(norm): [] for norm in self.norms}

        total_jobs = self.num_norms * self.num_runs
        print(f"\nRunning for {self.num_qubits} qubits")
        print(f"Total runs: {total_jobs}")
        start = time.time()

        # Precompute unitaries if per_qubits
        U_hyp = None
        U_labs = None
        if self.perturbation_mode == "per_qubits":
            U_hyp = expm(A_hyp)  # (dim, dim)
            U_labs = {}
            for norm in self.norms:
                A_lab = (-1j * tau * (H + norm * P_fixed)).astype(np.complex128)
                U_labs[float(norm)] = expm(A_lab)

        for i, norm in enumerate(self.norms, 1):
            norm_f = float(norm)
            for run in range(1, self.num_runs + 1):
                psi0 = self._draw_psi0()

                if self.perturbation_mode == "per_run":
                    # new perturbation each run
                    P = random_hermitian(dim).data.astype(np.complex128)
                    P /= np.linalg.norm(P)
                    
                    A_lab = (-1j * tau * (H + norm * P)).astype(np.complex128)
                    hyp = expm_multiply(A_hyp, psi0)   # e^{-i tau H} |psi0>
                    lab = expm_multiply(A_lab, psi0)   # e^{-i tau (H + norm P)} |psi0>
                else:
                    # reuse precomputed unitaries
                    hyp = U_hyp @ psi0
                    lab = U_labs[norm_f] @ psi0

                # certify + fidelity
                cert_val = certify(MyStateVector(sv=Statevector(hyp)),
                                   MyStateVector(sv=Statevector(lab)))
                fid_val = np.abs(np.vdot(hyp, lab))**2

                self.results[norm_f].append(float(cert_val))
                self.fidelities[norm_f].append(float(fid_val))

                if run % 10 == 0 or run == self.num_runs:
                    elapsed = time.time() - start
                    done = (i - 1) * self.num_runs + run
                    msg = (f"Qubits: {self.num_qubits} | Norm {i}/{self.num_norms} | "
                           f"Run {run}/{self.num_runs} ({done / total_jobs * 100:.1f}%)  "
                           f"Elapsed: {elapsed:.1f}s")
                    sys.stdout.write("\r" + " " * 120 + "\r")
                    sys.stdout.write(msg)
                    sys.stdout.flush()

        sys.stdout.write("\r" + " " * 120 + "\r")
        wall = (time.time() - start) / 60.0
        print(f"Finished {self.num_qubits}-qubit run in {wall:.2f} min.")

        self.meta = dict(
            num_qubits=self.num_qubits,
            num_runs=self.num_runs,
            num_norms=self.num_norms,
            perturbation_mode=self.perturbation_mode,
            psi0_type=self.psi0_type,
            tau=tau,
            Omega=Omega,
            Delta=Delta,
            rb=rb,
            a=a,
            use_fro_norm=use_fro_norm,
            H_norm=float(H_norm),
            dim=dim,
            wall_minutes=wall
        )
        return self.results, self.fidelities

class Plotter:
    def __init__(self, experiments):
        """
        experiments : list[Experiment]
        """
        self.experiments = experiments
        self._collect()

    def _collect(self):
        """Collect all results and metadata for convenience."""
        self.all_results = {exp.num_qubits: exp.results for exp in self.experiments}
        self.all_fidelities = {exp.num_qubits: exp.fidelities for exp in self.experiments}
        # assume consistent mode/tau across runs
        first = self.experiments[0].meta if self.experiments else {}
        self.perturbation_mode = first.get("perturbation_mode", "?")
        self.psi0_type = first.get("psi0_type", "?")
        self.tau = first.get("tau", "?")

    def plot(self):
        """Plot (1) acceptance vs fidelity and (2) rejection vs scaled infidelity side by side."""
        colors = plt.cm.viridis(np.linspace(0, 0.75, len(self.all_results)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = axes

        # --- Left panel: acceptance & fidelity ---
        for color, num_qubits in zip(colors, self.all_results.keys()):
            cert_results = self.all_results[num_qubits]
            fid_results = self.all_fidelities[num_qubits]

            xs, ps, errs = [], [], []
            xf, pf, erf = [], [], []

            for n in sorted(cert_results.keys()):
                cert_arr = np.asarray(cert_results[n], float)
                fid_arr = np.asarray(fid_results[n], float)

                xs.append(n)
                ps.append(cert_arr.mean())
                errs.append(cert_arr.std(ddof=1) / np.sqrt(len(cert_arr)))

                xf.append(n)
                pf.append(fid_arr.mean())
                erf.append(fid_arr.std(ddof=1) / np.sqrt(len(fid_arr)))

            ax1.errorbar(xs, ps, yerr=errs, fmt='o-', capsize=3, color=color,
                         label=f"{num_qubits} qubits (cert.)")
            ax1.errorbar(xf, pf, yerr=erf, fmt='--', capsize=3, color=color, alpha=0.6,
                         label=f"{num_qubits} qubits (fidelity)")

        ax1.set_xlabel(r"$\| \mathrm{perturbation} \|_F$")
        ax1.set_ylabel("Value")
        ax1.set_title(f"P(Accept) and fidelity\nMode={self.perturbation_mode, } ($\\tau$={self.tau})")
        ax1.legend(ncol=1)
        ax1.grid(True, ls=":")

        # --- Right panel: rejection & scaled infidelity ---
        for color, num_qubits in zip(colors, self.all_results.keys()):
            cert_results = self.all_results[num_qubits]
            fid_results = self.all_fidelities[num_qubits]

            xs, rej_ps, rej_errs = [], [], []
            xf, inf_ps, inf_errs = [], [], []

            for n in sorted(cert_results.keys()):
                cert_arr = np.asarray(cert_results[n], float)
                fid_arr = np.asarray(fid_results[n], float)

                rej = 1 - cert_arr
                xs.append(n)
                rej_ps.append(rej.mean())
                rej_errs.append(rej.std(ddof=1) / np.sqrt(len(rej)))

                inf = (1 - fid_arr) / num_qubits
                xf.append(n)
                inf_ps.append(inf.mean())
                inf_errs.append(inf.std(ddof=1) / np.sqrt(len(inf)))

            ax2.errorbar(xs, rej_ps, yerr=rej_errs, fmt='o-', capsize=3,
                         color=color, label=f"{num_qubits} qubits (rejection)")
            ax2.errorbar(xf, inf_ps, yerr=inf_errs, fmt='--', capsize=3,
                         color=color, alpha=0.6, label=f"{num_qubits} qubits (scaled infid.)")

        ax2.set_xlabel(r"$\| \mathrm{perturbation} \|_F$")
        ax2.set_ylabel("Value")
        ax2.set_title(f"P(Reject) and (1-F)/n\nMode={self.perturbation_mode, self.psi0_type} ($\\tau$={self.tau})")
        ax2.legend(ncol=1)
        ax2.grid(True, ls=":")

        plt.tight_layout()
        plt.show()