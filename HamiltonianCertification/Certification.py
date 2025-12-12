import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, Operator
from qiskit.quantum_info import partial_trace, random_statevector

projector_dict = {'0' : np.array([[1, 0], [0, 0]]), '1' : np.array([[0, 0], [0, 1]])}

def normalize(sv):
    '''returns normalized qiskit Statevector'''
    return sv / np.linalg.norm(sv)

def basis_to_unitary(b, b_perp):
    b /= np.linalg.norm(b)
    b_perp /= np.linalg.norm(b_perp)
    return Operator(np.column_stack((b, b_perp)))

def dm_to_sv(pure_dm):
    '''
    returns Statevector of a pure DensityMatrix
    '''
    vals, vecs = np.linalg.eigh(pure_dm.data if hasattr(pure_dm,"data") else pure_dm)
    return normalize(Statevector(vecs[:,np.argmax(vals)]))

def get_orth_basis(rho0, rho1):
    '''returns phase basis'''
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.array([[1,0],[0,-1]],complex)
    def bloch(r): M=r.data if hasattr(r,'data') else np.asarray(r,complex); return np.real([np.trace(M@X),np.trace(M@Y),np.trace(M@Z)])
    r0,r1=bloch(rho0),bloch(rho1); e=np.cross(r0,r1)
    if np.linalg.norm(e)<1e-12:
        v=r0 if np.linalg.norm(r0)>1e-12 else r1
        if np.linalg.norm(v)<1e-12: e=np.array([1,0,0.])
        else:
            e=np.cross(v,[1,0,0.]); 
            if np.linalg.norm(e)<1e-12: e=np.cross(v,[0,1,0.])
    e/=np.linalg.norm(e)
    H=e[0]*X+e[1]*Y+e[2]*Z
    w,V=np.linalg.eigh(H); i=np.argmax(w)
    return V[:,i], V[:,1-i]

class MyStateVector:
    def __init__(self, sv=None):
        self.data = sv.data
        self.sv = sv
        self.num_qubits = sv.num_qubits
    
    @staticmethod
    def from_label(label):
        return MyStateVector(sv=Statevector.from_label(label))
    
    def evolve(self, qc):
        return MyStateVector(sv=self.sv.evolve(qc))
    
    def measure_in_basis(self, basis, qubits):
        assert len(basis) == len(qubits)
        n = self.num_qubits
        qc = QuantumCircuit(n)
        
        for q, (b, b_perp) in zip(qubits, basis):
            qc.append(basis_to_unitary(b, b_perp), [n - 1 - q])
            
        transformed_sv = self.sv.copy().evolve(qc)
        meas = list(transformed_sv.sample_counts(1, qargs=np.array(n-1)-np.array(qubits)).keys())[0][::-1]
        sampled_states = [b[int(bit)] for b, bit in zip(basis, meas)]
        
        return sampled_states

    def get_conditioned_state(self, projectors):
        '''
        returns (n-a)-qubit state where a is the number of active qubits (on which projector is not I)
        '''
        active_qubits = []
        proj = np.array(1)
        for q, p in enumerate(projectors):
            proj = np.kron(proj, p)
            if not np.allclose(p, np.eye(2)):
                active_qubits.append(self.num_qubits - 1 - q)
                
        post_meas = proj @ self.data
        post_meas = post_meas / np.linalg.norm(post_meas)
        dm = partial_trace(post_meas, active_qubits)
        post_meas_reduced = dm_to_sv(dm)
        return MyStateVector(sv=post_meas_reduced)
    
    def get_dt_basis(self, x=''):
        '''
        len(x) = k - 1
        constructs tuples (b, b_perp)
        
        TODO: handle the cases when hyp_x0 or hyp_x1 = 0
        '''
        k = len(x) + 1
        n = self.num_qubits
        projectors0 = [projector_dict[bit] for bit in x + '0'] + [np.eye(2) for _ in range(n - len(x) - 1)]
        hyp_x0 = self.get_conditioned_state(projectors0)
        projectors1 = [projector_dict[bit] for bit in x + '1'] + [np.eye(2) for _ in range(n - len(x) - 1)]
        hyp_x1 = self.get_conditioned_state(projectors1)
        
        projectors = [np.eye(2) for _ in range(hyp_x0.num_qubits)]
        hyp_x0_t = hyp_x0
        hyp_x1_t = hyp_x1
        dt_basis = []
        for t in range(n-k):
            hyp_x0_t = hyp_x0_t.get_conditioned_state(projectors)
            hyp_x1_t = hyp_x1_t.get_conditioned_state(projectors)
            
            rho_0_t = partial_trace(hyp_x0_t.sv, range(hyp_x0_t.num_qubits - 1))
            rho_1_t = partial_trace(hyp_x1_t.sv, range(hyp_x1_t.num_qubits - 1))
            
            basis = get_orth_basis(rho_0_t, rho_1_t)
            meas = basis[np.random.choice([0, 1])]
            projectors = [np.outer(meas, meas.conj())] + [np.eye(2) for _ in range(hyp_x0_t.num_qubits - 1)]
            dt_basis.append(basis)
        return dt_basis
    
def certify(hyp, lab):
    n = hyp.num_qubits
    k = np.random.choice(list(range(1, n)))
    #print(k)
    if k == 1:
        x = ''
    else:
        x = list(lab.sv.sample_counts(1, qargs=list(range(n-1, n-k, -1))).keys())[0][::-1]
    project_x = [projector_dict[bit] for bit in x]
    
    dt_basis = hyp.get_dt_basis(x)
    lab_x = lab.get_conditioned_state(project_x + [np.eye(2) for _ in range(n-k+1)])
    leaf = lab_x.measure_in_basis(dt_basis, list(range(1, lab_x.num_qubits)))
    l_projectors = [np.outer(l, l.conj()) for l in leaf]
    
    hyp_prime = hyp.get_conditioned_state(project_x + [np.eye(2)] + l_projectors)
    lab_prime = lab.get_conditioned_state(project_x + [np.eye(2)] + l_projectors)
    accept_prob = np.abs(np.inner(hyp_prime.data.conj(), lab_prime.data))**2
    return np.random.random() < accept_prob
    