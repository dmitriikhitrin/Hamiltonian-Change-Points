import pytest
import numpy as np
import matlab.engine
from math import sqrt
from multiprocessing import Pool
from numpy.linalg import norm
from qiskit.quantum_info import Statevector
from HamiltonianCertification.Certification import certify, cert_prob, MyStateVector

@pytest.fixture(scope="session")
def engine():
    eng = matlab.engine.start_matlab()
    eng.cd('tests')
    yield eng
    eng.quit()

def _certify_python(hyp, lab, N):
    return sum(certify(hyp, lab, return_prob=True) for _ in range(N))

@pytest.mark.parametrize("hyp, lab", [
    ([1,0], [0,1]),
    ([1,2,3,4], [5,6,7,8]),
    ([1,2,3,4,5,6,7,8], [9,0,1,2,3,4,5,6]),
    ([1,0,0,0], [0,0,0,1]),
    ([1,0,0,1,1,0,0,1], [1,1,1,1,1,1,1,1]),
    ([sqrt(2),0,1,1], [1,1j,1,1j]),
    ([1,1,sqrt(2),0,1,1j,1j,-1], [1,-1j,0,0,1,1,0,0]),
])
def test_certify_match(engine, hyp, lab):
    hyp += 0.1 * np.random.rand(len(hyp),2).view(np.complex128).flatten()
    lab += 0.1 * np.random.rand(len(lab),2).view(np.complex128).flatten()

    hyp_python = Statevector(hyp)
    hyp_python = MyStateVector(sv=hyp_python/norm(hyp_python))
    lab_python = Statevector(lab)
    lab_python = MyStateVector(sv=lab_python/norm(lab_python))
    j = 12
    N = 1500
    acc_python = sum(Pool(j).starmap(_certify_python, [(hyp_python, lab_python, N) for _ in range(j)])) / (N * j)

    n = int.bit_length(len(hyp))-1
    hyp_matlab = engine.transpose(matlab.double(hyp_python.sv.data, is_complex=True))
    lab_matlab = engine.transpose(matlab.double(lab_python.sv.data, is_complex=True))
    acc_matlab = engine.certify(matlab.double(n), hyp_matlab, lab_matlab)

    assert abs(acc_python - acc_matlab) < 1e-2

@pytest.mark.parametrize("hyp, lab", [
    ([1,2j,3,4], [5,6,7j,8]),
    ([1,2,3,4j,5,6,7,8], [9,0,1j,2,3,4,5,6]),
])
def test_cert_prob_exact(engine, hyp, lab):
    hyp_python = Statevector(hyp)
    hyp_python = MyStateVector(sv=hyp_python/norm(hyp_python))
    lab_python = Statevector(lab)
    lab_python = MyStateVector(sv=lab_python/norm(lab_python))
    acc_python = cert_prob(hyp_python, lab_python)

    n = int.bit_length(len(hyp))-1
    hyp_matlab = engine.transpose(matlab.double(hyp_python.sv.data, is_complex=True))
    lab_matlab = engine.transpose(matlab.double(lab_python.sv.data, is_complex=True))
    acc_matlab = engine.certify(matlab.double(n), hyp_matlab, lab_matlab)

    assert abs(acc_python - acc_matlab) < 1e-12
