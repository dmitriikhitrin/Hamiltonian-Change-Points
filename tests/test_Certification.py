import pytest
import matlab.engine
from math import sqrt
from numpy.linalg import norm
from qiskit.quantum_info import Statevector
from HamiltonianCertification.Certification import certify, MyStateVector

@pytest.fixture()
def engine():
    eng = matlab.engine.start_matlab()
    eng.cd('tests')
    yield eng
    eng.quit()

@pytest.mark.parametrize("hyp, lab", [
    ([1,0], [0,1]),
    ([1,2,3,4], [5,6,7,8]),
    ([1,2,3,4,5,6,7,8], [9,0,1,2,3,4,5,6]),
    ([1,0,0,0], [0,0,0,1]),
    ([1,0,0,1,1,0,0,1], [1,1,1,1,1,1,1,1]),
    ([sqrt(2),0,1,1], [1j,-1,1j,-1]),
    ([1,1,sqrt(2),0,1,1j,1j,-1], [1,-1j,0,0,1,1,0,0]),
])
def test_certify_match(engine, hyp, lab):
    hyp_python = Statevector(hyp)
    hyp_python = MyStateVector(sv=hyp_python/norm(hyp_python))
    lab_python = Statevector(lab)
    lab_python = MyStateVector(sv=lab_python/norm(lab_python))
    N = 1000
    acc_python = 0
    for _ in range(N):
        acc_python += certify(hyp_python, lab_python) / N

    n = int.bit_length(len(hyp))-1
    hyp_matlab = engine.transpose(matlab.double(hyp_python.sv.data, is_complex=True))
    lab_matlab = engine.transpose(matlab.double(lab_python.sv.data, is_complex=True))
    acc_matlab = engine.certify(matlab.double(n), hyp_matlab, lab_matlab)

    assert abs(acc_python - acc_matlab) < 1e-2
