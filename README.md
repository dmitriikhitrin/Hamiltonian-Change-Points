# Hamiltonian Change Points
Numerical analysis for the work [*Autonomous Hamiltonian certification and changepoint detection*](https://arxiv.org/abs/2603.26655).

Implemented primarily using the `qiskit` package.

### Certification
1. Run experiments with `Rydberg.ipynb`, then
1. Create plots with `Plotting.ipynb` (Fig. 2)

Testing (compare to reference MATLAB code): run `pytest` from the main directory.

### Change detection
* `cusum.py`: direct CUSUM simulaton and example plot (Fig. 3)
* `integer.py`: analytical run length formulas in cases of integer multiple scores (Fig. 4)
