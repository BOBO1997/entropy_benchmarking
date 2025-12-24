import numpy as np

### ============ ###

num_qubits = 8

num_layers = 9

# Gamma_allowed = 1.5
Gammas_allowed = np.array([1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 10.0])

gap = 1e-1
# gap = 1e-2

Ns_shots = np.logspace(0, 7, num=100)

### depolarising rate per layer
ps_dep_global = np.logspace(-7, 0, num=100)
# Ps_dep_global = 1 - (1 - np.logspace(-7, 0, num=100)) ** num_layers

