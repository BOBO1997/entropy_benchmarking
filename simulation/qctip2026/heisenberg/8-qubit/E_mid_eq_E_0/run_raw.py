import numpy as np
import scipy as sp
from pprint import pprint
import pickle
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.compiler import transpile
from qiskit.transpiler.passes import RemoveBarriers
import qiskit.quantum_info as qi

from entropy_benchmarking.hamiltonian import make_H_Heisenberg_ring
from entropy_benchmarking.backend_simulator import hamiltonian_to_dm, DMExtended
from entropy_benchmarking.distribution import area_gaussian_outside

from setting import *

def statevector_to_dm(psi):
    """
    psi: 1D numpy array of shape (d,)
         (complex-valued state vector, not necessarily normalized)

    return: density matrix rho of shape (d, d)
    """
    psi = np.asarray(psi, dtype=np.complex128)
    norm = np.vdot(psi, psi)          # <psi|psi>
    if norm == 0:
        raise ValueError("Zero vector cannot form a density matrix.")
    psi = psi / np.sqrt(norm)         # normalize
    rho = np.outer(psi, np.conjugate(psi))
    return rho


H_Heisenberg = make_H_Heisenberg_ring(num_qubits=num_qubits,
                                      endian_str="little") ### create Heisenberg Hamiltonian in a dictionary format
pprint(H_Heisenberg)
matrix_Heisenberg = hamiltonian_to_dm(hamiltonian=H_Heisenberg,
                                      endian_hamiltonian="little",
                                      endian_dm="little") ### convert Heisenberg Hamiltonian to its matrix form
eigvals, eigvecs = np.linalg.eigh(matrix_Heisenberg)
args_sorted = np.argsort(eigvals)
eigvals_sorted = eigvals[args_sorted]
eigvecs_sorted = eigvecs[:, args_sorted]
energy_theoretical = eigvals_sorted[0].real
state_ground_theoretical = eigvecs_sorted[:, 0]
dm_theoretical = DMExtended(statevector_to_dm(state_ground_theoretical)) 


with open("run_classical.pkl", "rb") as f:
    run_classical = pickle.load(f)
    energy_classical_lower = run_classical["energy_classical_lower"]
    energy_classical_upper = run_classical["energy_classical_upper"] ### very close to energy_theoretical

E_minus = energy_classical_lower
# E_plus = energy_theoretical + np.abs(energy_theoretical - energy_classical_lower)
E_plus = 2 * energy_theoretical - energy_classical_lower
# energy_mid = (energy_classical_lower + energy_classical_upper) / 2
energy_mid = energy_theoretical # energy_classical_lower + np.abs(energy_theoretical - energy_classical_lower) / 2

deltas_2d = []
for N_shots in Ns_shots:
    deltas = []
    for p in ps_dep_global:
        P = 1 - (1 - p) ** num_layers
        dm_depolarised = (1 - P) * dm_theoretical + P * np.identity(2 ** num_qubits) / (2 ** num_qubits)
        assert dm_depolarised.is_valid()
        variance_base = np.abs((dm_depolarised @ (matrix_Heisenberg ** 2)).trace() - (dm_depolarised @ matrix_Heisenberg).trace() ** 2)
        # print("varince_base:", variance_base)
        # print(N_shots, p)
        delta = area_gaussian_outside(mu=((1 - p) ** num_layers) * energy_theoretical,
                                      var=variance_base / N_shots,
                                      E_minus=E_minus,
                                      E_plus=E_plus)
        deltas.append(delta)
    deltas_2d.append(deltas)


with open("run_raw.pkl", "wb") as f:
    pickle.dump(obj={"deltas_2d": np.array(deltas_2d),
                     "datetime": datetime.now(ZoneInfo("Europe/Paris")),
                    },
                file=f)