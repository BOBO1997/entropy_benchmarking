import numpy as np
import scipy as sp
from pprint import pprint
import pickle
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from entropy_benchmarking.hamiltonian import make_H_Heisenberg_ring
from entropy_benchmarking.backend_simulator import hamiltonian_to_dm

from setting import *

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
energy_theoretical = eigvals_sorted[0]
print("energy_theoretical:", energy_theoretical)

with open("run_theoretical.pkl", "wb") as f:
    pickle.dump(obj={"energy_theoretical": energy_theoretical,
                     "datetime": datetime.now(ZoneInfo("Europe/Paris")),
                    },
                file=f)