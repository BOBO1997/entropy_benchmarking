from typing import *
import copy, time
import numpy as np
import scipy as sp
import qiskit.quantum_info as qi
from entropy_benchmarking.hamiltonian import Hamiltonian


def make_O_Mst(num_qubits: int,
               endian_O_Mst: str = "big",
              ) -> Hamiltonian:
    O_Mst = Hamiltonian({})
    flag_reverse_endian = 1 if endian_O_Mst == "big" else -1
    for ith_qubit in range(num_qubits)[::flag_reverse_endian]:
        Z_i = list("I" * num_qubits)
        Z_i[ith_qubit] = "Z"
        Z_i = "".join(Z_i)
        O_Mst += Hamiltonian({Z_i: (1 / 2) * (-1) ** (ith_qubit + 1) / num_qubits})
    return O_Mst


def compute_expval_O_Mst(hist: dict,
                         endian_hist: str = "little",
                         endian_O_Mst: str = "big",
                        ) -> Tuple[float, float]:
    """
    Compute the expectation value of the observable O_{M_{st}}
    """
    num_clbits = len(list(hist.keys())[0])
    num_shots = sum(list(hist.values()))
    flag_reverse_endian = 1 if endian_hist == endian_O_Mst else -1 ### 1 is to do nothing, -1 is to flip the list

    expval = 0
    variance = 0
    for ith_clbit in range(num_clbits)[::flag_reverse_endian]:
        for key, item in hist.items():
            if key[ith_clbit] == "0": ### |0> -> eigval of Z is 1
                expval += item * (-1) ** (ith_clbit + 1)
            else: ### |1> -> eigval of Z is -1
                expval -= item * (-1) ** (ith_clbit + 1)
    expval /= num_shots ### taking average over the number of shots
    expval /= num_clbits ### removing the redundancy in summing up the shots every time for each classical bit
    expval /= 2 ### half spin: Z in Hamiltonian -> S = 1/2 Z
    return expval, variance