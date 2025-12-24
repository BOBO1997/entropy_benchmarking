from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction
from entropy_benchmarking.circuits_util import (
    gate_U_enc,
    gate_U_dec,
)


### ================================================== ###


def gate_swaps_blue_red(to_instruction: bool = True,
                        add_barrier: bool = False,
                       ) -> Union[QuantumCircuit, Instruction]:
    """
    a sequence of swap operation between blue H_eff and red H_eff
    """
    qc = QuantumCircuit(5,
                        name="swaps_blue_red")

    qc.cx(3, 4)

    qc.cx(4, 3)

    qc.cx(3, 2)

    qc.cx(4, 3)
    qc.cx(2, 1)

    qc.cx(2, 3)
    qc.cx(0, 1)

    qc.cx(1, 2)

    qc.cx(0, 1)

    qc.cx(1, 0)

    if add_barrier:
        qc.barrier(label="swaps_blue_red")

    return qc.to_instruction(label="swaps_blue_red") if to_instruction else qc


def gate_swaps_red_green(to_instruction: bool = True,
                         add_barrier: bool = False,
                        ) -> Union[QuantumCircuit, Instruction]:
    """
    a sequence of swap operation between red H_eff and green H_eff
    """
    qc = QuantumCircuit(4,
                        name="swaps_red_green")

    qc.cx(1, 0)

    qc.cx(0, 1)

    qc.cx(1, 2)

    qc.cx(0, 1)
    qc.cx(2, 3)

    qc.cx(1, 2)

    qc.cx(2, 3)

    qc.cx(3, 2)

    if add_barrier:
        qc.barrier(label="swaps_red_green")

    return qc.to_instruction(label="swaps_red_green") if to_instruction else qc


def gate_swaps_green_yellow(to_instruction: bool = True,
                            add_barrier: bool = False,
                           ) -> Union[QuantumCircuit, Instruction]:
    """
    a sequence of swap operation between green H_eff and yellow H_eff
    """
    qc = QuantumCircuit(5,
                        name="swaps_green_yellow")

    qc.cx(4, 3)

    qc.cx(3, 4)

    qc.cx(2, 3)

    qc.cx(3, 4)
    qc.cx(1, 2)

    qc.cx(3, 2)
    qc.cx(1, 0)

    qc.cx(2, 1)

    qc.cx(1, 0)

    qc.cx(0, 1)

    if add_barrier:
        qc.barrier(label="swaps_green_yellow")

    return qc.to_instruction(label="swaps_green_yellow") if to_instruction else qc


def gate_swaps_yellow_blue(to_instruction: bool = True,
                           add_barrier: bool = False,
                          ) -> Union[QuantumCircuit, Instruction]:
    """
    a sequence of swap operation between yellow H_eff and blue H_eff
    """
    qc = QuantumCircuit(4,
                        name="swaps_yellow_blue")

    qc.cx(1, 2)

    qc.cx(2, 1)

    qc.cx(0, 1)
    qc.cx(3, 2)

    qc.cx(1, 0)
    qc.cx(2, 3)

    qc.cx(2, 1)

    qc.cx(1, 2)

    if add_barrier:
        qc.barrier(label="swaps_yellow_blue")

    return qc.to_instruction(label="swaps_yellow_blue") if to_instruction else qc

