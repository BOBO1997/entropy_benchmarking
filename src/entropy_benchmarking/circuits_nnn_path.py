from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter
from entropy_benchmarking.circuits_util import (
    gate_U_enc, 
    gate_U_dec,
    gate_H_eff_triangle,
    gate_U_dec_H,
    gate_block_trotter_3cnot,
)
from entropy_benchmarking.circuits_nnn_util import (
    gate_swaps_red_green,
    gate_swaps_yellow_blue,
)


### ================================================== ###


###! under construction: just copied from ring
def gate_swaps_blue_red_path_parallel(num_qubits: int,
                                      to_instruction: bool = True,
                                      add_barrier: bool = False,
                                     ) -> Union[QuantumCircuit, Instruction]:
    """
    a sequence of swap operation between blue H_eff and red H_eff

    Parallel implementation optimised for the ring nnn structure
    """
    assert num_qubits % 4 == 0

    qc = QuantumCircuit(num_qubits,
                        name="swaps_blue_red")

    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 3:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 0: ### == 4
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 3:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 0: ### == 4
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
        if ith_qubit % 4 == 2:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 2:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
        if ith_qubit % 4 == 0:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 1:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 0:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 1:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)

    if add_barrier:
        qc.barrier(label="swaps_blue_red_ring_parallel")

    return qc.to_instruction(label="swaps_blue_red_ring_parallel") if to_instruction else qc


### ================================================== ###


###! under construction: just copied from ring
def gate_swaps_green_yellow_path_parallel(num_qubits: int,
                                          to_instruction: bool = True,
                                          add_barrier: bool = False,
                                         ) -> Union[QuantumCircuit, Instruction]:
    """
    a sequence of swap operation between green H_eff and yellow H_eff

    Parallel implementation optimised for the ring nnn structure
    """
    assert num_qubits % 4 == 0

    qc = QuantumCircuit(num_qubits,
                        name="swaps_green_yellow")

    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 0:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 3:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 2:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 3:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
        if ith_qubit % 4 == 1:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 3:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
        if ith_qubit % 4 == 1:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 2:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 1:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(num_qubits):
        if ith_qubit % 4 == 0:
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)

    if add_barrier:
        qc.barrier(label="swaps_green_yellow_ring_parallel")

    return qc.to_instruction(label="swaps_green_yellow_ring_parallel") if to_instruction else qc


### ================================================== ###


def gate_nnn_path_prr(num_qubits: int,
                      num_iterations: int,
                      dt: float,
                      J1: float = 1.0, ### / 4, ###! half spin: Z in Hamiltonian -> S = 1/2 Z, i.e. S_{i}S_{j} = Z_{i}Z_{j} / 4
                      J2: float = 0.5, ### / 4, ###! half spin: Z in Hamiltonian -> S = 1/2 Z, i.e. S_{i}S_{j} = Z_{i}Z_{j} / 4
                      to_instruction: bool = True,
                      add_barrier: bool = False,
                     ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Suzuki-Trotter iterations in PRR paper for the path next nearest neighbour (path NNN) structure
    https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033107
    """
    assert num_qubits & 1 ### 1 mod 2

    gate_block_trotter = gate_block_trotter_3cnot ###! 

    qc = QuantumCircuit(num_qubits,
                        name="nnn_path_prr")
    
    for ith_iteration in range(num_iterations):

        ### Trotter blocks among direct neighbours ###
        for ith_qubit in range(num_qubits - 1): ### path structure
            if not (ith_qubit & 1): ### even
                qc.compose(gate_block_trotter(dt=dt * J1,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits=[ith_qubit, ith_qubit + 1],
                            inplace=True,)
        for ith_qubit in range(num_qubits - 1): ### path structure
            if ith_qubit & 1: ### odd
                qc.compose(gate_block_trotter(dt=dt * J1,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits=[ith_qubit, ith_qubit + 1],
                            inplace=True,)
        
        ### Trotter blocks among next neighbours ###
        for ith_qubit in range(num_qubits - 1): ### the first swap layer ### path structure
            if ith_qubit % 4 == 1: ### 1 mod 4
                # qc.cx(ith_qubit, ith_qubit + 1)
                # qc.cx(ith_qubit + 1, ith_qubit)
                # qc.cx(ith_qubit, ith_qubit + 1)
                qc.swap(qubit1=ith_qubit, 
                        qubit2=ith_qubit + 1)
                if add_barrier:
                    qc.barrier([ith_qubit, ith_qubit + 1], label="swap")
        for ith_qubit in range(num_qubits - 1): ### the first swapped J2 layer ### path structure
            if not (ith_qubit & 1): ### 0 mod 2
                qc.compose(gate_block_trotter(dt=dt * J2,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits=[ith_qubit, ith_qubit + 1],
                            inplace=True,)
        for ith_qubit in range(num_qubits - 1): ### the second swap layer ### path structure
            if ith_qubit & 1: ### 1 mod 2
                # qc.cx(ith_qubit, ith_qubit + 1)
                # qc.cx(ith_qubit + 1, ith_qubit)
                # qc.cx(ith_qubit, ith_qubit + 1)
                qc.swap(qubit1=ith_qubit, 
                        qubit2=ith_qubit + 1)
                if add_barrier:
                    qc.barrier([ith_qubit, ith_qubit + 1], label="swap")
        for ith_qubit in range(num_qubits - 1): ### the second swapped J2 layer ### path structure
            if not (ith_qubit & 1): ### 0 mod 2
                qc.compose(gate_block_trotter(dt=dt * J2,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits=[ith_qubit, ith_qubit + 1],
                            inplace=True,)
        for ith_qubit in range(num_qubits - 1): ### the third swap layer ### path structure
            if ith_qubit % 4 == 3: ### 3 mod 4
                # qc.cx(ith_qubit, ith_qubit + 1)
                # qc.cx(ith_qubit + 1, ith_qubit)
                # qc.cx(ith_qubit, ith_qubit + 1)
                qc.swap(qubit1=ith_qubit, 
                        qubit2=ith_qubit + 1)
                if add_barrier:
                    qc.barrier([ith_qubit, ith_qubit + 1], label="swap")

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="nnn_path_prr") if to_instruction else qc


###! under construction: just pasted gate_nnn_ring_blue_only
def gate_nnn_path_blue_only(num_qubits: int,
                            num_iterations: int,
                            dt: float, ###! this dt is for the dt in each H_eff block
                            J1: float = 1.0, ### / 4, ###! half spin: Z in Hamiltonian -> S = 1/2 Z, i.e. S_{i}S_{j} = Z_{i}Z_{j} / 4
                            J2: float = 0.5, ### / 4, ###! half spin: Z in Hamiltonian -> S = 1/2 Z, i.e. S_{i}S_{j} = Z_{i}Z_{j} / 4
                            connectivity: str = "complete",
                            to_instruction: bool = True,
                            add_barrier: bool = False,
                           ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    proposed Suzuki-Trotter iterations for the path next nearest neighbour (path NNN) structure
    ###! Remark: one step (iteration pattern) = 2dt = dt (red, blue) + dt (yellow, green)
    ###! Remark: if one wishes to simulate time T, one needs to use dt = T / (num_iterations / 2)
    ###! Remark: this function is only for when J1 = 2J2
    """
    assert num_qubits % 4 == 0 ### 0 mod 4
    assert np.allclose(J1, 2 * J2) ###! Remark: this function is only for when J1 = 2J2

    qc = QuantumCircuit(num_qubits,
                        name="nnn_path_blue_only")

    for ith_iteration in range(num_iterations):
        ###
        for ith_qubit in range(num_qubits): ### encoder for blue H_eff
            if ith_qubit % 4 == 2: ### 2 mod 4
                qc.compose(gate_U_enc(connectivity=connectivity,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ###? blue H_eff
            if ith_qubit % 4 == 2: ### 2 mod 4
                qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ### encoder for blue H_eff
            if ith_qubit % 4 == 2: ### 2 mod 4
                qc.compose(gate_U_dec(connectivity=connectivity,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                            inplace=True,)
        ###
        for ith_qubit in range(num_qubits): ### encoder for blue H_eff
            if ith_qubit % 4 == 0: ### 0 mod 4
                qc.compose(gate_U_enc(connectivity=connectivity,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ###? blue H_eff
            if ith_qubit % 4 == 1: ### 1 mod 4
                qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ### encoder for blue H_eff
            if ith_qubit % 4 == 0: ### 0 mod 4
                qc.compose(gate_U_dec(connectivity=connectivity,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                            inplace=True,)
        ###
        for ith_qubit in range(num_qubits): ### encoder for blue H_eff
            if ith_qubit % 4 == 1: ### 1 mod 4
                qc.compose(gate_U_enc(connectivity=connectivity,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ###? blue H_eff
            if ith_qubit % 4 == 1: ### 1 mod 4
                qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ### encoder for blue H_eff
            if ith_qubit % 4 == 1: ### 1 mod 4
                qc.compose(gate_U_dec(connectivity=connectivity,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                            inplace=True,)
        ###
        for ith_qubit in range(num_qubits): ### encoder for blue H_eff
            if ith_qubit % 4 == 3: ### 3 mod 4
                qc.compose(gate_U_enc(connectivity=connectivity,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ###? blue H_eff
            if ith_qubit % 4 == 0: ### 0 mod 4
                qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ### encoder for blue H_eff
            if ith_qubit % 4 == 3: ### 3 mod 4
                qc.compose(gate_U_dec(connectivity=connectivity,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                            inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="nnn_path_blue_only") if to_instruction else qc


###! under construction: just pasted gate_nnn_ring_triangle
def gate_nnn_path_triangle(num_qubits: int,
                           num_iterations: int,
                           dt: float, ###! this dt is for the dt in each H_eff block
                           J1: float = 1.0, ### / 4, ###! half spin: Z in Hamiltonian -> S = 1/2 Z, i.e. S_{i}S_{j} = Z_{i}Z_{j} / 4
                           J2: float = 0.5, ### / 4, ###! half spin: Z in Hamiltonian -> S = 1/2 Z, i.e. S_{i}S_{j} = Z_{i}Z_{j} / 4
                           to_instruction: bool = True,
                           add_barrier: bool = False,
                          ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    proposed Suzuki-Trotter iterations for the path next nearest neighbour (path NNN) structure
    ###! Remark: one step (iteration pattern) = 2dt = dt (red, blue) + dt (yellow, green)
    ###! Remark: if one wishes to simulate time T, one needs to use dt = T / (num_iterations / 2)
    ###! Remark: this function is only for when J1 = 2J2
    """
    assert num_qubits % 4 == 0 ### 0 mod 4
    assert np.allclose(J1, 2 * J2) ###! Remark: this function is only for when J1 = 2J2

    qc = QuantumCircuit(num_qubits,
                        name="nnn_path_triangle")

    for ith_iteration in range(num_iterations):
        if ith_iteration == 0: ### initial encoding at the first Trotter iteration
            for ith_qubit in range(num_qubits): ### encoder for blue H_eff
                if ith_qubit % 4 == 2: ### 2 mod 4
                    qc.compose(gate_U_enc(connectivity="path",
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier),
                                qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                                inplace=True,)
                
        for ith_qubit in range(num_qubits): ###? blue H_eff
            if ith_qubit % 4 == 2: ### 2 mod 4
                qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                            inplace=True,)
        ### the swap layer between blue H_eff and red H_eff
        qc.compose(gate_swaps_blue_red_path_parallel(num_qubits=num_qubits,
                                                        to_instruction=to_instruction,
                                                        add_barrier=add_barrier),
                    qubits=[(jth_qubit + 0) % num_qubits for jth_qubit in range(num_qubits)],
                    inplace=True,)
        for ith_qubit in range(num_qubits): ###! red H_eff
            if ith_qubit % 4 == 1: ### 1 mod 4
                qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ### the swap layer between red H_eff and green H_eff
            if ith_qubit % 4 == 0: ### 0 mod 4
                qc.compose(gate_swaps_red_green(to_instruction=to_instruction,
                                                add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 4)],
                            inplace=True,)
        for ith_qubit in range(num_qubits): ### green H_eff
            if ith_qubit % 4 == 1: ### 1 mod 4
                qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                            inplace=True,)
        ### the swap layer between green H_eff and yellow H_eff
        qc.compose(gate_swaps_green_yellow_path_parallel(num_qubits=num_qubits,
                                                            to_instruction=to_instruction,
                                                            add_barrier=add_barrier),
                    qubits=[(jth_qubit + 3) % num_qubits for jth_qubit in range(num_qubits)],
                    inplace=True,)
        for ith_qubit in range(num_qubits): ### todo(fake) yellow H_eff
            if ith_qubit % 4 == 0: ### 1 mod 4
                qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                            inplace=True,)
        
        if ith_iteration < num_iterations - 1: ### there are still Trotter iterations later
            for ith_qubit in range(num_qubits): ### the swap layer between yellow H_eff and blue H_eff
                if ith_qubit % 4 == 2: ### 2 mod 4
                    qc.compose(gate_swaps_yellow_blue(to_instruction=to_instruction,
                                                        add_barrier=add_barrier),
                                qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 4)],
                                inplace=True,)
        else: ### i.e. if ith_iteration == num_iterations - 1
            for ith_qubit in range(num_qubits): ### final decoding at the last Trotter iteration, note that the decoder is with H and flipped
                if ith_qubit % 4 == 3: ### 3 mod 4
                    qc.compose(gate_U_dec_H(connectivity="path",
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier),
                                qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                                inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="nnn_path_triangle") if to_instruction else qc