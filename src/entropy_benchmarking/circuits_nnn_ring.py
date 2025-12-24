from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter
from entropy_benchmarking.circuits_util import (
    gate_U_enc, 
    gate_U_dec,
    gate_H_eff_triangle,
    gate_U_dec_H,
    gate_block_trotter_3cnot,
    compute_J_of_t,
    gate_block_trotter_triangle,
)
from entropy_benchmarking.circuits_nnn_util import (
    gate_swaps_red_green,
    gate_swaps_yellow_blue,
)


### ================================================== ###


def gate_swaps_blue_red_ring_parallel(num_qubits: int,
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

    for ith_qubit in range(3, num_qubits, 4): ### 3 mod 4
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(0, num_qubits, 4): ### 4 mod 4
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(3, num_qubits, 4): ### 3 mod 4
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(0, num_qubits, 4): ### 4 mod 4
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(2, num_qubits, 4): ### 2 mod 4
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(2, num_qubits, 4): ### 2 mod 4
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(0, num_qubits, 4): ### 0 mod 4
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(1, num_qubits, 4): ### 1 mod 4
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(0, num_qubits, 4): ### 0 mod 4
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(1, num_qubits, 4): ### 1 mod 4
            qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)

    if add_barrier:
        qc.barrier(label="swaps_blue_red_ring_parallel")

    return qc.to_instruction(label="swaps_blue_red_ring_parallel") if to_instruction else qc


### ================================================== ###


def gate_swaps_green_yellow_ring_parallel(num_qubits: int,
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

    for ith_qubit in range(0, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(3, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(1, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(3, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(1, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)
    for ith_qubit in range(3, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(1, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(2, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(1, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit - 1) % num_qubits)
    for ith_qubit in range(0, num_qubits, 4):
        qc.cx((ith_qubit + 0) % num_qubits, (ith_qubit + 1) % num_qubits)

    if add_barrier:
        qc.barrier(label="swaps_green_yellow_ring_parallel")

    return qc.to_instruction(label="swaps_green_yellow_ring_parallel") if to_instruction else qc


### ================================================== ###


def gate_nnn_ring_prr(num_qubits: int,
                      num_iterations: int,
                      time_evolution: float, ### total time to evolute
                      J1: float = 1.0,
                      J2: float = 0.5,
                      to_instruction: bool = True,
                      add_barrier: bool = False,
                     ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Suzuki-Trotter iterations in PRR paper for the ring next nearest neighbour (path NNN) structure
    https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033107
    """
    assert num_qubits % 4 == 0 ### 0 mod 4

    gate_block_trotter = gate_block_trotter_3cnot ###! 

    if time_evolution is not None:
        time_evolution_per_iteration = time_evolution / num_iterations
        dt = time_evolution_per_iteration ###! this dt is for the dt in each H_eff block, which is now the same as time_evolution_per_iteration
    else:
        dt = Parameter(r"$\Delta t$")

    qc = QuantumCircuit(num_qubits,
                        name="nnn_ring_prr")
    
    for ith_iteration in range(num_iterations):

        ### Trotter blocks among direct neighbours ###
        for ith_qubit in range(0, num_qubits, 2): ### even
            qc.compose(gate_block_trotter(dt=dt * J1,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[ith_qubit, (ith_qubit + 1) % num_qubits],
                        inplace=True,)
        for ith_qubit in range(1, num_qubits, 2): ### odd
            qc.compose(gate_block_trotter(dt=dt * J1,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[ith_qubit, (ith_qubit + 1) % num_qubits],
                        inplace=True,)
        
        ### Trotter blocks among next neighbours ###
        for ith_qubit in range(1, num_qubits, 4): ### the first swap layer ### 1 mod 4 ###! in this swap: no need for (ith_qubit + 1) % num_qubits
            qc.swap(qubit1=ith_qubit, 
                    qubit2=ith_qubit + 1)
            if add_barrier:
                qc.barrier([ith_qubit, ith_qubit + 1], label="swap")
        for ith_qubit in range(0, num_qubits, 2): ### the first swapped J2 layer ### 0 mod 2
            qc.compose(gate_block_trotter(dt=dt * J2,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[ith_qubit, ith_qubit + 1],
                        inplace=True,)
        for ith_qubit in range(1, num_qubits, 2): ### the second swap layer ### 1 mod 2
            qc.swap(qubit1=ith_qubit, 
                    qubit2=(ith_qubit + 1) % num_qubits)
            if add_barrier:
                qc.barrier([ith_qubit, (ith_qubit + 1) % num_qubits], label="swap")
        for ith_qubit in range(0, num_qubits, 2): ### the second swapped J2 layer ### 0 mod 2
            qc.compose(gate_block_trotter(dt=dt * J2,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[ith_qubit, ith_qubit + 1],
                        inplace=True,)
        for ith_qubit in range(3, num_qubits,4): ### the third swap layer ### 3 mod 4
            qc.swap(qubit1=ith_qubit, 
                    qubit2=(ith_qubit + 1) % num_qubits)
            if add_barrier:
                qc.barrier([ith_qubit, (ith_qubit + 1) % num_qubits], label="swap")

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="nnn_ring_prr") if to_instruction else qc


###! deprecated, not in use
def gate_nnn_ring_blue_only(num_qubits: int,
                            num_iterations: int,
                            time_evolution: float, ### total time to evolute
                            J1: float = 1.0,
                            J2: float = 0.5,
                            connectivity: str = "complete",
                            to_instruction: bool = True,
                            add_barrier: bool = False,
                           ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter

    proposed Suzuki-Trotter iterations for the ring next nearest neighbour (ring NNN) structure
    ###! i.e. connectivity = "path"

    ###! Remark: one step (iteration pattern) = 2dt = dt (red, blue) + dt (yellow, green)
    ###! Remark: if one wishes to simulate time T, one needs to use dt = T / (num_iterations / 2)
    ###! Remark: this function is only for when J1 = 2J2
    """
    assert num_qubits % 4 == 0 ### 0 mod 4
    assert np.allclose(J1, 2 * J2) ###! Remark: this function is only for when J1 = 2J2

    if time_evolution is not None:
        time_evolution_per_iteration = time_evolution / num_iterations
        dt = time_evolution_per_iteration ###! this dt is for the dt in each H_eff block, which is now the same as time_evolution_per_iteration
    else:
        dt = Parameter(r"$\Delta t$")

    qc = QuantumCircuit(num_qubits,
                        name="nnn_ring_blue_only")

    for ith_iteration in range(num_iterations):
        ###
        for ith_qubit in range(2, num_qubits, 4): ### encoder for blue H_eff ### 2 mod 4
            qc.compose(gate_U_enc(connectivity=connectivity,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        for ith_qubit in range(2, num_qubits, 4): ###? blue H_eff ### 2 mod 4
            qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        for ith_qubit in range(2, num_qubits, 4): ### encoder for blue H_eff ### 2 mod 4
            qc.compose(gate_U_dec(connectivity=connectivity,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(0, num_qubits, 4): ### encoder for blue H_eff ### 0 mod 4
            qc.compose(gate_U_enc(connectivity=connectivity,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                        inplace=True,)
        for ith_qubit in range(1, num_qubits, 4): ###? blue H_eff ### 1 mod 4
            qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        for ith_qubit in range(0, num_qubits, 4): ### encoder for blue H_eff ### 0 mod 4
            qc.compose(gate_U_dec(connectivity=connectivity,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                        inplace=True,)
        ###
        for ith_qubit in range(1, num_qubits, 4): ### encoder for blue H_eff ### 1 mod 4
            qc.compose(gate_U_enc(connectivity=connectivity,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        for ith_qubit in range(1, num_qubits, 4): ###? blue H_eff ### 1 mod 4
            qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        for ith_qubit in range(1, num_qubits, 4): ### encoder for blue H_eff ### 1 mod 4
            qc.compose(gate_U_dec(connectivity=connectivity,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(3, num_qubits, 4): ### encoder for blue H_eff ### 3 mod 4
            qc.compose(gate_U_enc(connectivity=connectivity,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                        inplace=True,)
        for ith_qubit in range(0, num_qubits, 4): ###? blue H_eff ### 0 mod 4
            qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        for ith_qubit in range(3, num_qubits, 4): ### encoder for blue H_eff ### 3 mod 4
            qc.compose(gate_U_dec(connectivity=connectivity,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                        inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="nnn_ring_blue_only") if to_instruction else qc


def gate_nnn_ring_triangle(num_qubits: int,
                           num_iterations: int,
                           time_evolution: float, ### total time to evolute
                           J1: float = 1.0,
                           J2: float = 0.5,
                           to_instruction: bool = True,
                           add_barrier: bool = False,
                          ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter

    proposed Suzuki-Trotter iterations for the ring next nearest neighbour (ring NNN) structure
    ###! i.e. connectivity = "path"

    ###! Remark: one step (iteration pattern) = 2dt = dt (red, blue) + dt (yellow, green)
    ###! Remark: if one wishes to simulate time T, one needs to use dt = T / (num_iterations / 2)
    ###! Remark: this function is only for when J1 = 2J2
    """
    assert num_qubits % 4 == 0 ### 0 mod 4
    assert np.allclose(J1, 2 * J2) ###! Remark: this function is only for when J1 = 2J2

    if time_evolution is not None:
        time_evolution_per_iteration = time_evolution / num_iterations
        dt = time_evolution_per_iteration ###! this dt is for the dt in each H_eff block, which is now the same as time_evolution_per_iteration
    else:
        dt = Parameter(r"$\Delta t$")

    qc = QuantumCircuit(num_qubits,
                        name="nnn_ring_triangle")

    for ith_iteration in range(num_iterations):
        ###
        if ith_iteration == 0: ### initial encoding at the first Trotter iteration
            for ith_qubit in range(2, num_qubits, 4): ### encoder for blue H_eff ### 2 mod 4
                qc.compose(gate_U_enc(connectivity="path",
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                            inplace=True,)
        ###
        for ith_qubit in range(2, num_qubits, 4): ###? blue H_eff ### 2 mod 4
            qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        ### the swap layer between blue H_eff and red H_eff
        qc.compose(gate_swaps_blue_red_ring_parallel(num_qubits=num_qubits,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                    qubits=[(jth_qubit + 0) % num_qubits for jth_qubit in range(num_qubits)],
                    inplace=True,)
        ###
        for ith_qubit in range(1, num_qubits, 4): ###! red H_eff ### 1 mod 4
            qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        ###
        for ith_qubit in range(0, num_qubits, 4): ### the swap layer between red H_eff and green H_eff ### 0 mod 4
            qc.compose(gate_swaps_red_green(to_instruction=to_instruction,
                                            add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 4)],
                        inplace=True,)
        ###
        for ith_qubit in range(1, num_qubits, 4): ###* green H_eff ### 1 mod 4
            qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        ### the swap layer between green H_eff and yellow H_eff
        qc.compose(gate_swaps_green_yellow_ring_parallel(num_qubits=num_qubits,
                                                            to_instruction=to_instruction,
                                                            add_barrier=add_barrier),
                    qubits=[(jth_qubit + 3) % num_qubits for jth_qubit in range(num_qubits)],
                    inplace=True,)
        ###
        for ith_qubit in range(0, num_qubits, 4): ### todo(fake) yellow H_eff ### 0 mod 4
            qc.compose(gate_H_eff_triangle(dt=dt * J1,
                                    to_instruction=to_instruction,
                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 2)],
                        inplace=True,)
        ###
        if ith_iteration < num_iterations - 1: ### there are still Trotter iterations later
            for ith_qubit in range(2, num_qubits, 4): ### the swap layer between yellow H_eff and blue H_eff ### 2 mod 4
                qc.compose(gate_swaps_yellow_blue(to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 4)],
                            inplace=True,)
        ###
        else: ### i.e. if ith_iteration == num_iterations - 1
            for ith_qubit in range(3, num_qubits, 4): ### final decoding at the last Trotter iteration, note that the decoder is with H and flipped ### 3 mod 4
                qc.compose(gate_U_dec_H(connectivity="path",
                                        to_instruction=to_instruction,
                                        add_barrier=add_barrier),
                            qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1],
                            inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="nnn_ring_triangle") if to_instruction else qc


### ================================================== ###
# for adiabatic process


def gate_adiabatic_nnn_ring_prr(num_qubits: int,
                                num_iterations: int,
                                time_evolution: float, ### total time to evolute
                                J1: float = 1.0, ###! by default, not adapted to other values
                                J2: float = 0.5,
                                type_interpolation: str = "exp",
                                eta: float = 1.0,
                                to_instruction: bool = True,
                                add_barrier: bool = False,
                               ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Suzuki-Trotter iterations in PRR paper for the ring next nearest neighbour (path NNN) structure
    https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033107
    """
    assert num_qubits % 4 == 0 ### 0 mod 4

    gate_block_trotter = gate_block_trotter_3cnot ###! 

    if time_evolution is not None:
        time_evolution_per_iteration = time_evolution / num_iterations
        dt = time_evolution_per_iteration ###! this dt is for the dt in each H_eff block, which is now the same as time_evolution_per_iteration
    else:
        dt = Parameter(r"$\Delta t$")

    qc = QuantumCircuit(num_qubits,
                        name="adiabatic_nnn_ring_prr")
    
    for ith_iteration in range(num_iterations + 1): ###! each iteration, here the number of iterations is n + 1
        
        ###! Trotter blocks among direct neighbours ###
        ###
        for ith_qubit in range(0, num_qubits, 2): ### even
            qc.compose(gate_block_trotter(dt=dt * J1,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[ith_qubit, (ith_qubit + 1) % num_qubits],
                        inplace=True,)
        ###
        for ith_qubit in range(1, num_qubits, 2): ### odd
            qc.compose(gate_block_trotter(dt=dt * J1,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[ith_qubit, (ith_qubit + 1) % num_qubits],
                        inplace=True,)
        
        ###! Trotter blocks among next neighbours ###
        J_of_t = compute_J_of_t(t=ith_iteration * dt,
                                J0=J2,
                                time_evolution=time_evolution,
                                type_interpolation=type_interpolation,
                                eta=eta)
        ###
        for ith_qubit in range(1, num_qubits, 4): ### the first swap layer ### 1 mod 4 ###! in this swap: no need for (ith_qubit + 1) % num_qubits
            qc.swap(qubit1=ith_qubit, 
                    qubit2=ith_qubit + 1)
            if add_barrier:
                qc.barrier([ith_qubit, ith_qubit + 1], label="swap")
        ###
        for ith_qubit in range(0, num_qubits, 2): ### the first swapped J2 layer ### 0 mod 2
            qc.compose(gate_block_trotter(dt=dt * J_of_t,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[ith_qubit, ith_qubit + 1],
                        inplace=True,)
        ###
        for ith_qubit in range(1, num_qubits, 2): ### the second swap layer ### 1 mod 2
            qc.swap(qubit1=ith_qubit, 
                    qubit2=(ith_qubit + 1) % num_qubits)
            if add_barrier:
                qc.barrier([ith_qubit, (ith_qubit + 1) % num_qubits], label="swap")
        ###
        for ith_qubit in range(0, num_qubits, 2): ### the second swapped J2 layer ### 0 mod 2
            qc.compose(gate_block_trotter(dt=dt * J_of_t,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                        qubits=[ith_qubit, ith_qubit + 1],
                        inplace=True,)
        ###
        for ith_qubit in range(3, num_qubits, 4): ### the third swap layer ### 3 mod 4
            qc.swap(qubit1=ith_qubit, 
                    qubit2=(ith_qubit + 1) % num_qubits)
            if add_barrier:
                qc.barrier([ith_qubit, (ith_qubit + 1) % num_qubits], label="swap")

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="adiabatic_nnn_ring_prr") if to_instruction else qc


def gate_adiabatic_nnn_ring_triangle(num_qubits: int,
                                     num_iterations: int, ### number of iterations
                                     time_evolution: float, ### total time to evolute
                                     J1: float = 1.0, ###! by default, not adapted to other values
                                     J2: float = 0.5,
                                     type_interpolation: str = "exp",
                                     eta: float = 1.0,
                                     connectivity: str = "complete",
                                     to_instruction: bool = True,
                                     add_barrier: bool = False,
                                    ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    """
    assert num_qubits % 4 == 0 ### 0 mod 4
    # assert np.allclose(J1, 2 * J2) ###! Remark: this function is only for when J1 = 2J2

    if time_evolution is not None:
        time_evolution_per_iteration = time_evolution / num_iterations
        dt = time_evolution_per_iteration ###! this dt is for the dt in each H_eff block, which is now the same as time_evolution_per_iteration
    else:
        dt = Parameter(r"$\Delta t$")

    qc = QuantumCircuit(num_qubits,
                        name="adiabatic_nnn_ring_triangle")

    for ith_iteration in range(num_iterations + 1): ###! each iteration, here the number of iterations is n + 1

        J_of_t = compute_J_of_t(t=ith_iteration * dt,
                                J0=J2,
                                time_evolution=time_evolution,
                                type_interpolation=type_interpolation,
                                eta=eta)
                
        ###
        for ith_qubit in range(2, num_qubits, 4): ### encoder for blue H_eff ### 2 mod 4
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    type_enc_dec=None,
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(0, num_qubits, 4): ### encoder for blue H_eff ### 0 mod 4
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    type_enc_dec=None,
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1], ###!
                        inplace=True,)
        ###
        for ith_qubit in range(1, num_qubits, 4): ### encoder for blue H_eff ### 1 mod 4
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    type_enc_dec="H", ###!
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(3, num_qubits, 4): ### encoder for blue H_eff ### 3 mod 4
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    type_enc_dec="H", ###!
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1], ###!
                        inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="adiabatic_nnn_ring_triangle") if to_instruction else qc


def gate_adiabatic_nnn_ring_triangle_nn(num_qubits: int,
                                        num_iterations: int, ### number of iterations
                                        time_evolution: float, ### total time to evolute
                                        J1: float = 1.0, ###! by default, not adapted to other values
                                        J2: float = 0.5,
                                        type_interpolation: str = "exp",
                                        eta: float = 1.0,
                                        connectivity: str = "complete",
                                        to_instruction: bool = True,
                                        add_barrier: bool = False,
                                       ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    """
    assert num_qubits % 4 == 0 ### 0 mod 4
    # assert np.allclose(J1, 2 * J2) ###! Remark: this function is only for when J1 = 2J2

    if time_evolution is not None:
        time_evolution_per_iteration = time_evolution / num_iterations
        dt = time_evolution_per_iteration ###! this dt is for the dt in each H_eff block, which is now the same as time_evolution_per_iteration
    else:
        dt = Parameter(r"$\Delta t$")

    qc = QuantumCircuit(num_qubits,
                        name="adiabatic_nnn_ring_triangle_nn")

    for ith_iteration in range(num_iterations + 1): ###! each iteration, here the number of iterations is n + 1

        J_of_t = compute_J_of_t(t=ith_iteration * dt,
                                J0=J2,
                                time_evolution=time_evolution,
                                type_interpolation=type_interpolation,
                                eta=eta)
                
        ###
        for ith_qubit in range(2, num_qubits, 4): ### encoder for blue H_eff ### 2 mod 4
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    type_enc_dec=None,
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(0, num_qubits, 4): ### encoder for blue H_eff ### 0 mod 4
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    type_enc_dec=None,
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1], ###!
                        inplace=True,)
        ###
        for ith_qubit in range(1, num_qubits, 4): ### encoder for blue H_eff ### 1 mod 4
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    type_enc_dec="H", ###!
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        ###
        for ith_qubit in range(3, num_qubits, 4): ### encoder for blue H_eff ### 3 mod 4
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    type_enc_dec="H", ###!
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1], ###!
                        inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="adiabatic_nnn_ring_triangle_nn") if to_instruction else qc