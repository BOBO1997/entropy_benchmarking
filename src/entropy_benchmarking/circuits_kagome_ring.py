from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter
from entropy_benchmarking.circuits_util import (
    gate_U_enc, 
    gate_U_dec,
    gate_block_trotter_qiskit,
    gate_block_trotter_6cnot,
    gate_block_trotter_3cnot,
    gate_block_trotter_triangle,
    compute_J_of_t,
)


### ================================================== ###


def gate_kagome_ring_conventional(num_qubits: int, 
                                  num_iterations: int, 
                                  dt: float,
                                  type_block: str = "3cnot",
                                  to_instruction: bool = True,
                                  add_barrier: bool = False,
                                 ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    conventional Suzuki-Trotter iterations for kagome_ring structure
    """

    ### choose the type of Trotter block
    if type_block == "qiskit":
        gate_block_trotter = gate_block_trotter_qiskit
    elif type_block == "6cnot":
        gate_block_trotter = gate_block_trotter_6cnot
    elif type_block == "3cnot":
        gate_block_trotter = gate_block_trotter_3cnot
    else:
        raise Exception("specify a valid type for the Trotter block")

    qc = QuantumCircuit(num_qubits,
                        name="conventional_kagome_ring")
    
    for ith_iteration in range(num_iterations):

        ### Trotter blocks among direct neighbours
        for ith_qubit in range(num_qubits):
            if not (ith_qubit & 1): ### even
                if ith_qubit == num_qubits - 1:
                    qargs_one_next = [ith_qubit, 0]
                else:
                    qargs_one_next = [ith_qubit, ith_qubit + 1]
                qc.compose(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits=qargs_one_next,
                            inplace=True,)
        for ith_qubit in range(num_qubits):
            if ith_qubit & 1: ### odd
                if ith_qubit == num_qubits - 1:
                    qargs_one_next = [ith_qubit, 0]
                else:
                    qargs_one_next = [ith_qubit, ith_qubit + 1]
                qc.compose(gate_block_trotter(dt=dt,
                                            to_instruction=to_instruction,
                                            add_barrier=add_barrier), 
                            qubits=qargs_one_next,
                            inplace=True,)
                    
        ### Trotter blocks among next neighbours: kagome ring structure
        if not (num_qubits & 1): ### should be even
            for ith_qubit in range(num_qubits):
                if ith_qubit % 4 == 0: ### 0 mod 4
                    if ith_qubit == num_qubits - 2:
                        qargs_two_next = [ith_qubit, 0]
                    else:
                        qargs_two_next = [ith_qubit, ith_qubit + 2]
                    qc.compose(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits=qargs_two_next,
                            inplace=True,)
            for ith_qubit in range(num_qubits):
                if ith_qubit % 4 == 2: ### 2 mod 4
                    if ith_qubit == num_qubits - 2:
                        qargs_two_next = [ith_qubit, 0]
                    else:
                        qargs_two_next = [ith_qubit, ith_qubit + 2]
                    qc.compose(gate_block_trotter(dt=dt,
                                                to_instruction=to_instruction,
                                                add_barrier=add_barrier), 
                            qubits=qargs_two_next,
                            inplace=True,)

        if add_barrier:
            qc.barrier(str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="kagome_ring_conventional") if to_instruction else qc


def gate_kagome_ring_triangle(num_qubits: int,
                              num_iterations: int,
                              dt: float,
                              connectivity: str = "complete",
                              to_instruction: bool = True,
                              add_barrier: bool = False,
                             ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    proposed Suzuki-Trotter iterations for kagome_ring structure
    """
    qc = QuantumCircuit(num_qubits,
                        name="kagome_ring_triangle")

    for ith_iteration in range(num_iterations):

        ###? blue: upper triangle
        for ith_qubit in range(num_qubits):
            if ith_qubit % 4 == 2:
                qc.compose(gate_block_trotter_triangle(dt=dt,
                                                       connectivity=connectivity,
                                                       to_instruction=to_instruction,
                                                       add_barrier=add_barrier),
                           qubits=[ith_qubit, ith_qubit + 1, (ith_qubit + 2) % num_qubits],
                           inplace=True,)
        
        ###! red: lower triangle
        for ith_qubit in range(num_qubits):
            if ith_qubit % 4 == 0:
                qc.compose(gate_block_trotter_triangle(dt=dt,
                                                       connectivity=connectivity,
                                                       to_instruction=to_instruction,
                                                       add_barrier=add_barrier),
                           qubits=[ith_qubit, ith_qubit + 1, (ith_qubit + 2) % num_qubits][::-1],
                           inplace=True,)
                
        if add_barrier:
            qc.barrier(str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="kagome_ring_triangle") if to_instruction else qc


### ================================================== ###
# for adiabatic process


def gate_adiabatic_kagome_ring_triangle(num_qubits: int,
                                        num_iterations: int, ### number of iterations
                                        time_evolution: Union[float, None], ### total time to evolute
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
                        name="trotter_adiabatic_kagome_ring")

    for ith_iteration in range(num_iterations + 1): ###! each iteration

        J_of_t = compute_J_of_t(t=ith_iteration * dt,
                                J0=J2,
                                time_evolution=time_evolution,
                                type_interpolation=type_interpolation,
                                eta=1)

        ###? blue: upper triangle
        for ith_qubit in range(2, num_qubits, 4):
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)],
                        inplace=True,)
        
        ###! red: lower triangle
        for ith_qubit in range(0, num_qubits, 4):
            qc.compose(gate_block_trotter_triangle(dt=dt / 2,
                                                    J31=2 * J_of_t,
                                                    connectivity=connectivity,
                                                    to_instruction=to_instruction,
                                                    add_barrier=add_barrier),
                        qubits=[jth_qubit % num_qubits for jth_qubit in range(ith_qubit, ith_qubit + 3)][::-1], ###!
                        inplace=True,)

        if add_barrier:
            qc.barrier(label=str(ith_iteration+1)+"-th iteration")

    return qc.to_instruction(label="trotter_adiabatic_kagome_ring") if to_instruction else qc