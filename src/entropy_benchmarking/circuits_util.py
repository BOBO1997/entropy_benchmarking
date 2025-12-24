from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Parameter


### ================================================== ###

def gate_block_trotter_qiskit(dt: float, 
                              to_instruction: bool = True,
                              add_barrier: bool = False,
                             ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """

    qc = QuantumCircuit(2,
                        name="block_trotter_qiskit")
    qc.rzz(theta=2 * dt,
           qubit1=0,
           qubit2=1)
    qc.ryy(theta=2 * dt,
           qubit1=0,
           qubit2=1)
    qc.rxx(theta=2 * dt,
           qubit1=0,
           qubit2=1)

    if add_barrier:
        qc.barrier(label="block_trotter_qiskit")

    return qc.to_instruction(label="block_trotter_qiskit") if to_instruction else qc


def gate_block_trotter_6cnot(dt: float, 
                             to_instruction: bool = True,
                             add_barrier: bool = False,
                            ) -> Union[QuantumCircuit, Instruction]:
    """
    ### ! NOT RECOMMENDED ! ### use gate_block_trotter_qiskit instead.
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    based on https://github.com/qiskit-community/open-science-prize-2021
    """
    
    # Build a subcircuit for XX(t) two-qubit gate
    qc_XX = QuantumCircuit(2, name='XX')
    qc_XX.ry(np.pi/2,[0,1])
    qc_XX.cx(0,1)
    qc_XX.rz(2 * dt, 1)
    qc_XX.cx(0,1)
    qc_XX.ry(-np.pi/2,[0,1])

    # Build a subcircuit for YY(t) two-qubit gate
    qc_YY = QuantumCircuit(2, name='YY')
    qc_YY.rx(np.pi/2,[0,1])
    qc_YY.cx(0,1)
    qc_YY.rz(2 * dt, 1)
    qc_YY.cx(0,1)
    qc_YY.rx(-np.pi/2,[0,1])

    # Build a subcircuit for ZZ(t) two-qubit gate
    qc_ZZ = QuantumCircuit(2, name='ZZ')
    qc_ZZ.cx(0,1)
    qc_ZZ.rz(2 * dt, 1)
    qc_ZZ.cx(0,1)

    qc = QuantumCircuit(2,
                        name="block_trotter_6cnot")
    qc.compose(instruction=qc_ZZ.to_instruction(), qubits=[0,1], inplace=True)
    qc.compose(instruction=qc_YY.to_instruction(), qubits=[0,1], inplace=True)
    qc.compose(instruction=qc_XX.to_instruction(), qubits=[0,1], inplace=True)

    if add_barrier:
        qc.barrier(label="block_trotter_6cnot")

    return qc.to_instruction(label="block_trotter_6cnot") if to_instruction else qc


def gate_block_trotter_3cnot(dt: float,
                             option: str = "b",
                             to_instruction: bool = True,
                             add_barrier: bool = False,
                            ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """

    theta = np.pi / 2 - 2 * dt
    phi = - theta

    theta_negative_t = np.pi / 2 + 2 * dt
    phi_negative_t = - theta_negative_t
    
    qc = QuantumCircuit(2,
                        name="block_trotter_3cnot")

    if option == "a":
        qc.cx(1, 0)

        qc.rz(- theta, 0)
        qc.rz(- np.pi / 2, 1)
        qc.ry(- phi, 1)

        qc.cx(0, 1)

        qc.ry(- theta, 1)
        
        qc.cx(1, 0)

        qc.rz(np.pi / 2, 0)

    elif option == "b":
        qc.cx(0, 1)

        qc.rz(- np.pi / 2, 0)
        qc.ry(- phi, 0)
        qc.rz(- theta, 1)

        qc.cx(1, 0)

        qc.ry(- theta, 0)
        
        qc.cx(0, 1)

        qc.rz(np.pi / 2, 1)

    elif option == "c":
        qc.cx(1, 0)

        qc.rz(- np.pi / 2, 1)
        qc.ry(- phi, 1)
        qc.rz(- theta, 0)

        qc.cx(0, 1)

        qc.ry(- theta, 1)
        
        qc.cx(1, 0)

        qc.rz(np.pi / 2, 0)

    elif option == "d":
        qc.rz(- np.pi / 2, 1)

        qc.cx(0, 1)

        qc.ry(theta_negative_t, 0)

        qc.cx(1, 0)

        qc.rz(theta_negative_t, 1)
        qc.ry(phi_negative_t, 0)
        qc.rz(np.pi / 2, 0)

        qc.cx(0, 1)
        
    else:
        raise Exception
    
    if add_barrier:
        qc.barrier(label="block_trotter_3cnot")

    return qc.to_instruction(label="block_trotter_3cnot") if to_instruction else qc


### ================================================== ###


def gate_U_enc(connectivity: str = "complete",
               to_instruction: bool = True,
               add_barrier: bool = False,
              ) -> Union[QuantumCircuit, Instruction]:
    """
    ### for blue and red
    big endian: following the QuantumCircuit instance which adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    qc = QuantumCircuit(3)

    if connectivity == "complete":
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
    elif connectivity == "path":
        qc.cx(0, 1)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 1)
        qc.cx(1, 2)
    else:
        raise Exception("invalid connectivity")

    if add_barrier:
        qc.barrier(label="U_enc")

    return qc.to_instruction(label="U_enc") if to_instruction else qc


def gate_U_dec(connectivity: str = "complete",
               to_instruction: bool = True,
               add_barrier: bool = False,
              ) -> Union[QuantumCircuit, Instruction]:
    """
    ### for blue and red
    big endian: following the QuantumCircuit instance which adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    qc = QuantumCircuit(3)

    if connectivity == "complete":
        qc.cx(2, 0)
        qc.cx(1, 2)
        qc.cx(0, 1)
    elif connectivity == "path":
        qc.cx(1, 2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 1)
        qc.cx(0, 1)
    else:
        raise Exception("invalid connectivity")
    
    if add_barrier:
        qc.barrier(label="U_dec")

    return qc.to_instruction(label="U_dec") if to_instruction else qc


def gate_U_enc_H(connectivity: str = "complete",
                 to_instruction: bool = True,
                 add_barrier: bool = False,
                ) -> Union[QuantumCircuit, Instruction]:
    """
    ### for yellow and green
    big endian: following the QuantumCircuit instance which adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    qc = QuantumCircuit(3)

    if connectivity == "complete":
        qc.cx(1, 0)
        qc.cx(2, 1)
        qc.cx(0, 2)
    elif connectivity == "path":
        qc.cx(1, 0)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 1)
    else:
        raise Exception("invalid connectivity")

    if add_barrier:
        qc.barrier(label="U_enc_H")

    return qc.to_instruction(label="U_enc_H") if to_instruction else qc


def gate_U_dec_H(connectivity: str = "complete",
                 to_instruction: bool = True,
                 add_barrier: bool = False,
                ) -> Union[QuantumCircuit, Instruction]:
    """
    ### for yellow and green
    big endian: following the QuantumCircuit instance which adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    qc = QuantumCircuit(3)

    if connectivity == "complete":
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.cx(1, 0)
    elif connectivity == "path":
        qc.cx(2, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(1, 0)
    else:
        raise Exception("invalid connectivity")
    
    if add_barrier:
        qc.barrier(label="U_dec_H")

    return qc.to_instruction(label="U_dec_H") if to_instruction else qc


### ================================================== ###


def gate_12_23_J31(dt: float,
                   J31: float = 1,
                   to_instruction: bool = True,
                   add_barrier: bool = False,
                  ) -> Union[QuantumCircuit, Instruction]:
    """
    exp(-i(2XX+ZZ)dt), used in the proposed method
    """
    qc = QuantumCircuit(2)

    if np.allclose(J31, 1):

        qc.sxdg(0) ### sqrt(X)^{dag}
        qc.sxdg(1) ### sqrt(X)^{dag}

        qc.cx(0, 1)
        qc.rx(4 * dt, 0)
        qc.rz(2 * dt, 1)
        qc.cx(0, 1)

        qc.sx(0) ### sqrt(X)
        qc.sx(1) ### sqrt(X)

    else:
        
        qc.cx(1, 0)

        qc.rz(2 * (J31 - 1) * dt - np.pi / 2, 0)

        qc.rx(2 * dt, 1)
        qc.rz(- np.pi / 2, 1)
        qc.ry(np.pi / 2 - 2 * J31 * dt, 1)

        qc.cx(0, 1)

        qc.ry(2 * J31 * dt - np.pi / 2, 1)

        qc.cx(1, 0)

        qc.rz(np.pi / 2, 0)

    ###! force not to add barrier
    # if add_barrier:
    #     qc.barrier()

    return qc.to_instruction(label="12_23_J31") if to_instruction else qc


def gate_H_eff_triangle(dt: float,
                        J31: float = 1,
                        to_instruction: bool = True,
                        add_barrier: bool = False,
                       ) -> Union[QuantumCircuit, Instruction]:
    """
    dt: qiskit._accelerate.circuit.Parameter
    Create and return the circuit instruction of the one trotter step with rotation angle `dt`.
    """

    qc = QuantumCircuit(2)

    ### V_enc
    qc.ry(- 1 * np.pi / 4, 0) ###! cos(\pi/8) + i sin (\pi/8 Y)
    qc.ry(- 1 * np.pi / 4, 1) ###! cos(\pi/8) + i sin (\pi/8 Y)
    
    ### further trotter, exp(-i (Z_1 + Z_2) t / sqrt(2))
    qc.rz(np.sqrt(2) * dt, 0) ### exp(-i Z_1 t / sqrt(2))
    qc.rz(np.sqrt(2) * dt, 1) ### exp(-i Z_2 t / sqrt(2))

    ###! force not to add barrier
    # if add_barrier:
    #     qc.barrier()
    
    qc.compose(gate_12_23_J31(dt=dt,
                              J31=J31,
                              to_instruction=to_instruction,
                              add_barrier=add_barrier),
                qubits=[0, 1],
                inplace=True,)

    ### further trotter, exp(-i (Z_1 + Z_2) t / sqrt(2))
    qc.rz(np.sqrt(2) * dt, 0) ### exp(-i Z_1 t / sqrt(2))
    qc.rz(np.sqrt(2) * dt, 1) ### exp(-i Z_2 t / sqrt(2))
    
    ### V_dnc = V_enc^{dag}
    qc.ry(1 * np.pi / 4, 0) ###! cos(\pi/8) - i sin (\pi/8 Y)
    qc.ry(1 * np.pi / 4, 1) ###! cos(\pi/8) - i sin (\pi/8 Y)

    if add_barrier:
        qc.barrier(label="H_eff_triangle")

    return qc.to_instruction(label="H_eff_triangle") if to_instruction else qc


def gate_block_trotter_triangle(dt: float,
                                J31: float = 1.0,
                                type_enc_dec: str = None,
                                connectivity: str = "complete",
                                to_instruction: bool = True,
                                add_barrier: bool = False,
                               ) -> Union[QuantumCircuit, Instruction]:
    """
    J31: coefficient of edge 13 in the triangular 1-2-3, where the coefficients of 1-2 and 2-3 are 1.
    """
    
    if type_enc_dec == "H":
        gate_U_enc_temp = gate_U_enc_H
        gate_U_dec_temp = gate_U_dec_H
    else:
        gate_U_enc_temp = gate_U_enc
        gate_U_dec_temp = gate_U_dec

    qc = QuantumCircuit(3,
                        name="block_trotter_triangle")
    qc.compose(gate_U_enc_temp(connectivity=connectivity,
                               to_instruction=to_instruction,
                               add_barrier=add_barrier),
                qubits=[jth_qubit for jth_qubit in range(0, 3)],
                inplace=True,)
    qc.compose(gate_H_eff_triangle(dt=dt,
                                   J31=J31,
                                   to_instruction=to_instruction,
                                   add_barrier=add_barrier),
                qubits=[jth_qubit for jth_qubit in range(0, 2)],
                inplace=True,)
    qc.compose(gate_U_dec_temp(connectivity=connectivity,
                               to_instruction=to_instruction,
                               add_barrier=add_barrier),
                qubits=[jth_qubit for jth_qubit in range(0, 3)],
                inplace=True,)
    if add_barrier:
        qc.barrier(label="block_trotter_triangle")

    return qc.to_instruction(label="block_trotter_triangle") if to_instruction else qc


### ================================================== ###
# for adiabatic process


def gate_state_ground_triangle_ring(num_qubits: int,
                                    to_instruction: bool = True,
                                    add_barrier: bool = False,
                                   ) -> Union[QuantumCircuit, Instruction]:
    """
    This function applies to only J2=1/2 nnn and kagome ring.
    Tensor product of |01> - |10> state
    # Tensor product of |10> - |01> state

    The state_initial should be the string in big endian.

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """

    assert not (num_qubits & 1)

    qc = QuantumCircuit(num_qubits) ### |0>|0>
    
    for ith_qubit in range(num_qubits)[::2]:
        qc.x(ith_qubit) ### |1>|0>
        qc.h(ith_qubit) ### |->|0> = |0>|0> - |1>|0>
        qc.cx(ith_qubit, ith_qubit + 1) ### |0>|0> - |1>|1>
        qc.x(ith_qubit + 1) ### |0>|1> - |1>|0>
        # qc.x(ith_qubit) ### |1>|0> - |0>|1>

    if add_barrier:
        qc.barrier(label="state_ground_triangle_ring")

    return qc.to_instruction(label="state_ground_triangle_ring") if to_instruction else qc


def compute_J_of_t(t: float,
                   J0: float,
                   time_evolution: float,
                   type_interpolation: str = "exp",
                   eta: float = None,
                  ) -> float:
    """
    J(t) = 1/2 + (2J_0 - 1) * e^{-\eta (T-t)} / 2
    ### eta should be small enough, and
    ### eta * time_evolution should be large enough.
    ### Thus, time_evolution should be eta ** d, where d < -1
    ### in this function, we set time_evolution = eta ** (-2) by default
    """
    if eta is None:
        eta = 1 / np.sqrt(time_evolution) ###! by default
    if type_interpolation == "exp":
        J_of_t = 1 / 2 + (2 * J0 - 1) * np.exp(- eta * (time_evolution - t)) / 2
    elif type_interpolation == "linear":
        J_of_t = 1 / 2 + (2 * J0 - 1) * (np.abs(t) / time_evolution) / 2
    else:
        raise Exception("specify a valid type_interpolation")
    return J_of_t