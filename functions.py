# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:09:28 2021

@author: mohsen
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
from tqdm import tqdm_notebook as tqdm
import _pickle as cPickle
import os
import itertools

from eigenvalues_distribution import generate_haar_random_eigenvalues_two_qubits
from linear_algebra_toolkit import *
# from universal_two_qubits_decomposition import *
# import universal_two_qubits_decomposition
from noise_models import *

from pyquil import get_qc, Program
from pyquil.api import get_qc, BenchmarkConnection
from forest.benchmarking.randomized_benchmarking import generate_rb_sequence
from pyquil.quil import *
from pyquil.gates import RX, RZ, CZ, I
from pyquil.simulation.tools import lifted_gate, program_unitary, lifted_gate_matrix


lambda_unitary = np.array([ [1, 1j , 0 , 0],[0, 0, 1j, 1],[0, 0, 1j, -1],[1, -1j, 0, 0] ]) / np.sqrt(2)

def find_phi_theta_omega(single_rot):
    """
    Calculate the values of phi, theta, and omega for a single-qubit rotation matrix.

    Args:
        single_rot (np.ndarray): Single-qubit rotation matrix.

    Returns:
        tuple: phi, theta, omega values.
    """
    cos_theta_2 = np.round(abs(single_rot[0, 0]), decimals=3)
    if cos_theta_2 > 1:
        raise Exception('Not a rotation matrix')
    theta = 2 * np.arccos(cos_theta_2)
    phi_plus_omega_2 = cmath.phase(single_rot[1, 1])
    phi_minus_omega_2 = -cmath.phase(single_rot[1, 0])
    phi = phi_plus_omega_2 + phi_minus_omega_2
    omega = phi_plus_omega_2 - phi_minus_omega_2
    return phi, theta, omega


def get_corresponding_entangling_part(magic_u, target_qubits):
    """
    Get the corresponding entangling part of a magic unitary matrix.

    Args:
        magic_u (np.ndarray): Magic unitary matrix.
        target_qubits (list): List of target qubits.

    Returns:
        Program: Corresponding entangling part as a Quil program.
    """
    u_u_T = np.dot(magic_u, magic_u.transpose())
    u_u_T_eigen_values, u_u_T_eigen_vectors = get_ordered_eig(u_u_T)

    eigen_values_phases = [cmath.phase(x) for x in u_u_T_eigen_values]
    alpha, beta, delta = np.array(
        [eigen_values_phases[0] + eigen_values_phases[1], eigen_values_phases[0] + eigen_values_phases[2],
         eigen_values_phases[1] + eigen_values_phases[2]]) / 2
    v_circuit = give_v_circuit(alpha, beta, delta, qubits=target_qubits)
    return v_circuit


def get_program_of_single_unitary(single_qubit_unitary_matrix, target_qubit):
    """
    Get the Quil program corresponding to a single-qubit unitary matrix.

    Args:
        single_qubit_unitary_matrix (np.ndarray): Single-qubit unitary matrix.
        target_qubit (int): Target qubit index.

    Returns:
        Program: Corresponding Quil program.
    """
    phi, theta, omega = find_phi_theta_omega(single_qubit_unitary_matrix)
    return arbitary_single_qubit_circuit(phi, theta, omega, qubit=target_qubit)


def get_matrix_of_single_member_two_design_two_qubits():
    """
    Generate a matrix of a single member of the two-design on two qubits.

    Returns:
        np.ndarray: Matrix of a single member of the two-design on two qubits.
    """
    bm = BenchmarkConnection()
    sequences = generate_rb_sequence(bm, qubits=[0, 1], depth=2)
    prog = sequences[:-1][0]
    mat = program_unitary(prog, n_qubits=2)
    return mat

def two_design_two_qubits_packs_generator(qmachine, target_qubits, num_layer: int):
    """
    Generate gate sequences using the two-design on two qubits.

    Args:
        qmachine: Quantum machine.
        target_qubits (list): List of target qubits.
        num_layer (int): Number of layers.

    Returns:
        list: List of gates representing the gate sequence.
    """
    list_gates = []
    # total_mat = np.eye(4, 4)
    for index in range(num_layer):
        still_failed_question = True
        while still_failed_question:
            try:
                mat = get_matrix_of_single_member_two_design_two_qubits()
                # total_mat = np.matmul(mat, total_mat)
                draft_circuit = get_corresponding_universal_circuit(mat, target_qubits)
                still_failed_question = False
            except:
                continue
        
        list_gates.extend(draft_circuit)

    list_gates = [ins for ins in list_gates if isinstance(ins, Gate)]
    list_gates.extend(get_inverse_circuit(qmachine, list_gates))

    return list_gates

# def two_design_two_qubits_packs_generator(qmachine, target_qubits, num_layer:int):
#     list_gates = []
#     total_mat = np.eye(4,4)
#     for index in range(num_layer):
#         mat = get_matrix_of_single_member_two_design_two_qubits()
#         total_mat = np.matmul(mat, total_mat)
#         draft_circuit = get_corresponding_universal_circuit(mat, target_qubits)
#         # list_gates.extend( qmachine.compiler.quil_to_native_quil(draft_circuit) )
#         list_gates.extend( draft_circuit )
    
#     # inverse_mat = np.linalg.inv(total_mat)
    
#     # inverse_circuit = Program( get_inverse_circuit(qmachine, list_gates) )
#     # inverse_circuit = program_unitary(program, n_qubits)
    
#     # inverse_circuit = get_corresponding_universal_circuit(inverse_mat, target_qubits)
#     # list_gates = [ ins for ins in list_gates if isinstance(ins, Gate)]
#     # list_gates.extend( inverse_circuit )
    
#     list_gates = [ ins for ins in list_gates if isinstance(ins, Gate)]
#     list_gates.extend( get_inverse_circuit(qmachine, list_gates) )
        
#     return list_gates



def native_universal_two_qubits_packs_generator(qmachine, target_qubits: list, num_layer: int):
    """
    Generate gate sequences using the native universal two-qubit gates.

    Args:
        qmachine: Quantum machine.
        target_qubits (list): List of target qubits.
        num_layer (int): Number of layers.

    Returns:
        list: List of gates representing the gate sequence.
    """
    list_gates = []
    for index in range(num_layer):
        draft_circuit = give_random_two_qubit_circuit(target_qubits)
        list_gates.extend(draft_circuit)
    list_gates = [ins for ins in list_gates if isinstance(ins, Gate)]
    list_gates.extend(get_inverse_circuit(qmachine, list_gates))
    return list_gates


def native_rigetti_single_qubit_packs_generator(qmachine, target_qubit, num_layer: int):
    """
    Generate gate sequences using native Rigetti single-qubit gates.

    Args:
        qmachine: Quantum machine.
        target_qubit (int): Target qubit index.
        num_layer (int): Number of layers.

    Returns:
        list: List of gates representing the gate sequence.
    """
    try:
        temp = iter(target_qubit)
        if len(target_qubit) == 1:
            target_qubit = target_qubit[0]
        else:
            raise ValueError('target qubit should be only one index')
    except:
        pass

    list_gates = []
    angles = np.linspace(0, np.pi, 100)

    for index in range(0, num_layer):
        omega, phi = np.random.uniform(0, 2 * np.pi, size=2)
        theta = np.random.choice(angles, p=np.sin(angles) / np.sum(np.sin(angles)))

        list_gates.extend(arbitary_single_qubit_circuit(omega, theta, phi, target_qubit))

    list_gates.extend(get_inverse_circuit(qmachine, list_gates))
    return list_gates

def get_matrix_of_single_member_two_design_one_qubit():
    """
    Generate a matrix of a single member of the two-design on one qubit.

    Returns:
        np.ndarray: Matrix of a single member of the two-design on one qubit.
    """
    bm = BenchmarkConnection()
    sequences = generate_rb_sequence(bm, qubits=[0], depth=2)
    prog = sequences[:-1][0]
    mat = program_unitary(prog, n_qubits=1)
    return mat

def two_design_single_qubit_packs_generator(qmachine, target_qubit, num_layer: int):
    
    
    if type(target_qubit) == list:
        if len(target_qubit) == 1:
            target_qubit = target_qubit[0]
        else:
            raise ValueError('Only pass one qubit!')
    else:
        ValueError('Indicate one qubit surrounded by [] like [0]')
        
    list_gates = []

    for index in range(0, num_layer):
        mat = get_matrix_of_single_member_two_design_one_qubit()

        list_gates.extend(get_program_of_single_unitary(mat, target_qubit))
        
    list_gates = [ins for ins in list_gates if isinstance(ins, Gate)]
    list_gates.extend(get_inverse_circuit(qmachine, list_gates))
    return list_gates


# def two_design_single_qubit_packs_generator_non_uniform(qmachine, target_qubit, num_layer: int):
#     """
#     Generate non-uniform gate sequences using the two-design on a single qubit.

#     Args:
#         qmachine: Quantum machine.
#         target_qubit (int): Target qubit index.
#         num_layer (int): Number of layers.

#     Returns:
#         list: List of gates representing the gate sequence.
#     """
#     bm = BenchmarkConnection()

#     sequences = generate_rb_sequence(bm, qubits=target_qubit, depth=num_layer)
#     gates_list = []

#     for prog in sequences:
#         gates_list.extend(prog.instructions)

#     return gates_list


bench_protocol_func_dict = {'native_conditional_single_qubit':native_rigetti_single_qubit_packs_generator,
                           'native_conditional_conditional_two_qubits':native_universal_two_qubits_packs_generator,
                           'standard_rb_single_qubit':two_design_single_qubit_packs_generator,
                           'standard_rb_two_qubits':two_design_two_qubits_packs_generator}



def get_corresponding_universal_circuit(u_matrix, target_qubits):
    """
    Constructs the corresponding universal circuit for a given unitary matrix.

    Args:
        u_matrix (numpy.ndarray): The unitary matrix.
        target_qubits (list): The indices of the target qubits.

    Returns:
        pyquil.Program: The corresponding universal circuit.

    Raises:
        AssertionError: If the matrices `l_matrix` and `k_matrix` do not satisfy the closeness conditions.

    Note:
        The function assumes that the input unitary matrix `u_matrix` is a valid unitary operator.

    The universal circuit construction process involves the following steps:
    1. Preprocessing:
        - The global phase of the `u_matrix` is stripped.
        - A phase factor is applied to the `u_matrix`.

    2. Magic Basis Transformation:
        - The `u_matrix` is transformed into the magic basis using the `matrix_in_magic_basis` function.

    3. Eigenvalue Decomposition:
        - The matrix product `u_u_T` is computed, where `u_u_T` is the dot product of `u_magic_matrix` with its transpose.
        - The eigenvalues and eigenvectors of `u_u_T` are computed using the `get_ordered_eig` function.

    4. Parameter Extraction:
        - The phases of the eigenvalues are extracted and used to calculate `alpha`, `beta`, and `delta` parameters.

    5. V Circuit Generation:
        - Two V circuits are generated using the `give_v_circuit` function with different sets of qubits.

    6. Magic Basis Transformation of V Matrix:
        - The V matrix is calculated from the V circuit using the `program_unitary` function.
        - The V matrix is transformed into the magic basis using the `matrix_in_magic_basis` function.

    7. Eigenvalue Decomposition of V Matrix:
        - The matrix product `v_v_T` is computed, where `v_v_T` is the dot product of `v_magic_matrix` with its transpose.
        - The eigenvalues and eigenvectors of `v_v_T` are computed using the `get_ordered_eig` function.

    8. Orthogonal Basis Calculation:
        - The `k_matrix` is calculated using the `get_orthogonal_basis` function on `v_v_T`.
        - The `l_matrix` is calculated using the `get_orthogonal_basis` function on `u_u_T`.

    9. Closeness Check:
        - The `l_matrix` and `k_matrix` are checked for closeness to the identity matrix using `np.isclose`.
        - If the closeness conditions are not satisfied, a secondary method is attempted to ensure closeness.

    10. Decomposition of A-B Tensor:
        - The tensor product `a_tensor_b` is calculated using matrix operations.
        - The `a_tensor_b` tensor is decomposed into `a` and `b` using the `break_rotation_tensor_into_two` function.

    11. Program Generation for A and B:
        - Programs `prog_a` and `prog_b` are generated using the `get_program_of_single_unitary` function.

    12. Decomposition of C-D Tensor:
        - The tensor product `c_tensor_d` is calculated using matrix operations.
        - The `c_tensor_d` tensor is decomposed into `c` and `d` using the `break_rotation_tensor_into_two` function.

    13. Program Generation for C and D:
        - Programs `prog_c` and `prog_d` are generated using the `get_program_of_single_unitary` function.

    14. Universal Circuit Construction:
        - The individual sub-circuits are combined into the universal circuit using the `pyquil.Program` class.

    Finally, the constructed universal circuit is returned as the output.

    """
    u_matrix = strip_global_factor(u_matrix)
    # print(u_matrix)
    u_matrix *= np.e**(-1j*np.pi/4)
    
    u_magic_matrix = matrix_in_magic_basis(u_matrix)
    u_u_T = np.dot(u_magic_matrix, u_magic_matrix.transpose())
    u_u_T_eigen_values, u_u_T_eigen_vectors = get_ordered_eig(u_u_T)
    
    
    eigen_values_phases = [cmath.phase(x) for x in u_u_T_eigen_values]
    alpha, beta, delta = np.array([eigen_values_phases[0] + eigen_values_phases[1],
                                    eigen_values_phases[0] + eigen_values_phases[2],
                                    eigen_values_phases[1] + eigen_values_phases[2] ]) / 2
    v_circuit = give_v_circuit(alpha, beta, delta, qubits = target_qubits)
    v_circuit_zero_one = give_v_circuit(alpha, beta, delta, qubits = [0,1])
    
    v_matrix = program_unitary(v_circuit_zero_one, n_qubits=2)
    v_magic_matrix = matrix_in_magic_basis(v_matrix)
    
    
    v_v_T = np.dot(v_magic_matrix, v_magic_matrix.transpose())
    v_v_T_eigen_values, v_v_T_eigen_vectors = get_ordered_eig(v_v_T)
    
    
    k_matrix = get_orthogonal_basis(v_v_T)
    l_matrix = get_orthogonal_basis(u_u_T)
    
    try:
        assert np.all( np.isclose( l_matrix.dot(l_matrix.T), np.eye(4,4), atol = 1e-03) )
        assert np.all( np.isclose( k_matrix.dot(k_matrix.T), np.eye(4,4), atol = 1e-03) )
        
    except:
        try:
            k_matrix = unitary_to_orthogonal(k_matrix)
            l_matrix = unitary_to_orthogonal(l_matrix)
            
            k_matrix = orthonormal_matrix_to_special_one(k_matrix)
            l_matrix = orthonormal_matrix_to_special_one(l_matrix)
            
            assert np.all( np.isclose( l_matrix.dot(l_matrix.T), np.eye(4,4), atol = 1e-03) )
            assert np.all( np.isclose( k_matrix.dot(k_matrix.T), np.eye(4,4), atol = 1e-03) )
            
        except:
            print('Sounds eigenvectors contain comparable imaginary part! Quantlity of decompositon decreased.')
            print(l_matrix)
            print(k_matrix)
            raise Exception('I did not make this!')
    a_tensor_b = matrix_out_magic_basis( np.matmul( v_magic_matrix.conjugate().transpose(),
                                                    np.matmul(k_matrix.transpose(), np.matmul(l_matrix, u_magic_matrix))) )
    a,b = break_rotation_tensor_into_two(a_tensor_b)
    prog_a, prog_b = get_program_of_single_unitary(a, target_qubit = target_qubits[0]), get_program_of_single_unitary(b, target_qubit = target_qubits[1])
    
    c_tensor_d = matrix_out_magic_basis( np.matmul(l_matrix.transpose(), k_matrix) )
    c,d = break_rotation_tensor_into_two(c_tensor_d)
    prog_c, prog_d = get_program_of_single_unitary(c, target_qubit = target_qubits[0]), get_program_of_single_unitary(d, target_qubit = target_qubits[1])
    
    prog = Program(prog_a, prog_b, v_circuit, prog_c, prog_d)
    return  prog


def averageOfFidelity(Data_Array):
    """
    Calculates the average fidelity from a given array of data.

    Args:
        Data_Array (numpy.ndarray): The array of fidelity data.

    Returns:
        float: The average fidelity.

    The function calculates the average fidelity by taking the mean value of the fidelity data array along axis 1.

    """
    k_m, n_m = Data_Array.shape
    Pacc_Array = np.average(Data_Array, axis=1)
    Pacc = np.average(Pacc_Array)
    return Pacc


def stdOfFidelity(Data_Array):
    """
    Calculates the standard deviation of fidelity from a given array of data.

    Args:
        Data_Array (numpy.ndarray): The array of fidelity data.

    Returns:
        float: The standard deviation of fidelity.

    The function calculates the standard deviation of fidelity by taking the standard deviation of the fidelity data array along axis 1.

    """
    k_m, n_m = Data_Array.shape
    Pacc_Array = np.average(Data_Array, axis=1)
    Pacc_err = np.std(Pacc_Array)
    return Pacc_err


def decay_func(m, p, a0, b0):
    """
    Calculates the decay function value for a given set of parameters.

    Args:
        m (float): The decay parameter.
        p (float): The input parameter.
        a0 (float): The decay coefficient a0.
        b0 (float): The decay coefficient b0.

    Returns:
        float: The value of the decay function.

    The decay function is defined as a linear combination of `p` and the decay coefficients `a0` and `b0`. The value `m` determines the exponent of `p` in the decay function.

    """
    return a0 * p**m + b0


def decay_param(p, qubit_num):
    """
    Calculates the decay parameter for a given probability and number of qubits.

    Args:
        p (float): The input probability.
        qubit_num (int): The number of qubits.

    Returns:
        float: The decay parameter.

    The decay parameter is determined based on the input probability `p` and the number of qubits. It uses the formula `r = (d-1)*(1-p)/d`, where `d` is `2` raised to the power of `qubit_num`.

    """
    d = 2**qubit_num
    r = (d - 1) * (1 - p) / d
    return r


def qvirtual_machine(given_program):
    """
    Simulates the execution of a given program on a quantum virtual machine (QVM).

    Args:
        given_program (pyquil.Program): The program to be executed on the QVM.

    Returns:
        numpy.ndarray: The measured outcome of the program execution.

    The function creates a quantum virtual machine based on the number of qubits in the given program. It compiles the program and runs it on the QVM. The result is obtained, and the measured outcome is returned as a numpy array.

    """
    n_qubits = given_program.get_qubits()
    qc = get_qc(str(n_qubits) + 'q-qvm')

    executable = qc.compile(given_program)
    result = qc.run(executable)
    measured_outcome = result.readout_data.get('ro')

    return measured_outcome

def qreal_machine(given_program):
    n_qubits = given_program.get_qubits()
    qc = get_qc(  'Aspen-11',execution_timeout=60, compiler_timeout=100 )
    
    executable = qc.compile(given_program)
    result = qc.run(executable)
    measured_outcome = result.readout_data.get('ro')

    return measured_outcome

def daggered_gate(gate):
    """
    Returns the daggered version of a given gate.

    Args:
        gate (Gate): The gate for which the daggered version is needed.

    Returns:
        Gate: The daggered version of the input gate.

    The function checks the name of the gate and returns the corresponding daggered gate. For gates CZ, CNOT, and H, the function returns the same gate. For XY, S, RX, RY, and RZ gates, the function calculates the negative angle and returns the corresponding daggered gate. If the gate name is not supported, a ValueError is raised.

    """
    if gate.name in ['CZ', 'CNOT', 'H', 'I']:
        return gate
    elif gate.name == 'XY':
        angle = gate.params[0]
        return XY(angle=-angle, q1=gate.qubits[0].index, q2=gate.qubits[1].index)
    elif gate.name == 'S':
        return PHASE(-np.pi/2, gate.qubits[0].index)
    elif gate.name == 'RX':
        angle = gate.params[0]
        return RX(-angle, qubit=gate.qubits[0].index)
    elif gate.name == 'RY':
        angle = gate.params[0]
        return RY(-angle, qubit=gate.qubits[0].index)
    elif gate.name == 'RZ':
        angle = gate.params[0]
        return RZ(-angle, qubit=gate.qubits[0].index)
    else:
        raise ValueError("This gate daggered is not yet considered!")


def g_gate(control, target):
    """
    Creates a G-gate program with two qubits.

    Args:
        control (int): The index of the control qubit.
        target (int): The index of the target qubit.

    Returns:
        Program: The G-gate program.

    The G-gate program is created by applying a series of RZ and CZ gates with specific angles on the control and target qubits.

    """
    return Program(RZ(-3*np.pi/2, qubit=control), RZ(-3*np.pi/2, qubit=target),
                   CZ(control=target, target=control),
                   RZ(-np.pi, qubit=control), RZ(-np.pi, qubit=target))


def arbitary_single_qubit_circuit(phi, theta, omega, qubit):
    """
    Creates an arbitrary single-qubit circuit program.

    Args:
        phi (float): The rotation angle around the z-axis.
        theta (float): The rotation angle around the x-axis.
        omega (float): The rotation angle around the z-axis.
        qubit (int): The index of the target qubit.

    Returns:
        Program: The single-qubit circuit program.

    The program applies a series of RZ, RX, and RZ gates with the specified angles on the target qubit.

    """
    draft_circuit = Program([RZ(phi, qubit=qubit),
                            RX(np.pi/2, qubit=qubit),
                            RZ(theta, qubit=qubit),
                            RX(-np.pi/2, qubit=qubit),
                            RZ(omega, qubit=qubit)])
    return draft_circuit


def give_random_single_qubit_gate(qubit):
    """
    Generates a random single-qubit gate program.

    Args:
        qubit (int): The index of the target qubit.

    Returns:
        Program: The random single-qubit gate program.

    The function generates random values for the phi and omega angles within the range [0, 2*pi]. The theta angle is randomly selected from a distribution that makes the distribution of angles normalized and centered around pi. The program is then created using the arbitary_single_qubit_circuit function.

    """
    phi, omega = np.random.uniform(0, 2*np.pi, size=2)

    theta_range = np.linspace(0, np.pi)
    p_theta = np.sin(theta_range) / np.sum(np.sin(theta_range))
    theta = np.random.choice(theta_range, p=p_theta)
    return arbitary_single_qubit_circuit(phi, theta, omega, qubit=qubit)


def normalized_abs_angle_dist(angle_range):
    """
    Calculates the normalized absolute angle distribution.

    Args:
        angle_range (numpy.ndarray): The range of angles.

    Returns:
        numpy.ndarray: The normalized absolute angle distribution.

    The function calculates the absolute difference between pi and each angle in the angle range. It then normalizes the distribution to ensure that the sum of all values is 1.

    """
    dist = np.pi - np.abs(np.pi - angle_range)
    dist /= np.sum(dist)
    return dist



# def give_v_circuit(alpha, beta, delta, qubits = [0,1]):
    # prog = Program(CNOT(control=qubits[1], target=qubits[0]),
    #                RZ(angle = delta, qubit =qubits[0]),
    #                RY(beta, qubit =qubits[1]),
    #                CNOT(control=qubits[0], target=qubits[1]))
    # prog += Program(RY(angle= alpha, qubit = qubits[1]),
    #                 CNOT(control=qubits[1], target=qubits[0]))

    # return prog

def give_v_circuit(alpha, beta, delta, qubits=[0, 1]):
    """
    Constructs a V-circuit program given the angles alpha, beta, and delta.

    Args:
        alpha (float): The angle alpha.
        beta (float): The angle beta.
        delta (float): The angle delta.
        qubits (List[int]): The indices of the target qubits. Default is [0, 1].

    Returns:
        Program: The V-circuit program.

    The V-circuit program is created by applying a sequence of gates with specific angles on the specified qubits.

    """
    prog = Program(RZ(np.pi, qubits[0]), RX(np.pi/2, qubits[0]), RZ(np.pi/2, qubits[0]), RX(-np.pi/2, qubits[0]),
                   CZ(control=qubits[1], target=qubits[0]),
                   I(qubits[0]), I(qubits[1]),
                   RZ(np.pi, qubits[0]), RX(np.pi/2, qubits[0]), RZ(np.pi/2, qubits[0]), RX(-np.pi/2, qubits[0]), RZ(delta, qubits[0]),
                   RX(np.pi/2, qubits[1]), RZ(np.pi/2 + beta, qubits[1]), RX(np.pi/2, qubits[1]),
                   CZ(control=qubits[0], target=qubits[1]),
                   I(qubits[0]), I(qubits[1]))
    prog += Program(RZ(np.pi, qubits[1]), RX(np.pi/2, qubits[1]), RZ(np.pi/2 + alpha, qubits[1]), RX(-np.pi/2, qubits[1]),
                    RZ(np.pi, qubits[0]), RX(np.pi/2, qubits[0]), RZ(np.pi/2, qubits[0]), RX(-np.pi/2, qubits[0]),
                    CZ(control=qubits[1], target=qubits[0]),
                    I(qubits[0]), I(qubits[1]),
                    RZ(np.pi, qubits[0]), RX(np.pi/2, qubits[0]), RZ(np.pi/2, qubits[0]), RX(-np.pi/2, qubits[0]))
    return prog

def used_qubits_index(gates_sequence):
    """
    Returns the indices of the qubits used in the gate sequence.

    Args:
        gates_sequence (List[Gate]): The sequence of gates.

    Returns:
        List[int]: The indices of the qubits used in the gate sequence.

    The function extracts the qubit indices from each gate in the sequence and returns the unique sorted indices.

    """
    qubits = np.array([np.array(gate.qubits) for gate in gates_sequence])
    qubits = np.array([ele.index for sub_arr in qubits for ele in sub_arr])  # some gates might have multiple indices
    qubits_indices = np.unique(qubits)
    qubits_indices.sort()
    return [int(x) for x in qubits_indices]

def give_random_two_qubit_circuit(qubits):
    """
    Generates a random two-qubit circuit program.

    Args:
        qubits (List[int]): The indices of the target qubits.

    Returns:
        Program: The random two-qubit circuit program.

    The function generates random single-qubit gates for each qubit using the give_random_single_qubit_gate function. It then generates random eigenvalues for the angles alpha, beta, and delta using the generate_haar_random_eigenvalues_two_qubits function. The V-circuit program is constructed using the give_v_circuit function and the single-qubit gates.

    """
    a, b, c, d = [give_random_single_qubit_gate(qubit=qubit) for _ in range(2) for qubit in qubits]
    phi_one, phi_two, phi_three, phi_four = generate_haar_random_eigenvalues_two_qubits()

    alpha, beta, delta = [(phi_one + phi_two)/2, (phi_one + phi_three)/2, (phi_two + phi_three)/2]

    prog = Program(a, b)
    prog += give_v_circuit(alpha, beta, delta, qubits=qubits)
    prog += Program(c, d)
    return prog


def extrapolate_decay_func(layers_arr, avg_fdlty_arr):
    """
    Extrapolates the decay function parameters.

    Args:
        layers_arr (numpy.ndarray): The array of layers.
        avg_fdlty_arr (numpy.ndarray): The array of average fidelities.

    Returns:
        tuple: A tuple containing the parameters and covariance of the extrapolated decay function.

    The function fits the decay function to the layers and average fidelities using the curve_fit function from scipy.optimize. It returns the optimized parameters and covariance.

    """
    try:
        popt, pcov = curve_fit(decay_func, layers_arr, avg_fdlty_arr, bounds=([0, 0, 0], [1., 1., 1.]))
    except:
        popt, pcov = curve_fit(decay_func, layers_arr, avg_fdlty_arr, bounds=([0, 0, 0], [1., 1., 1.]))
    return popt, pcov


def plot_decay(layers_arr, avg_fdlty_arr, err_fdlty_arr, label: str, *args, **kwargs):
    """
    Plots the decay of average fidelities.

    Args:
        layers_arr (numpy.ndarray): The array of layers.
        avg_fdlty_arr (numpy.ndarray): The array of average fidelities.
        err_fdlty_arr (numpy.ndarray): The array of fidelity errors.
        label (str): The label for the plot.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    The function plots the decay of average fidelities with error bars. It extrapolates the decay function parameters using the extrapolate_decay_func function and plots the extrapolated function curve. The plot is customized using the provided arguments and keyword arguments.

    """
    fmt = kwargs.get('fmt', 'o')

    if 'axes' in kwargs:
        axes = kwargs.get('axes')
    else:
        fig = plt.figure()
        axes = fig.add_subplot()

    popt, pcov = extrapolate_decay_func(layers_arr, avg_fdlty_arr)

    err = axes.errorbar(layers_arr, avg_fdlty_arr, yerr=err_fdlty_arr, fmt=fmt, capsize=4, capthick=2)
    err[-1][0].set_linestyle('--')
    err[-1][0].set_alpha(0.5)
    between_layers = np.arange(layers_arr.min(), layers_arr.max()+1, 1).astype('int')
    axes.plot(between_layers, decay_func(between_layers, *popt), color=err[0].get_color(),
              label=label + ':' + r'${1}*{0}^m+{2}$'.format(*np.round(popt, 4)))

    plt.xlabel('Depth', fontsize=18)
    plt.ylabel('Average of Fidelity', fontsize=16)

    axes.legend()


def convert_measured_to_response_matrix(measured_outcome):
    """
    Converts the measured outcome to a response matrix.

    Args:
        measured_outcome (numpy.ndarray): The measured outcome array.

    Returns:
        numpy.ndarray: The response matrix.

    The function calculates the response matrix by summing the measured outcome along each row and applying a boolean operation to convert the result to 1 if it is equal to the n_zero state.

    """
    return 1 - np.bool_(np.sum(measured_outcome, axis=1))


def run_bench_experiment(qmachine, program, number_of_shots):
    """
    Runs a benchmark experiment using the given quantum machine and program.

    Args:
        qmachine (QVMConnection or QPUConnection): The quantum machine connection.
        program (Program): The program to be executed.
        number_of_shots (int): The number of shots for each measurement.

    Returns:
        numpy.ndarray: The measured outcome array.

    The function wraps the program in a numshots loop, compiles and runs it on the quantum machine, and returns the measured outcome array.

    """
    program = program.wrap_in_numshots_loop(number_of_shots)

    # Run the program
    # executable = qmachine.compile(program)
    # result = qmachine.run(executable)
    result = qmachine.run(program)
    measured_outcome = result.readout_data.get('ro')
    return measured_outcome


def get_inverse_circuit(qmachine, gates_sequence):
    """
    Constructs the inverse circuit of the input gates sequence.

    Args:
        qmachine (QVMConnection or QPUConnection): The quantum machine connection.
        gates_sequence (iterable): The sequence of circuit gates.

    Returns:
        numpy.ndarray: The array of gates constructing the inverse circuit.

    The function iterates over the gates in reverse order, applies the daggered_gate function to each gate, and constructs the inverse circuit. The inverse circuit gates are converted to native Quil instructions using the qmachine compiler.

    """
    target_qubits = used_qubits_index(gates_sequence)
    n_qubits = len(target_qubits)

    prog = Program()
    for gate in reversed(gates_sequence):
        prog += daggered_gate(gate)
    prog_daggered_native = qmachine.compiler.quil_to_native_quil(prog)
    instructions = prog_daggered_native.instructions
    inverting_gates_list = [ins for ins in instructions if isinstance(ins, Gate)]
    return np.array(inverting_gates_list)


def save_experiment(experiment, protocol_name, target_qubits, layer_num, num_of_sequences):
    """
    Saves the experiment to a pickle file.

    Args:
        experiment: The experiment object to be saved.
        protocol_name (str): The name of the protocol.
        target_qubits (int): The target qubits.
        layer_num (int): The layer number.
        num_of_sequences (int): The number of sequences.

    Returns:
        str: The file path of the saved experiment.

    The function creates a directory for the protocol and target qubits if it does not exist. It saves the experiment object to a pickle file with the format "L{}_K{}.pickle" where L is the layer number and K is the number of sequences.

    """
    path = os.path.join(os.getcwd(), 'experiments_warehouse', protocol_name, str(target_qubits))
    try:
        os.makedirs(path)
    except:
        pass

    file_path = os.path.join(path, 'L{}_K{}.pickle'.format(layer_num, num_of_sequences))

    with open(file_path, "wb") as output_file:
        cPickle.dump(experiment, output_file)
    return file_path


def catch_experiments(qmachine, target_qubits: list, protocol_name: str, layers_num: int, exp_num: int):
    """
    Retrieves or generates the experiments for the given target qubits, protocol, number of layers, and experiment number.

    Args:
        qmachine (QVMConnection or QPUConnection): The quantum machine connection.
        target_qubits (list): The list of target qubits.
        protocol_name (str): The name of the protocol.
        layers_num (int): The number of layers.
        exp_num (int): The number of experiments.

    Returns:
        list: The list of experiments.

    The function checks if the experiments file exists. If it does, the function retrieves the experiments from the file. Otherwise, it generates the experiments using the specified circuit generation function, saves them to the file, and returns the list of experiments.

    """
    file_path = os.path.join(os.getcwd(), 'experiments_warehouse', protocol_name,
                             str(target_qubits), 'L{}_K{}.pickle'.format(layers_num, exp_num))

    if os.path.isfile(file_path):  # If such experiment file exists
        with open(file_path, "rb") as input_file:
            exps = cPickle.load(input_file)
    else:  # If it does not exist
        circuit_gen_func = bench_protocol_func_dict[protocol_name]
        exps = generate_experiments(qmachine, target_qubits, circuit_gen_func, layers_num, exp_num)
        save_experiment(exps, protocol_name, target_qubits, layers_num, exp_num)
    return exps


def generate_experiments(qmachine, target_qubits: list, circuit_gen_func, layers_num: int, exp_num: int):
    """
    Generates a list of experiments for the given target qubits, circuit generation function, number of layers, and experiment number.

    Args:
        qmachine (QVMConnection or QPUConnection): The quantum machine connection.
        target_qubits (list): The list of target qubits.
        circuit_gen_func: The circuit generation function.
        layers_num (int): The number of layers.
        exp_num (int): The number of experiments.

    Returns:
        list: The list of experiments.

    The function generates the experiments by calling the circuit generation function for the specified number of times. Each generated experiment is appended to a list, which is then returned.

    """
    n_qubits = len(target_qubits)
    exp_list = []
    for i in tqdm(range(exp_num), desc='exp. generation'):
        exp_list.append(circuit_gen_func(qmachine, target_qubits, layers_num))

    return exp_list


def find_machine_response(qmachine, rb_experiments, number_of_shots):
    """
    Finds the machine response by running the RB experiments on the given quantum machine.

    Args:
        qmachine (QVMConnection or QPUConnection): The quantum machine connection.
        rb_experiments (list): The list of RB experiments.
        number_of_shots (int): The number of shots for each measurement.

    Returns:
        numpy.ndarray: The response matrix.

    The function constructs the program for each RB experiment, preserves the gates from optimization, performs measurements, and converts the measured outcome to the response matrix.

    """
    target_qubits = used_qubits_index(rb_experiments[0])
    n_qubits = len(target_qubits)
    sequ_num = len(rb_experiments)
    response_matrix = np.zeros((sequ_num, number_of_shots))
    
    damping_per_CZ = 0.01
    for i_sequ, sequ in enumerate(tqdm(rb_experiments, desc='Examing the seq.')):
        prog = Program()  # All qubits begin with |0> state
        ro = prog.declare('ro', 'BIT', n_qubits)
        for gate in sequ:
            prog += gate
        
        corrupted_CZ = append_kraus_to_gate(
        tensor_kraus_maps(
            dephasing_kraus_map(damping_per_CZ),
            dephasing_kraus_map(damping_per_CZ)
        ),
        np.diag([1, 1, 1, -1]))
        
        # Measurements
        for ind, qubit_ind in enumerate(target_qubits):
            prog += MEASURE(qubit_ind, ro[ind])
            
        prog.define_noisy_gate("CZ", target_qubits, corrupted_CZ)

        # Do not let the quilc alter the gates by optimization        
        prog = Program('PRAGMA PRESERVE_BLOCK') + prog
        prog += Program('PRAGMA END_PRESERVE_BLOCK')
        
        response = convert_measured_to_response_matrix(run_bench_experiment(qmachine, prog, number_of_shots))
        response_matrix[i_sequ, :] = np.copy(response)
    return response_matrix



if __name__ == '__main__':
    qmachine = get_qc("2q-qvm")
    # pack_mat = two_design_two_qubits_packs_generator(qmachine, [0,1], 10)
    # circuit = get_corresponding_universal_circuit(pack_mat, [0,1])
    # u_matrix = get_matrix_of_single_member_two_design_two_qubits()
    u_matrix = [[-0.35355339+0.35355339j, -0.35355339+0.35355339j,  0.35355339+0.35355339j, -0.35355339-0.35355339j],
                [ 0.35355339+0.35355339j,  0.35355339+0.35355339j, -0.35355339+0.35355339j, 0.35355339-0.35355339j],
                [ 0.35355339-0.35355339j, -0.35355339+0.35355339j, -0.35355339-0.35355339j, -0.35355339-0.35355339j],
                [-0.35355339-0.35355339j,  0.35355339+0.35355339j,  0.35355339-0.35355339j, 0.35355339-0.35355339j]]
    # u_matrix = unitary_group.rvs(4)
    circuit = get_corresponding_universal_circuit(u_matrix, target_qubits=[0,1])
    
    # program = two_design_two_qubits_packs_generator(qmachine, [0,1], 2)