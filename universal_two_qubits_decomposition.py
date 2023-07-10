# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:31:35 2023

@author: mohsen
"""

from scipy.stats import unitary_group
import scipy.linalg as la
import cmath
import numpy as np

from functions import *
# from functions import give_v_circuit, arbitary_single_qubit_circuit, get_inverse_circuit
from linear_algebra_toolkit import *
# import functions

from pyquil import get_qc, Program
from pyquil.api import get_qc, BenchmarkConnection
from forest.benchmarking.randomized_benchmarking import generate_rb_sequence
from pyquil.quil import *
from pyquil.gates import *
from pyquil.simulation.tools import lifted_gate, program_unitary, lifted_gate_matrix

# lambda_unitary = np.array([ [1, 1j , 0 , 0],[0, 0, 1j, 1],[0, 0, 1j, -1],[1, -1j, 0, 0] ]) / np.sqrt(2)
# zero_state = np.array([[1],[0]])
# one_state = np.array([[0],[1]])




# def find_phi_theta_omega(single_rot):
#     cos_theta_2 = np.round(abs(single_rot[0,0]), decimals=3)
#     theta = 2*np.arccos(cos_theta_2)
#     phi_plus_omega_2 = cmath.phase(single_rot[1,1])
#     phi_minus_omega_2 = - cmath.phase(single_rot[1,0])
#     phi = phi_plus_omega_2 + phi_minus_omega_2
#     omega =  phi_plus_omega_2 - phi_minus_omega_2
#     return phi, theta, omega

# def get_corresponding_entangling_part(magic_u, target_qubits):
#     u_u_T = np.dot(magic_u, magic_u.transpose())
#     u_u_T_eigen_values, u_u_T_eigen_vectors = get_ordered_eig(u_u_T)
    
#     eigen_values_phases = [cmath.phase(x) for x in u_u_T_eigen_values]
#     alpha, beta, delta = np.array([eigen_values_phases[0] + eigen_values_phases[1],
#                                     eigen_values_phases[0] + eigen_values_phases[2],
#                                     eigen_values_phases[1] + eigen_values_phases[2] ]) / 2
#     v_circuit = give_v_circuit(alpha, beta, delta, qubits = target_qubits)
#     return v_circuit

# def get_program_of_single_unitary(single_qubit_unitary_matrix, target_qubit):
#     phi, theta, omega = find_phi_theta_omega(single_qubit_unitary_matrix)
#     return arbitary_single_qubit_circuit(phi, theta, omega, qubit = target_qubit)


# def get_matrix_of_single_member_two_design_two_qubits():
#     bm = BenchmarkConnection()
#     sequences = generate_rb_sequence(bm, qubits=[0,1], depth=2)
#     prog = sequences[:-1][0]
#     mat = program_unitary(prog, n_qubits=2)
#     return mat

# def two_design_two_qubits_packs_generator(qmachine, target_qubits, num_layer:int):
#     list_gates = []
#     total_mat = np.eye(4,4)
#     for index in range(num_layer):
#         mat = get_matrix_of_single_member_two_design_two_qubits()
#         total_mat = np.matmul(mat, total_mat)
#         draft_circuit = get_corresponding_universal_circuit(mat, target_qubits)
#         # list_gates.extend( qmachine.compiler.quil_to_native_quil(draft_circuit) )
#         list_gates.extend( draft_circuit )
    
#     inverse_mat = np.linalg.inv(total_mat)
#     inverse_circuit = get_corresponding_universal_circuit(inverse_mat, target_qubits)
#     list_gates = [ ins for ins in list_gates if isinstance(ins, Gate)]
#     list_gates.extend( inverse_circuit )
        
#     return list_gates


# def get_corresponding_universal_circuit(u_matrix, target_qubits):
#     u_matrix = strip_global_factor(u_matrix)
#     print(u_matrix)
#     u_matrix *= np.e**(-1j*np.pi/4)
    
#     u_magic_matrix = matrix_in_magic_basis(u_matrix)
#     u_u_T = np.dot(u_magic_matrix, u_magic_matrix.transpose())
#     u_u_T_eigen_values, u_u_T_eigen_vectors = get_ordered_eig(u_u_T)
    
    
#     eigen_values_phases = [cmath.phase(x) for x in u_u_T_eigen_values]
#     alpha, beta, delta = np.array([eigen_values_phases[0] + eigen_values_phases[1],
#                                     eigen_values_phases[0] + eigen_values_phases[2],
#                                     eigen_values_phases[1] + eigen_values_phases[2] ]) / 2
#     v_circuit = give_v_circuit(alpha, beta, delta, qubits = target_qubits)
#     v_circuit_zero_one = give_v_circuit(alpha, beta, delta, qubits = [0,1])
    
#     v_matrix = program_unitary(v_circuit_zero_one, n_qubits=2)
#     v_magic_matrix = matrix_in_magic_basis(v_matrix)
    
    
#     v_v_T = np.dot(v_magic_matrix, v_magic_matrix.transpose())
#     v_v_T_eigen_values, v_v_T_eigen_vectors = get_ordered_eig(v_v_T)
    
    
#     k_matrix = get_orthogonal_basis(v_v_T)
#     l_matrix = get_orthogonal_basis(u_u_T)
    
#     assert np.all( np.isclose( l_matrix.dot(l_matrix.T), np.eye(4,4), atol = 1e-04) )
#     assert np.all( np.isclose( k_matrix.dot(k_matrix.T), np.eye(4,4), atol = 1e-04) )
    
#     assert np.isclose( np.linalg.det(k_matrix), 1, atol = 1e-04)
#     assert np.isclose( np.linalg.det(l_matrix), 1, atol = 1e-04)
        
#     a_tensor_b = matrix_out_magic_basis( np.matmul( v_magic_matrix.conjugate().transpose(),
#                                                     np.matmul(k_matrix.transpose(), np.matmul(l_matrix, u_magic_matrix))) )
#     a,b = break_rotation_tensor_into_two(a_tensor_b)
#     prog_a, prog_b = get_program_of_single_unitary(a, target_qubit = target_qubits[0]), get_program_of_single_unitary(b, target_qubit = target_qubits[1])
    
#     c_tensor_d = matrix_out_magic_basis( np.matmul(l_matrix.transpose(), k_matrix) )
#     c,d = break_rotation_tensor_into_two(c_tensor_d)
#     prog_c, prog_d = get_program_of_single_unitary(c, target_qubit = target_qubits[0]), get_program_of_single_unitary(d, target_qubit = target_qubits[1])
    
#     prog = Program(prog_a, prog_b, v_circuit, prog_c, prog_d)
#     return  prog


if __name__ == '__main__':
    qmachine = get_qc("2q-qvm")
    pack_mat = two_design_two_qubits_packs_generator(qmachine, [0,1], 10)
    # circuit = get_corresponding_universal_circuit(pack_mat, [0,1])
    u_matrix = get_matrix_of_single_member_two_design_two_qubits()
    # u_matrix = unitary_group.rvs(4)
    circuit = get_corresponding_universal_circuit(u_matrix, target_qubits=[0,1])
    
    # program = two_design_two_qubits_packs_generator(qmachine, [0,1], 2)
    
    