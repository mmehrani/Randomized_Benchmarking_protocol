# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:31:35 2023

@author: mohsen
"""

from scipy.stats import unitary_group
import cmath
import numpy as np
from functions import give_v_circuit, arbitary_single_qubit_circuit

from pyquil import get_qc, Program
from pyquil.api import get_qc, BenchmarkConnection
from forest.benchmarking.randomized_benchmarking import generate_rb_sequence
from pyquil.quil import *
from pyquil.gates import *
from pyquil.simulation.tools import lifted_gate, program_unitary, lifted_gate_matrix

lambda_unitary = np.array([ [1, 1j , 0 , 0],[0, 0, 1j, 1],[0, 0, 1j, -1],[1, -1j, 0, 0] ]) / np.sqrt(2)
zero_state = np.array([[1],[0]])
one_state = np.array([[0],[1]])

def partial_trace_on_left(one_tensor_two):
    states_id_mat = [np.kron(state, np.eye(2)) for state in [zero_state, one_state]]
    tr_one_dot_two =  np.zeros((2,2))
    for vec_id in states_id_mat:
        tr_one_dot_two = tr_one_dot_two +  np.matmul(vec_id.conj().transpose(), np.matmul(one_tensor_two, vec_id))
    return tr_one_dot_two

def partial_trace_on_right(one_tensor_two):
    states_id_mat = [np.kron(np.eye(2), state) for state in [zero_state, one_state]]
    one_dot_tr_two =  np.zeros((2,2))
    for vec_id in states_id_mat:
        one_dot_tr_two = one_dot_tr_two + np.matmul(vec_id.conj().transpose(), np.matmul(one_tensor_two, vec_id))
    return one_dot_tr_two

def orthonormal_matrix_to_special_one(ortho_matrix):
    if np.linalg.det(ortho_matrix) < 0:
        ortho_matrix[0] = ortho_matrix[0] * -1
    return ortho_matrix

def matrix_in_magic_basis(matrix):
    return np.matmul( lambda_unitary.conj().transpose(), np.matmul(matrix, lambda_unitary) )

def matrix_out_magic_basis(magic_matrix):
    return np.matmul( lambda_unitary, np.matmul(magic_matrix, lambda_unitary.conj().transpose()) )

def phase_distance(complex_arr:np.array):
    phases = np.array( [cmath.phase(x) for x in complex_arr] )
    phases = np.sort(phases)
    phases = [phases[i+1] - phases[i] for i,x in enumerate(phases[:-1])]
    return phases

def strip_global_factor(matrix):
    shape_length = np.shape(matrix)[0]
    return matrix / np.linalg.det(matrix)**(1/shape_length)


def get_ordered_eig(matrix):
    values, vecs = np.linalg.eig(matrix)
    order = np.argsort([cmath.phase(x) for x in values])
    values = values[order]
    vecs = np.transpose(vecs.transpose()[order])
    return values, vecs

def find_phi_theta_omega(single_rot):
    cos_theta_2 = abs(single_rot[0,0])
    theta = 2*np.arccos(cos_theta_2)
    phi_plus_omega_2 = cmath.phase(single_rot[1,1])
    phi_minus_omega_2 = - cmath.phase(single_rot[1,0])
    phi = phi_plus_omega_2 + phi_minus_omega_2
    omega =  phi_plus_omega_2 - phi_minus_omega_2
    return phi, theta, omega

def get_corresponding_entangling_part(magic_u, target_qubits):
    u_u_T = np.dot(magic_u, magic_u.transpose())
    u_u_T_eigen_values, u_u_T_eigen_vectors = get_ordered_eig(u_u_T)
    
    eigen_values_phases = [cmath.phase(x) for x in u_u_T_eigen_values]
    alpha, beta, delta = np.array([eigen_values_phases[0] + eigen_values_phases[1],
                                   eigen_values_phases[0] + eigen_values_phases[2],
                                   eigen_values_phases[1] + eigen_values_phases[2] ]) / 2
    v_circuit = give_v_circuit(alpha, beta, delta, qubits = target_qubits)
    return v_circuit

def get_program_of_single_unitary(single_qubit_unitary_matrix, target_qubit):
    phi, theta, omega = find_phi_theta_omega(single_qubit_unitary_matrix)
    return arbitary_single_qubit_circuit(phi, theta, omega, qubit = target_qubit)


def get_single_parts_of_tensor_prod(x_tensor_y):
    x = strip_global_factor(partial_trace_on_right(x_tensor_y))
    y = strip_global_factor(partial_trace_on_left(x_tensor_y))
    return x,y

def get_matrix_of_single_member_two_design_two_qubits():
    bm = BenchmarkConnection()
    
    sequences = generate_rb_sequence(bm, qubits=[0,1], depth=2)
    prog = sequences[:-1][0]
    mat = program_unitary(prog, n_qubits=2)
    return mat

def two_design_two_qubits_packs_generator_uni(qmachine, target_qubits, num_layer:int):
    mat = get_matrix_of_single_member_two_design_two_qubits()
    program = get_corresponding_universal_circuit(mat, target_qubits)
    return program


def get_corresponding_universal_circuit(u_matrix, target_qubits):
    u_matrix = strip_global_factor(u_matrix)
    u_matrix *= np.e**(-1j*np.pi/4)
    
    u_magic_matrix = matrix_in_magic_basis(u_matrix)
    u_u_T = np.dot(u_magic_matrix, u_magic_matrix.transpose())
    u_u_T_eigen_values, u_u_T_eigen_vectors = get_ordered_eig(u_u_T)
    
    v_circuit = get_corresponding_entangling_part(u_magic_matrix, target_qubits)
    v_circuit_zero_one = get_corresponding_entangling_part(u_magic_matrix, [0,1]) #program_unitary works with zero and one only bad luck!
    v_matrix = program_unitary(v_circuit_zero_one, n_qubits=2)
    v_magic_matrix = matrix_in_magic_basis(v_matrix)
    v_v_T = np.dot(v_magic_matrix, v_magic_matrix.transpose())
    v_v_T_eigen_values, v_v_T_eigen_vectors = get_ordered_eig(v_v_T)
    
    k_matrix = np.copy(v_v_T_eigen_vectors.transpose()) # transpose needed to be consistent with the paper
    l_matrix = np.copy(u_u_T_eigen_vectors.transpose())
    k_matrix = orthonormal_matrix_to_special_one(k_matrix)
    l_matrix = orthonormal_matrix_to_special_one(l_matrix)
    
    a_tensor_b = matrix_out_magic_basis( np.matmul( v_magic_matrix.conjugate().transpose(),
                                                    np.matmul(k_matrix.transpose(), np.matmul(l_matrix, u_magic_matrix))) )
    a,b = get_single_parts_of_tensor_prod(a_tensor_b)
    prog_a, prog_b = get_program_of_single_unitary(a, target_qubit = target_qubits[0]), get_program_of_single_unitary(b, target_qubit = target_qubits[1])
    
    c_tensor_d = matrix_out_magic_basis( np.matmul(l_matrix.transpose(), k_matrix) )
    c,d = get_single_parts_of_tensor_prod(c_tensor_d)
    prog_c, prog_d = get_program_of_single_unitary(c, target_qubit = target_qubits[0]), get_program_of_single_unitary(d, target_qubit = target_qubits[1])
    
    prog = Program(prog_a, prog_b, v_circuit, prog_c, prog_d)
    return  prog


if __name__ == '__main__':
    qmachine = get_qc("2q-qvm")
    # pack_mat = two_design_two_qubits_packs_generator_uni(qmachine, [0,1], 5)
    # circuit = get_corresponding_universal_circuit(pack_mat, [0,1])
    mat = get_matrix_of_single_member_two_design_two_qubits()
    program = get_corresponding_universal_circuit(mat, [0,1])