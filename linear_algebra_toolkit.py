# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:05:14 2023

@author: mohsen
"""
from scipy.optimize import root
import numpy as np
import cmath

lambda_unitary = np.array([ [1, 1j , 0 , 0],[0, 0, 1j, 1],[0, 0, 1j, -1],[1, -1j, 0, 0] ]) / np.sqrt(2)


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
    values_real = [np.round(z.real, 3) for z in values]
    values_imag = [np.round(z.imag, 3) for z in values]
    values_approx = [values_real[i] + 1j*values_imag[i] for i,_ in enumerate(values)]
    phases = [cmath.phase(x) for x in values_approx]
    
    order = np.argsort(phases)
    values = values[order]
    vecs = np.transpose(vecs.transpose()[order])
    return values, vecs


def orthonormal_matrix_to_special_one(ortho_matrix):
    if np.linalg.det(ortho_matrix) < 0:
        ortho_matrix[0] = ortho_matrix[0] * -1
    return ortho_matrix

def gram_schmidt(matrix):
    orthogonal_basis = np.linalg.qr(matrix)[0]
    return orthogonal_basis

def degeneracy_class(eigenvalues):
    if len(eigenvalues) != 4: raise Exception('Eigenvalues set unsupported!')
    case = None
    rounded_eigvals = np.sort( np.round(eigenvalues, 4) )
    num_distinct_eigenvals = len(np.unique(rounded_eigvals))
    if num_distinct_eigenvals == 4:
        case = (1,1,1,1)
    elif num_distinct_eigenvals == 2:
        case = (2,2)
    elif num_distinct_eigenvals == 1:
        case = (4)
    else:
        raise Exception('Unknown degeneracy case')
        
    # print(case, '\n', eigenvalues)
    return case

def make_vectors_real(vector):
    #vectors should be in the rows of the matrix
    non_zero_ele = vector[np.nonzero( np.round(vector, 4) )][0]
    phase_factor = np.conj(non_zero_ele) / abs(non_zero_ele)  # Compute the phase factor
    real_vector = vector * phase_factor  # Multiply the vector by the phase factor to make it real
    return real_vector

def real_linear_combination(v1, v2):
    def _realization_equations(x):
        r_matrix = np.array([ [abs(z) for z in v] for v in [v1, v2] ])
        phi_matrix = np.array([ [cmath.phase(z) for z in v] for v in [v1, v2] ])
        out_coors = np.zeros(v1.shape[0])
        for i in range(v1.shape[0]):
            for j in range(2):
                out_coors[i] += x[0 + 2*j]*r_matrix[j,i]*np.sin(phi_matrix[j,i] + x[1 + 2*j])    
        return np.concatenate((out_coors,[x[0]**2 + x[2]**2 - 1],[0]))

    roots = root(_realization_equations, [1, 0, 1, 0, 0, 0])
    coeff = roots['x']
    u1 = coeff[0]*np.e**(1j*coeff[1]) * v1 + coeff[2]*np.e**(1j*coeff[3]) * v2
    u2 = make_vectors_real( - coeff[2] * np.e**(-1j*coeff[3]) * v1 + coeff[0]*np.e**(-1j*coeff[1])*v2 )
    u1 = u1 / np.linalg.norm( u1 )
    u2 = u2 / np.linalg.norm( u2 )
    return u1, u2

def unitary_to_orthogonal(unitary):
    orthogonal_matrix = np.zeros_like(unitary)
    for i in [0,2]:
        v1 = unitary[0 + i]
        v2 = unitary[1 + i]
        u1, u2 = real_linear_combination(v1, v2)
        orthogonal_matrix[i] = u1
        orthogonal_matrix[i+1] = u2
    return orthogonal_matrix

def get_orthogonal_basis(complex_symmetric_matrix):
    eigen_values, eigen_vectors = get_ordered_eig(complex_symmetric_matrix)
    
    if degeneracy_class(eigen_values) == (1,1,1,1): #four diff eigenvalues
        orth_matrix = np.copy(eigen_vectors.transpose())
        orth_matrix = orthonormal_matrix_to_special_one(orth_matrix)
    
    elif degeneracy_class(eigen_values) == (2,2): # two distinct degenerate eigenvalues
        eigen_vectors = gram_schmidt(eigen_vectors)
        eigen_vectors = np.copy(eigen_vectors.transpose()) # transpose needed to be consistent with the paper
        orth_matrix = unitary_to_orthogonal(eigen_vectors)
        orth_matrix = orthonormal_matrix_to_special_one(orth_matrix)
    
    elif degeneracy_class(eigen_values) == 4: # one degenerate eigenvalue
        orth_matrix = np.eye(4)
        
    return orth_matrix

def break_rotation_tensor_into_two(one_tensor_two):
    if np.all( np.isclose(one_tensor_two, np.zeros((4,4))) ):
        raise Exception('Null array')

    quater_matrices_arr = np.array( [one_tensor_two[:,:2].reshape(2,2,2), one_tensor_two[:,2:].reshape(2,2,2)] )

    for i,j in [(i,j) for i in range(2) for j in range(2)]:
        if not np.all( np.isclose( quater_matrices_arr[i,j], np.zeros((2,2)) ) ):
            second_part_matrix = strip_global_factor( quater_matrices_arr[i,j] )
            break
        else:
            continue

    nonzero_indices = np.where(np.abs(second_part_matrix) > 1e-6)
    non_zero_row, non_zero_col = nonzero_indices[0][0], nonzero_indices[1][0]

    first_part_matrix = np.array([[one_tensor_two[0 + non_zero_row, 0 + non_zero_col], one_tensor_two[0 + non_zero_row, 2 + non_zero_col]],
                                  [one_tensor_two[2 + non_zero_row, 0 + non_zero_col], one_tensor_two[2 + non_zero_row, 2 + non_zero_col]]])
    first_part_matrix = strip_global_factor(first_part_matrix)
    constructed_prod = strip_global_factor( np.kron(first_part_matrix, second_part_matrix) )
    assert np.all( np.isclose(constructed_prod, one_tensor_two, atol=1e-03 ) ) or np.all( np.isclose(constructed_prod, - one_tensor_two, atol=1e-03 ) )
    return first_part_matrix, second_part_matrix