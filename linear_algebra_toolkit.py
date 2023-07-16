# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:05:14 2023

@author: mohsen
"""
from scipy.optimize import root, fsolve
import numpy as np
import cmath

# The magic basis for a 2-qubit system
lambda_unitary = np.array([ [1, 1j , 0 , 0],[0, 0, 1j, 1],[0, 0, 1j, -1],[1, -1j, 0, 0] ]) / np.sqrt(2)


def matrix_in_magic_basis(matrix):
    """
    Transform the given matrix from the standard basis to the magic basis.
    This is done using a change of basis matrix, lambda_unitary in this case.
    The magic basis is used in quantum information theory.

    Args:
        matrix (np.ndarray): The matrix to be transformed, given in the standard basis.

    Returns:
        np.ndarray: The matrix transformed into the magic basis.
    """
    return np.matmul( lambda_unitary.conj().transpose(), np.matmul(matrix, lambda_unitary) )


def matrix_out_magic_basis(magic_matrix):
    """
    Transform the given matrix from the magic basis to the standard basis.
    This is the reverse operation of `matrix_in_magic_basis`.

    Args:
        magic_matrix (np.ndarray): The matrix to be transformed, given in the magic basis.

    Returns:
        np.ndarray: The matrix transformed into the standard basis.
    """
    return np.matmul( lambda_unitary, np.matmul(magic_matrix, lambda_unitary.conj().transpose()) )


def phase_distance(complex_arr:np.array):
    """
    Calculate the phase differences between adjacent elements in the sorted array of complex numbers.
    The phase of a complex number is the angle it makes with the positive real axis.

    Args:
        complex_arr (np.ndarray): An array of complex numbers.

    Returns:
        list: A list of phase differences between adjacent elements in the sorted array.
    """
    phases = np.array( [cmath.phase(x) for x in complex_arr] )
    phases = np.sort(phases)
    phases = [phases[i+1] - phases[i] for i,x in enumerate(phases[:-1])]
    return phases


def strip_global_factor(matrix):
    """
    Remove the global phase factor from a matrix.
    The global phase factor is a complex number of modulus 1 that multiplies all the entries of a matrix.

    Args:
        matrix (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The matrix with its global phase factor stripped.
    """
    shape_length = np.shape(matrix)[0]
    return matrix / np.linalg.det(matrix)**(1/shape_length)


def get_ordered_eig(matrix):
    """
    Calculate the eigenvalues and eigenvectors of the given matrix.
    Then, sort them by the phase of the eigenvalues.

    Args:
        matrix (np.ndarray): The input matrix.

    Returns:
        tuple: A tuple containing two lists: the first list contains the ordered eigenvalues, and the second list contains the corresponding eigenvectors.
    """
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
    """
    Converts an orthonormal matrix to a special orthogonal matrix (determinant 1).
    If the determinant of the input matrix is -1, one of its rows is multiplied by -1.

    Args:
        ortho_matrix (np.ndarray): The orthonormal matrix.

    Returns:
        np.ndarray: The special orthogonal matrix.
    """
    if np.linalg.det(ortho_matrix) < 0:
        ortho_matrix[0] = ortho_matrix[0] * -1
    return ortho_matrix


def gram_schmidt(matrix):
    """
    Applies the Gram-Schmidt process to the input matrix to produce an orthogonal basis.
    The Gram-Schmidt process orthogonalizes a set of vectors in an inner product space.

    Args:
        matrix (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The matrix with an orthogonal basis.
    """
    orthogonal_basis = np.linalg.qr(matrix)[0]
    return orthogonal_basis


def degeneracy_class(eigenvalues):
    """
    Determines the degeneracy class of a set of eigenvalues.
    The degeneracy class is a tuple indicating the multiplicity of each distinct eigenvalue.

    Args:
        eigenvalues (np.ndarray): The set of eigenvalues.

    Returns:
        tuple: A tuple indicating the degeneracy class.
    """
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
        
    return case


def make_vectors_real(vector):
    """
    Converts a complex vector into a real vector by removing the phase factor.
    This is done by multiplying the vector by the conjugate of a non-zero element divided by its absolute value.

    Args:
        vector (np.ndarray): The complex vector.

    Returns:
        np.ndarray: The real vector.
    """
    non_zero_ele = vector[np.nonzero( np.round(vector, 4) )][0]
    phase_factor = np.conj(non_zero_ele) / abs(non_zero_ele)  # Compute the phase factor
    real_vector = vector * phase_factor  # Multiply the vector by the phase factor to make it real
    return real_vector


def real_linear_combination(v1, v2):
    """
    Returns a real linear combination of two complex vectors.
    This is done by solving a system of equations which imposes the realness condition.

    Args:
        v1 (np.ndarray), v2 (np.ndarray): The complex vectors.

    Returns:
        tuple: A tuple of two real vectors.
    """
    def _realization_equations(x):
        r_matrix = np.array([ [abs(z) for z in v] for v in [v1, v2] ])
        phi_matrix = np.array([ [cmath.phase(z) for z in v] for v in [v1, v2] ])
        out_coors = np.zeros(v1.shape[0])
        for i in range(v1.shape[0]):
            for j in range(2):
                out_coors[i] += x[0 + 2*j]*r_matrix[j,i]*np.sin(phi_matrix[j,i] + x[1 + 2*j])    
        return np.concatenate((out_coors,[x[0]**2 + x[2]**2 - 1]))

    coeff = fsolve(_realization_equations, [1, 0, 1, 0, 0], xtol = 1e-10, maxfev = 5 * 300)
    u1 = coeff[0]*np.e**(1j*coeff[1]) * v1 + coeff[2]*np.e**(1j*coeff[3]) * v2
    u2 = make_vectors_real( - coeff[2] * np.e**(-1j*coeff[3]) * v1 + coeff[0]*np.e**(-1j*coeff[1])*v2 )
    u1 = u1 / np.linalg.norm( u1 )
    u2 = u2 / np.linalg.norm( u2 )
    return u1, u2


def unitary_to_orthogonal(unitary):
    """
    Converts a unitary matrix into an orthogonal matrix.
    This is done by creating real linear combinations of pairs of its rows.

    Args:
        unitary (np.ndarray): The unitary matrix.

    Returns:
        np.ndarray: The orthogonal matrix.
    """
    orthogonal_matrix = np.zeros_like(unitary)
    for i in [0,2]:
        v1 = unitary[0 + i]
        v2 = unitary[1 + i]
        u1, u2 = real_linear_combination(v1, v2)
        orthogonal_matrix[i] = u1
        orthogonal_matrix[i+1] = u2
    return orthogonal_matrix


def get_orthogonal_basis(complex_symmetric_matrix):
    """
    Obtains an orthogonal basis from a complex symmetric matrix.
    The method depends on the degeneracy class of the eigenvalues of the matrix.

    Args:
        complex_symmetric_matrix (np.ndarray): The complex symmetric matrix.

    Returns:
        np.ndarray: The orthogonal basis.
    """
    eigen_values, eigen_vectors = get_ordered_eig(complex_symmetric_matrix)
    
    if degeneracy_class(eigen_values) == (1,1,1,1): #four different eigenvalues
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
    """
    Breaks a 4x4 rotation tensor into two 2x2 rotation matrices.
    This is done by splitting the tensor into four 2x2 blocks and finding the non-zero block.

    Args:
        one_tensor_two (np.ndarray): The 4x4 rotation tensor.

    Returns:
        tuple: A tuple of two 2x2 rotation matrices.
    """
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

    return first_part_matrix, second_part_matrix
