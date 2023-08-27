# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:19:37 2023

@author: mohsen
"""
import numpy as np

# def damping_channel(damp_prob=.1):
#     """
#     Generate the Kraus operators corresponding to an amplitude damping
#     noise channel.

#     :params float damp_prob: The one-step damping probability.
#     :return: A list [k1, k2] of the Kraus operators that parametrize the map.
#     :rtype: list
#     """
#     damping_op = np.sqrt(damp_prob) * np.array([[0, 1],
#                                                 [0, 0]])

#     residual_kraus = np.diag([1, np.sqrt(1-damp_prob)])
#     return [residual_kraus, damping_op]

# # def damping_channel(damp_prob=.1):
# #     """
# #     Generate the Kraus operators corresponding to an amplitude damping
# #     noise channel.

# #     :params float damp_prob: The one-step damping probability.
# #     :return: A list [k1, k2] of the Kraus operators that parametrize the map.
# #     :rtype: list
# #     """
# #     damping_op = np.sqrt(damp_prob) * np.array([[0, 1],
# #                                                 [0, 0]])

# #     residual_kraus = np.diag([1, np.sqrt(1-damp_prob)])
    
# #     kraus_set_two_qubits = [np.kron(damping_op, damping_op), np.kron(damping_op, residual_kraus),
# #                             np.kron(residual_kraus, damping_op), np.kron(residual_kraus, residual_kraus)]
# #     return np.array(kraus_set_two_qubits)
def append_kraus_to_gate(kraus_ops, g):
    """
    Follow a gate `g` by a Kraus map described by `kraus_ops`.

    :param list kraus_ops: The Kraus operators.
    :param numpy.ndarray g: The unitary gate.
    :return: A list of transformed Kraus operators.
    """
    return [kj.dot(g) for kj in kraus_ops]


# def append_damping_to_gate(gate, damp_prob=.1):
#     """
#     Generate the Kraus operators corresponding to a given unitary
#     single qubit gate followed by an amplitude damping noise channel.

#     :params np.ndarray|list gate: The 2x2 unitary gate matrix.
#     :params float damp_prob: The one-step damping probability.
#     :return: A list [k1, k2] of the Kraus operators that parametrize the map.
#     :rtype: list
#     """
#     return append_kraus_to_gate(damping_channel(damp_prob), gate)


def dephasing_kraus_map(p=.1):
    """
    Generate the Kraus operators corresponding to a dephasing channel.

    :params float p: The one-step dephasing probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    return [np.sqrt(1-p)*np.eye(2), np.sqrt(p)*np.diag([1, -1])]

def tensor_kraus_maps(k1, k2):
    """
    Generate the Kraus map corresponding to the composition
    of two maps on different qubits.

    :param list k1: The Kraus operators for the first qubit.
    :param list k2: The Kraus operators for the second qubit.
    :return: A list of tensored Kraus operators.
    """
    return [np.kron(k1j, k2l) for k1j in k1 for k2l in k2]