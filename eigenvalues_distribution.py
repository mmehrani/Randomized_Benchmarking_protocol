# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
from math import pi, factorial
import os




# def prob_density_one_two(phi_one, phi_two):
#     phi_two + np.sin(phi_one - phi_two) - np.sin(phi_one)
#     return

# def generate_phi_one():
#     return np.random.uniform(0,2*np.pi)

# def generate_phi_two(phi_one):
#     x = np.random.uniform(0,2*np.pi)
#     func = lambda phi_two: phi_two + np.sin(phi_one - phi_two) - np.sin(phi_one)  - x
#     answer = optimize.root(func, [0])
#     phi_two = answer['x'][0]
#     return phi_two

# def generate_phi_three(phi_one, phi_two):
#     normalization_const = np.pi/8 * (4*np.pi +  2*np.pi*np.cos(phi_one - phi_two) - 
#                                      4*np.sin(phi_one) - 4*np.sin(phi_two) + 
#                                      np.sin(phi_one + phi_two))
#     x_max = 8*(np.pi/2 + np.pi/4 * np.cos(phi_one - phi_two)) / \
#         ( np.pi*(4*np.pi + 2*np.pi*np.cos(phi_one - phi_two) - \
#                  4*np.sin(phi_one) - 4*np.sin(phi_two) +  np.sin(phi_one + phi_two)) )
#     x = np.random.uniform(0, x_max)
#     func = lambda phi_three: ( - np.sin(phi_one + phi_two - 2*phi_three)/16 + \
#         np.sin(phi_one - phi_three)/4 + np.sin(phi_two - phi_three)/4 + phi_three / 4 + \
#             np.cos(phi_one - phi_two)*phi_three / 8 - \
#                 (np.sin(phi_one)/4 + np.sin(phi_two)/4 - np.sin(phi_one + phi_two)/16) )/normalization_const - x
#     answer = optimize.root(func, [0])
#     phi_three = answer['x'][0]
#     return phi_three

# def generate_phi_four(phi_one, phi_two, phi_three):
#     normalization_const = 1/48 * np.pi * (12 * np.pi + 6 * np.pi * np.cos(phi_one - phi_two) + \
#                                           6 * np.pi * np.cos(phi_one - phi_three) + \
#         6 * np.pi * np.cos(phi_two - phi_three) - 12 * np.sin(phi_one) - 12 * np.sin(phi_two) + \
#             3 * np.sin(phi_one + phi_two) + 3 * np.sin(phi_one - phi_two - phi_three) - \
#                 3 * np.sin(phi_one + phi_two - phi_three) - 12 * np.sin(phi_three) + \
#                     3 * np.sin(phi_one + phi_three) - 3 * np.sin(phi_one - phi_two + phi_three) + \
#                         3 * np.sin(phi_two + phi_three) -  np.sin(phi_one + phi_two + phi_three) )
    
        
#     x_max = (6 * (2 + np.cos(phi_one - phi_two) + np.cos(phi_one - phi_three) + np.cos(phi_two - phi_three))) / \
#         (12 * np.pi + 6 * np.pi * np.cos(phi_one - phi_two) + 6 * np.pi * np.cos(phi_one - phi_three) +\
#          6 * np.pi * np.cos(phi_two - phi_three) - 12 * np.sin(phi_one) - 12 * np.sin(phi_two) +\
#              3 * np.sin(phi_one + phi_two) + 3 * np.sin(phi_one - phi_two - phi_three) - \
#                  3 * np.sin(phi_one + phi_two - phi_three) - 12 * np.sin(phi_three) + \
#                      3 * np.sin(phi_one + phi_three) - 3 * np.sin(phi_one - phi_two + phi_three) + \
#                          3 * np.sin(phi_two + phi_three) - np.sin(phi_one + phi_two + phi_three))
        
#     x = np.random.uniform(0, x_max)
#     func = lambda phi_four: ( 1/96 * np.sin(phi_one + phi_two + phi_three - 3 * phi_four) \
#         - 1/32 * np.sin(phi_one + phi_two - 2*phi_four) \
#             - 1/32 * np.sin(phi_one + phi_three - 2 * phi_four) \
#                 - 1/32 * np.sin(phi_two + phi_three - 2 * phi_four) \
#                     + 1/8 * np.sin(phi_one - phi_four) + 1/8 * np.sin(phi_two - phi_four) \
#                         + 1/32 * np.sin(phi_one + phi_two - phi_three - phi_four) \
#                             + 1/8 * np.sin(phi_three - phi_four) + 1/32 * np.sin(phi_one - phi_two + phi_three - phi_four) \
#                                 - 1/32 * np.sin(phi_one - phi_two - phi_three + phi_four) + phi_four/8 + \
#                                     1/16 * np.cos(phi_one - phi_two) * phi_four \
#                                         + 1/16 * np.cos(phi_one - phi_three) * phi_four \
#                                             + 1/16 * np.cos(phi_two - phi_three) * phi_four \
#                                                 - (np.sin(phi_one)/8 + np.sin(phi_two)/8 - 1/32 * np.sin(phi_one + phi_two) \
#                                                    - 1/32 * np.sin(phi_one - phi_two - phi_three) \
#                                                        + 1/32 * np.sin(phi_one + phi_two - phi_three) + np.sin(phi_three)/8 \
#                                                            - 1/32 * np.sin(phi_one + phi_three) + 1/32 * np.sin(phi_one - phi_two + phi_three) \
#                                                              - 1/32 * np.sin(phi_two + phi_three) + 1/96 * np.sin(phi_one + phi_two + phi_three)) ) / normalization_const\
#                                                     - x
#     answer = optimize.root(func, [0])
#     phi_four = answer['x'][0]
#     return phi_four

# def generate_haar_random_eigenvalues_two_qubits():
#     phi_one = generate_phi_one()
#     phi_two = generate_phi_two(phi_one)
#     phi_three = generate_phi_three(phi_one, phi_two)
#     phi_four = generate_phi_four(phi_one, phi_two, phi_three)
#     return phi_one, phi_two, phi_three, phi_four

def haar_distribution(phis, n):
    phi_exps = np.exp(1j * phis)
    output = 1 / (factorial(n) * (2 * pi) ** n)
    for (i, phi_exp) in enumerate(phi_exps):
        for j in range(i + 1, len(phi_exps)):
            output *= np.abs(phi_exp - phi_exps[j]) ** 2
    return output

def rejection_sample_haar(n, haar_degree=4, heuristic_ratio=3):
    """Get `n` samples from a Haar distribution with degree `haar_degree`

    Parameters
    ----------
    n : int
        sample count
    haar_degree : int, optional
        Degree of the Haar distribution (i.e. number of angles), by default 4
    heuristic_ratio : int, optional
        Proposed entries per sample for each pass, by default 3

    Returns
    -------
    numpy.ndarray
        Matrix with `n` rows and `haar_degree` columns that contains the
        random samples
    """
    samples = np.empty((n, haar_degree))
    sample_count = 0
    while sample_count < n:
        rand_phis = 2 * pi * (np.random.random_sample((
            heuristic_ratio * (n - sample_count), haar_degree)) - 0.5)
        rand_nums = np.random.random_sample(
            heuristic_ratio * (n - sample_count))
        for (phis, rand_num) in zip(rand_phis, rand_nums):
            if rand_num < (factorial(haar_degree) * pi ** haar_degree
                           * haar_distribution(phis, haar_degree)):
                samples[sample_count] = phis
                sample_count += 1
                if sample_count >= n:
                    break
    return samples


# Specify the file path
file_path = 'haar_generated_phases_4_dim_1e4_samples.npy'

# Check if the file exists
if os.path.exists(file_path):
    all_phis = np.load('haar_generated_phases_4_dim_1e4_samples.npy')
else:
    all_phis = rejection_sample_haar(10000)
    np.save('haar_generated_phases_4_dim_1e4_samples.npy',all_phis)

def generate_haar_random_eigenvalues_two_qubits():
    i = np.random.randint(0, len(all_phis))
    haar_random_phis = all_phis[i]
    return haar_random_phis

if __name__ == '__main__':
    # phi_one = generate_phi_one()
    # plt.hist([generate_phi_two(generate_phi_one()) for _ in range(10000)], bins = 100)
    # plt.hist([generate_phi_two(np.pi) for _ in range(10000)])
    # plt.hist([generate_phi_three(0,0) for _ in range(10000)])
    # plt.hist([generate_phi_four(0,2*np.pi/3,4*np.pi/3) for _ in range(10000)])
    pass