# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:09:28 2021

@author: mohsen
"""


import numpy as np
from pyquil import get_qc, Program
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pyquil.quil import *
from pyquil.gates import *
# calculate the average fidelity for a given m

def calculate_lower_bound(p_jm):
    if p_jm == 1:
        return 1
    elif p_jm == 0:
        return 0
    else:
        R = int(1/np.log(1/p_jm)+0.5)
        if R < 1: R = 1
        return 1 - 1/( R * p_jm**R )



def averageOfFidelity(Data_Array):
    k_m, n_m = Data_Array.shape
    Pacc_Array = np.average(Data_Array, axis = 1)
    Pacc = np.average(Pacc_Array)
    return Pacc

def stdOfFidelity(Data_Array):
    k_m, n_m = Data_Array.shape
    Pacc_Array = np.average(Data_Array, axis = 1)
    Pacc_err = np.std(Pacc_Array)
    return Pacc_err


def decay_func(m,p,a0,b0):
    return a0*p**m+b0

def decay_param(p,qubit_num):
    d = 2**qubit_num
    r = (d-1)*(1-p)/d
    return r

def qvirtual_machine(given_program):
    n_qubits = given_program.get_qubits()
    qc = get_qc(  str(n_qubits) + 'q-qvm' )
    
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
    if gate.name in ['CZ','CNOT','H'] :
        return gate
    elif gate.name == 'S':
        return PHASE(- np.pi/2, gate.qubits[0].index)
    elif gate.name == 'RX':
        angle = gate.params[0]
        return RX( - angle, qubit = gate.qubits[0].index)
    elif gate.name == 'RZ':
        angle = gate.params[0]
        return RZ( - angle, qubit = gate.qubits[0].index)
    
    else :
        raise ValueError("This gate daggered is not yet considered!")


def convert_to_bloch_vector(rho):
    """Convert a density matrix to a Bloch vector."""
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # Used the mixed state simulator so we could have the density matrix for this part!
    ax = np.trace(np.dot(rho, X)).real
    ay = np.trace(np.dot(rho, Y)).real
    az = np.trace(np.dot(rho, Z)).real
    return [ax, ay, az]

def plot_bloch_sphere(bloch_vectors):
    """ Helper function to plot vectors on a sphere."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(30, 45)
    ax.dist = 7

    # Draw the axes (source: https://github.com/matplotlib/matplotlib/issues/13575)
    x, y, z = np.array([[-1.5,0,0], [0,-1.5,0], [0,0,-1.5]])
    u, v, w = np.array([[3,0,0], [0,3,0], [0,0,3]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color="black", linewidth=0.5)

    ax.text(0, 0, 1.7, r"|0⟩", color="black", fontsize=16)
    ax.text(0, 0, -1.9, r"|1⟩", color="black", fontsize=16)
    ax.text(1.9, 0, 0, r"|+⟩", color="black", fontsize=16)
    ax.text(-1.7, 0, 0, r"|–⟩", color="black", fontsize=16)
    ax.text(0, 1.7, 0, r"|i+⟩", color="black", fontsize=16)
    ax.text(0,-1.9, 0, r"|i–⟩", color="black", fontsize=16)

    ax.scatter(
        bloch_vectors[:,0], bloch_vectors[:,1], bloch_vectors[:, 2], c='#e29d9e', alpha=0.3
    )

G = Program( CPHASE01(-np.pi/2, control=0, target=1), CPHASE10(-np.pi/2, control=0, target=1) )

def arbitary_single_qubit_circuit(theta, phi, si, qubit):
    return Program( RZ(si, qubit = qubit), RY(phi, qubit = qubit), RZ(theta, qubit = qubit) )

def r_theta_phi_rotation(theta, phi, qubit):
    return arbitary_single_qubit_circuit( - phi/2, theta, phi/2, qubit)

def give_random_single_qubit_gate(qubit):
    theta, si = np.random.uniform(0,2*np.pi, size = 2)
    
    phi_range = np.linspace(0,np.pi)
    p_phi = np.sin(phi_range) / np.sum(np.sin(phi_range))
    phi = np.random.choice(phi_range, p = p_phi)
    return arbitary_single_qubit_circuit(theta, phi, si, qubit = qubit)

def normalized_abs_angle_dist(angle_range):
    dist = np.pi - np.abs( np.pi - angle_range )
    dist /= np.sum(dist)
    return dist

def give_v_circuit(alpha, beta, delta, qubits = [0,1]):
    prog = Program(G,  r_theta_phi_rotation(alpha, 0, qubit =qubits[0]),
                   r_theta_phi_rotation(3*np.pi/2,0, qubit =qubits[1]), G)
    prog += Program( r_theta_phi_rotation(beta, np.pi/2, qubit = qubits[0]), 
                    r_theta_phi_rotation(3*np.pi/2, delta, qubit = qubits[1]), G)
    return prog

def give_random_two_quibt_circuit(qubits):
    a,b,c,d = [give_random_single_qubit_gate(qubit=qubit) for _ in range(2) for qubit in qubits]
    
    angles_range = np.linspace(0,2*np.pi)
    alpha, beta, delta = np.random.choice(angles_range, p = normalized_abs_angle_dist(angles_range),
                                          size = 3)
    
    prog = Program(a, b )
    prog += give_v_circuit(alpha, beta, delta, qubits = [0,1])
    prog += Program(c, d )
    return prog

def extrapolate_decay_func(layers_arr, avg_fdlty_arr):
    try:
        popt, pcov = curve_fit(decay_func, layers_arr, avg_fdlty_arr)
    except:
        popt, pcov = curve_fit(decay_func, layers_arr, avg_fdlty_arr, bounds=([0,0,-10], [1., 10., 0.]))
    return popt, pcov

def plot_decay(layers_arr, avg_fdlty_arr, err_fdlty_arr, label:str, *args, **kwargs):
    fig = plt.figure()
    plt.close(fig)
    
    ax = fig.add_subplot()

    axes = kwargs.get('axes', ax)
    popt, pcov = extrapolate_decay_func(layers_arr, avg_fdlty_arr)
    
    axes.errorbar(layers_arr, avg_fdlty_arr, yerr = err_fdlty_arr, fmt = 'o', color = 'k')
    
    between_layers = np.arange(layers_arr.min(),layers_arr.max()+1,1).astype('int')
    axes.plot(between_layers, decay_func(between_layers, *popt),
              label =  label + ':' + r'${1}*{0}^m+{2}$'.format(*np.round(popt,2)))


    plt.xlabel('Depth', fontsize=18)
    plt.ylabel('Average of Fidelity', fontsize=16)
    # plt.title('RB' + title)

    plt.legend()
    # plt.close(fig)
    # fig.savefig('RB_' + title + '.png')