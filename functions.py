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
# from universal_two_qubits_decomposition import *
# import universal_two_qubits_decomposition

from pyquil import get_qc, Program
from pyquil.api import get_qc, BenchmarkConnection
from forest.benchmarking.randomized_benchmarking import generate_rb_sequence
from pyquil.quil import *
from pyquil.gates import RX, RZ, CZ



lambda_unitary = np.array([ [1, 1j , 0 , 0],[0, 0, 1j, 1],[0, 0, 1j, -1],[1, -1j, 0, 0] ]) / np.sqrt(2)


def find_phi_theta_omega(single_rot):
    cos_theta_2 = np.round(abs(single_rot[0,0]), decimals=3)
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


def get_matrix_of_single_member_two_design_two_qubits():
    bm = BenchmarkConnection()
    sequences = generate_rb_sequence(bm, qubits=[0,1], depth=2)
    prog = sequences[:-1][0]
    mat = program_unitary(prog, n_qubits=2)
    return mat

def two_design_two_qubits_packs_generator(qmachine, target_qubits, num_layer:int):
    list_gates = []
    total_mat = np.eye(4,4)
    for index in range(num_layer):
        mat = get_matrix_of_single_member_two_design_two_qubits()
        total_mat = np.matmul(mat, total_mat)
        draft_circuit = get_corresponding_universal_circuit(mat, target_qubits)
        # list_gates.extend( qmachine.compiler.quil_to_native_quil(draft_circuit) )
        list_gates.extend( draft_circuit )
    
    inverse_mat = np.linalg.inv(total_mat)
    inverse_circuit = get_corresponding_universal_circuit(inverse_mat, target_qubits)
    list_gates = [ ins for ins in list_gates if isinstance(ins, Gate)]
    list_gates.extend( inverse_circuit )
        
    return list_gates


def get_corresponding_universal_circuit(u_matrix, target_qubits):
    u_matrix = strip_global_factor(u_matrix)
    print(u_matrix)
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
    
    assert np.all( np.isclose( l_matrix.dot(l_matrix.T), np.eye(4,4), atol = 1e-04) )
    assert np.all( np.isclose( k_matrix.dot(k_matrix.T), np.eye(4,4), atol = 1e-04) )
    
    assert np.isclose( np.linalg.det(k_matrix), 1, atol = 1e-04)
    assert np.isclose( np.linalg.det(l_matrix), 1, atol = 1e-04)
        
    a_tensor_b = matrix_out_magic_basis( np.matmul( v_magic_matrix.conjugate().transpose(),
                                                    np.matmul(k_matrix.transpose(), np.matmul(l_matrix, u_magic_matrix))) )
    a,b = break_rotation_tensor_into_two(a_tensor_b)
    prog_a, prog_b = get_program_of_single_unitary(a, target_qubit = target_qubits[0]), get_program_of_single_unitary(b, target_qubit = target_qubits[1])
    
    c_tensor_d = matrix_out_magic_basis( np.matmul(l_matrix.transpose(), k_matrix) )
    c,d = break_rotation_tensor_into_two(c_tensor_d)
    prog_c, prog_d = get_program_of_single_unitary(c, target_qubit = target_qubits[0]), get_program_of_single_unitary(d, target_qubit = target_qubits[1])
    
    prog = Program(prog_a, prog_b, v_circuit, prog_c, prog_d)
    return  prog

def native_universal_two_qubits_packs_generator(qmachine, target_qubits:list, num_layer:int):
    list_gates = []
    for index in range(num_layer):
        draft_circuit = give_random_two_qubit_circuit(target_qubits)
        list_gates.extend( draft_circuit )
        # list_gates.extend( qmachine.compiler.quil_to_native_quil(draft_circuit) )
    list_gates = [ ins for ins in list_gates if isinstance(ins, Gate)]
    list_gates.extend( get_inverse_circuit(qmachine, list_gates) )
    return list_gates


def native_rigetti_single_qubit_packs_generator(qmachine, target_qubit, num_layer:int):
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
    
    for index in range(0,num_layer):
        omega, phi = np.random.uniform(0, 2*np.pi, size = 2)
        theta = np.random.choice(angles, p = np.sin(angles) / np.sum( np.sin(angles) ))
        
        # draft_circuit = Program( [RZ(phi, qubit = target_qubit),
        #                           RX(np.pi/2, qubit = target_qubit),
        #                           RZ(theta, qubit = target_qubit),
        #                           RX(-np.pi/2, qubit = target_qubit),
        #                           RZ(omega, qubit = target_qubit)])
        
        list_gates.extend( arbitary_single_qubit_circuit(omega, theta, phi, target_qubit) )
    
    list_gates.extend( get_inverse_circuit(qmachine, list_gates) )
    return list_gates

def two_design_single_qubit_packs_generator(qmachine, target_qubit, num_layer:int):

    
    bm = BenchmarkConnection()
    
    sequences = generate_rb_sequence(bm, qubits=target_qubit, depth=num_layer)
    for prog in sequences:
        diff_length =  5 - len(prog)
        if diff_length == 0:
            pass
        else:
            for num in range(diff_length):
                prog += I(target_qubit[0])
    
    gates_list = []
    for prog in sequences:
        gates_list.extend(prog.instructions)
    
    return gates_list

def two_design_single_qubit_packs_generator_non_uniform(qmachine, target_qubit, num_layer:int):
    bm = BenchmarkConnection()
    
    sequences = generate_rb_sequence(bm, qubits=target_qubit, depth=num_layer)
    gates_list = []
    
    for prog in sequences:
        gates_list.extend(prog.instructions)
    
    return gates_list


bench_protocol_func_dict = {'native_conditional_single_qubit':native_rigetti_single_qubit_packs_generator,
                           'native_conditional_conditional_two_qubits':native_universal_two_qubits_packs_generator,
                           'standard_rb_single_qubit':two_design_single_qubit_packs_generator,
                           'standard_rb_two_qubits':two_design_two_qubits_packs_generator}



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
    elif gate.name == 'XY':
        angle = gate.params[0]
        return XY(angle = - angle,  q1= gate.qubits[0].index, q2= gate.qubits[1].index)
    elif gate.name == 'S':
        return PHASE(- np.pi/2, gate.qubits[0].index)
    elif gate.name == 'RX':
        angle = gate.params[0]
        return RX( - angle, qubit = gate.qubits[0].index)
    elif gate.name == 'RY':
        angle = gate.params[0]
        return RY( - angle, qubit = gate.qubits[0].index)
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


def g_gate(control, target):
    return Program( RZ(-3*np.pi/2, qubit = control), RZ(-3*np.pi/2, qubit = target),
                    CZ(control = target, target=control),
                    RZ(-np.pi, qubit = control), RZ(-np.pi, qubit = target) )


def arbitary_single_qubit_circuit(phi, theta, omega, qubit):
    draft_circuit = Program( [RZ(phi, qubit = qubit),
                              RX(np.pi/2, qubit = qubit),
                              RZ(theta, qubit = qubit),
                              RX(-np.pi/2, qubit = qubit),
                              RZ(omega, qubit = qubit)])
    return draft_circuit


def give_random_single_qubit_gate(qubit):
    phi, omega = np.random.uniform(0,2*np.pi, size = 2)
    
    theta_range = np.linspace(0,np.pi)
    p_theta = np.sin(theta_range) / np.sum(np.sin(theta_range))
    theta = np.random.choice(theta_range, p = p_theta)
    return arbitary_single_qubit_circuit(phi, theta, omega, qubit = qubit)

def normalized_abs_angle_dist(angle_range):
    dist = np.pi - np.abs( np.pi - angle_range )
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

def give_v_circuit(alpha, beta, delta, qubits = [0,1]):
    prog = Program(RZ(np.pi, qubits[0]), RX(np.pi/2, qubits[0]), RZ(np.pi/2, qubits[0]), RX(- np.pi/2, qubits[0]),
                    CZ(control=qubits[1], target=qubits[0]),
                    RZ(np.pi, qubits[0]), RX(np.pi/2, qubits[0]), RZ(np.pi/2, qubits[0]), RX(- np.pi/2, qubits[0]), RZ(delta, qubits[0]),
                    RX(np.pi/2, qubits[1]), RZ(np.pi/2 + beta, qubits[1]), RX(np.pi/2, qubits[1]),
                    CZ(control=qubits[0], target=qubits[1]))
    prog += Program(RZ(np.pi, qubits[1]), RX(np.pi/2, qubits[1]), RZ(np.pi/2 + alpha, qubits[1]), RX(-np.pi/2, qubits[1]),
                    RZ(np.pi, qubits[0]), RX(np.pi/2, qubits[0]), RZ(np.pi/2, qubits[0]), RX(- np.pi/2, qubits[0]),
                    CZ(control=qubits[1], target=qubits[0]),
                    RZ(np.pi, qubits[0]), RX(np.pi/2, qubits[0]), RZ(np.pi/2, qubits[0]), RX(- np.pi/2, qubits[0]))
    return prog

def give_random_two_qubit_circuit(qubits):
    a,b,c,d = [give_random_single_qubit_gate(qubit=qubit) for _ in range(2) for qubit in qubits]
    phi_one, phi_two, phi_three, phi_four = generate_haar_random_eigenvalues_two_qubits()
    
    alpha, beta, delta = [(phi_one + phi_two)/2, (phi_one + phi_three)/2, (phi_two + phi_three)/2]
    
    prog = Program(a, b )
    prog += give_v_circuit(alpha, beta, delta, qubits = qubits)
    prog += Program(c, d )
    return prog


def extrapolate_decay_func(layers_arr, avg_fdlty_arr):
    try:
        popt, pcov = curve_fit(decay_func, layers_arr, avg_fdlty_arr, bounds=([0,0,0], [1., 1., 1.]))
    except:
        popt, pcov = curve_fit(decay_func, layers_arr, avg_fdlty_arr, bounds=([0,0,0], [1., 1., 1.]))
    return popt, pcov

def plot_decay(layers_arr, avg_fdlty_arr, err_fdlty_arr, label:str, *args, **kwargs):
    
    fmt = kwargs.get('fmt','o')
    
    if 'axes' in kwargs:
        axes = kwargs.get('axes')
    else:
        fig = plt.figure()
        axes = fig.add_subplot()

    popt, pcov = extrapolate_decay_func(layers_arr, avg_fdlty_arr)
    
    # axes.errorbar(layers_arr, avg_fdlty_arr, yerr = err_fdlty_arr, fmt = 'o', color = 'k')
    err = axes.errorbar(layers_arr, avg_fdlty_arr, yerr = err_fdlty_arr, fmt = fmt, capsize = 4, capthick = 2)
    err[-1][0].set_linestyle('--')
    err[-1][0].set_alpha(0.5)
    between_layers = np.arange(layers_arr.min(),layers_arr.max()+1,1).astype('int')
    axes.plot(between_layers, decay_func(between_layers, *popt), color = err[0].get_color(),
              label =  label + ':' + r'${1}*{0}^m+{2}$'.format(*np.round(popt,4)))


    plt.xlabel('Depth', fontsize=18)
    plt.ylabel('Average of Fidelity', fontsize=16)

    axes.legend()
    

def used_qubits_index(gates_sequence):
    qubits = np.array([np.array(gate.qubits) for gate in gates_sequence])
    qubits = np.array([ ele.index for sub_arr in qubits for ele in sub_arr]) #some gates might have multiple indices
    qubits_indices = np.unique(qubits)
    qubits_indices.sort()
    return [ int(x) for x in qubits_indices ]

def convert_measured_to_response_matrix(measured_outcome):
    return 1 - np.bool_(np.sum(measured_outcome, axis = 1)) # 1 if it is equal to n_zero state

def run_bench_experiment(qmachine, program, number_of_shots):
    
    program = program.wrap_in_numshots_loop(number_of_shots)
    
    #Run the program
    executable = qmachine.compile(program)
    result = qmachine.run(executable)
    measured_outcome = result.readout_data.get('ro')
    return measured_outcome

def get_inverse_circuit(qmachine, gates_sequence):
    """
    :params gates_sequence: iterable sequence of circuit gates.
    :return: numpy array of gates constructing inverse circuit of the input 
    """
    target_qubits = used_qubits_index(gates_sequence)
    n_qubits = len(target_qubits)
    
    prog = Program()
    for gate in reversed(gates_sequence):
        prog += daggered_gate(gate)
    prog_daggered_native = qmachine.compiler.quil_to_native_quil(prog)
    instructions = prog_daggered_native.instructions
    inverting_gates_list = [ ins for ins in instructions if isinstance(ins, Gate)]
    return np.array(inverting_gates_list)

def save_experiment(experiment, protocol_name, target_qubits, layer_num, num_of_sequences):
    path = os.path.join( os.getcwd(), 'experiments_warehouse', protocol_name, str(target_qubits))
    try:
        os.makedirs( path )
    except:
        pass
    
    file_path = os.path.join( path, 'L{}_K{}.pickle'.format(layer_num, num_of_sequences) )
    
    with open(file_path, "wb") as output_file:
        cPickle.dump(experiment, output_file)
    return

def catch_experiments(qmachine, target_qubits:list, protocol_name:str, layers_num:int, exp_num:int):
    
    file_path = os.path.join( os.getcwd(), 'experiments_warehouse', protocol_name,
                        str(target_qubits), 'L{}_K{}.pickle'.format(layers_num, exp_num) )
    
    if os.path.isfile(file_path): #if such exp exists
        with open(file_path, "rb") as input_file:
            exps = cPickle.load(input_file)
    else: #if it does not exists
        circuit_gen_func = bench_protocol_func_dict[protocol_name]
        exps = generate_experiments(qmachine, target_qubits, circuit_gen_func, layers_num, exp_num)
        save_experiment(exps, protocol_name, target_qubits, layers_num, exp_num)
    return exps

def generate_experiments(qmachine, target_qubits:list, circuit_gen_func, layers_num:int, exp_num:int):
    n_qubits = len(target_qubits)
    exp_list = []
    for i in tqdm(range(exp_num), desc = 'exp. generation'):
        exp_list.append(circuit_gen_func(qmachine, target_qubits, layers_num))
    
    return exp_list

def find_machine_response(qmachine, rb_experiments, number_of_shots):
    """
    It samples and record the accept or reject of the machine with given gate sequences
    :return: response_matrix including accepts and rejects in columns
    """
    target_qubits = used_qubits_index(rb_experiments[0])
    n_qubits = len(target_qubits)
    sequ_num = len(rb_experiments)
    response_matrix = np.zeros((sequ_num, number_of_shots))

    for i_sequ, sequ in enumerate(tqdm(rb_experiments, desc = 'Examing the seq.')):
        prog = Program() #All qubits begin with |0> state
        for gate in sequ:
            prog += gate
        
        #Do not let the quilc to alter the gates by optimization
        prog = Program('PRAGMA PRESERVE_BLOCK') + prog
        prog += Program('PRAGMA END_PRESERVE_BLOCK')
        
        #Measurments
        ro = prog.declare('ro', 'BIT', n_qubits)
        for ind, qubit_ind in enumerate(target_qubits):
            prog += MEASURE(qubit_ind, ro[ind])
            
        response = convert_measured_to_response_matrix( run_bench_experiment(qmachine, prog, number_of_shots) )
        response_matrix[i_sequ,:] = np.copy(response)
    return response_matrix


