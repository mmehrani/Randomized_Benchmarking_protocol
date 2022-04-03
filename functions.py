# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:09:28 2021

@author: mohsen
"""


import numpy as np
from pyquil import get_qc, Program

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

# def averageOfFidelity(Data_Array):
    
#     k_m, n_m = Data_Array.shape
#     Pacc_Array = np.array( [sum(i) for i in Data_Array] ) / n_m
    
#     vfunc = np.vectorize(calculate_lower_bound)
#     Lower_Bound_F = vfunc(Pacc_Array)
#     AvergeOf_F = sum(Lower_Bound_F) / k_m
#     return AvergeOf_F

def averageOfFidelity(Data_Array):
    
    k_m, n_m = Data_Array.shape
    Pacc_Array = np.average(Data_Array, axis = 1)
    Pacc = np.average(Pacc_Array)
    # AvergeOf_F_lower_bound = np.average( [1 + np.e * np.log(x) if x!=0 else -100 for x in Pacc_Array] )
    return Pacc

# def averageOfFidelity(Data_Array):
    
#     k_m, n_m = Data_Array.shape
#     Pacc_Array = np.average(Data_Array, axis = 1)
    
#     AvergeOf_F_lower_bound = np.average( [1 + np.e * np.log(x) if x!=0 else -100 for x in Pacc_Array] )
#     return AvergeOf_F_lower_bound

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
    
