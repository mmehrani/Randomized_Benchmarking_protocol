#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyquil import get_qc, Program
from pyquil.gates import *
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare
from pyquil.simulation.tools import lifted_gate, program_unitary
from pyquil.quil import *


# In[2]:


import numpy as np
import math
from math import pi
import random


# In[3]:


if __name__ == "__main__":
    num_qubits = 2

#     First step choose m and the K_m sequences of 
    m = 5
    k_m = 10 #n. of diff sequences
    n_m = 10  #n. of samples from a certain sequence


# In[4]:


def native_reggeti_gate_generator(num_Qbit,num_gates):
    list_gates = []
    for i in range(0,num_gates):
        k = random.randint(1,3)
#         k = 1
        if k==1:
            s_1 = random.randint(0,num_Qbit-1)
            angle_1 = random.choice([-1,-1/2,+1/2,1])
            list_gates.append(RX(angle = angle_1*pi,qubit = s_1))
        
        if k==2:
            s_2 = random.randint(0,num_Qbit-1)
            angle_2 = (random.random())
            list_gates.append(RZ(angle = 2*pi*angle_2,qubit = s_2))
            
        if k==3:
            control_qubit,target_qubit = random.sample(range(0,num_Qbit),2)
            list_gates.append(CZ(control_qubit,target_qubit))
    return list_gates


# In[5]:


def iden_generator(num_qubit,num_gates):
    list_gates = []
    for i in range(0,num_gates):
        #s = random.randint(0,num_qubit-1)
        #list_gates.append(RZ((1/2)*pi,s))
        control_qubit,target_qubit = random.sample(range(0,num_qubit),2)
        list_gates.append(CZ(control_qubit,target_qubit))
    return list_gates


# In[6]:


def machine_response_srb_native_gate(qmachine, num_qubits, m, k_m, n_m):
    """
    It samples and record the accept or reject of the machine with standard native gates for rigetti.
    ::return response_matrix including accepts and rejects in columns
    """
    response_matrix = np.zeros((k_m,n_m))
    
    for i_sequ in range(k_m):
        gate_list = native_reggeti_gate_generator(num_qubits, m)
        prog = Program() #All qubits begin with |0> state
        
        for gate in gate_list:
            prog += gate
        
        #compute the unitary of circuit U
#         equivalent_unitary = program_unitary(prog, n_qubits= num_qubits)
        
#         if np.all( np.matmul(equivalent_unitary,equivalent_unitary.conj().T) != np.eye(2**num_qubits) ): print('spotted')
#         #report the reversed unitary operator of the total transforamtions 
#         equivalent_unitary_inv = equivalent_unitary.conj().T
#         equivalent_unitary_inv_def = DefGate("U_r", equivalent_unitary_inv)
#         U_r = equivalent_unitary_inv_def.get_constructor() # Get the gate constructor
        
#         n_tuple = tuple(range(num_qubits))
#         prog += Program( equivalent_unitary_inv_def, U_r(*n_tuple) )
        
        for gate in reversed(gate_list):
            prog += gate
        #Do not let the quilc to alter the gates by optimization
        prog = Program('PRAGMA INITIAL_REWIRING "NAIVE"') + Program('PRAGMA PRESERVE_BLOCK') + prog
        prog += Program('PRAGMA END_PRESERVE_BLOCK')
        
        #Measurments
        ro = prog.declare('ro', 'BIT', num_qubits)
        for q in range(num_qubits):
            prog += MEASURE(q, ro[q])
        prog = prog.wrap_in_numshots_loop(n_m)

        #Run the program
        executable = qmachine.compile(prog)
        result = qmachine.run(executable)
        measured_outcome = result.readout_data.get('ro')

        response_matrix[i_sequ,:] = 1 - np.bool_(np.sum(measured_outcome, axis = 1)) # 1 if it is equal to n_zero state
    return response_matrix


# In[7]:


if __name__ == "__main__":
    get_ipython().system('ipython nbconvert --to python RB_with_Rigetti_native_gates.ipynb')


# In[ ]:





# In[8]:


# qc = get_qc( str(num_qubits) + 'q-qvm')  # You can make any 'nq-qvm'
# machine_response_srb_native_gate(qc,num_qubits, m, k_m, n_m)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




