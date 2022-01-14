#!/usr/bin/env python
# coding: utf-8

# # Standard Randomized Benchmarking

# In[1]:


from pyquil import get_qc, Program
from pyquil.gates import *
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare
from pyquil.simulation.tools import lifted_gate, program_unitary
from pyquil.quil import *


# In[2]:


import numpy as np
import random
from functions import averageOfFidelity


# In[ ]:





# In[3]:


if __name__ == "__main__":
    num_qubits = 2

    #First step choose m and the K_m sequences of Clifford group
    m = 5
    k_m = 10 #n. of diff sequences
    n_m = 10  #n. of samples from a certain sequence


# In[4]:


def generate_clifford_group(num_qubits):
    #The glossary of Clifford gates
    clifford_glossary = []
    clifford_glossary.extend([CNOT(i,j) for i in range(num_qubits) for j in range(num_qubits)])
    for i in range(num_qubits): clifford_glossary.remove(CNOT(i,i))
    clifford_glossary.extend([H(i) for i in range(num_qubits)])
    clifford_glossary.extend([S(i) for i in range(num_qubits)])
    return clifford_glossary


# In[5]:


def machine_response_standard_bench(qmachine, num_qubits, m, k_m, n_m):
    """
    It samples and record the accept or reject of the machine.
    ::return response_matrix including accepts and rejects in columns
    """
    response_matrix = np.zeros((k_m,n_m))
    clifford_glossary = generate_clifford_group(num_qubits)
    
    for i_sequ in range(k_m):
        c_jm =  random.choices(clifford_glossary, k = m)
        #compute the unitary of circuit U
        prog = Program() #All qubits begin with |0> state
        for gate in c_jm:
            prog += gate

        c_jm_unitary = program_unitary(prog, n_qubits= num_qubits)

        #report the reversed unitary operator of the total transforamtions 
        c_jm_unitary_r = np.linalg.inv( c_jm_unitary )
        c_jm_unitary_r_definition = DefGate("U_r", c_jm_unitary_r)
        U_r = c_jm_unitary_r_definition.get_constructor() # Get the gate constructor

        n_tuple = tuple(range(num_qubits))
        prog += Program( c_jm_unitary_r_definition, U_r(*n_tuple) )
        
        #Do not let the quilc to alter the gates by optimization
        prog = Program('PRAGMA INITIAL_REWIRING "NAIVE"') + Program('PRAGMA PRESERVE_BLOCK') + prog
        prog += Program('PRAGMA END_PRESERVE_BLOCK')
        
        #Measurments
        ro = prog.declare('ro', 'BIT', num_qubits)
        for q in range(num_qubits):
            prog += MEASURE(q, ro[q])

        prog = prog.wrap_in_numshots_loop(n_m)

        #Run the program
#         qc = get_qc( machine_type )
        executable = qmachine.compile(prog)
        result = qmachine.run(executable)
        measured_outcome = result.readout_data.get('ro')

        response_matrix[i_sequ,:] = 1 - np.bool_(np.sum(measured_outcome, axis = 1)) # 1 if it is equal to n_zero state
    
    return response_matrix


# In[ ]:





# In[6]:


# response_matrix = machine_response_standard_bench(machine_type, num_qubits, m, k_m, n_m)


# In[7]:


# averageOfFidelity(response_matrix)


# In[8]:


if __name__ == "__main__":
    get_ipython().system('ipython nbconvert --to python RB_standard_verification.ipynb')


# In[ ]:




