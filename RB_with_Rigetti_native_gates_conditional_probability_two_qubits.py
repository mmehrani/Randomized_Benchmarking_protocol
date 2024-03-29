#!/usr/bin/env python
# coding: utf-8

# # Benchmark with conditional formating
# All we need in the protocal is the Haar-random unitaries. A random unitary can be represeted by rotation matrices like this:
# 
# In this project we benchmark with those conditional probabilities

# In[13]:


from pyquil import get_qc, Program
from pyquil.gates import *
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare
from pyquil.simulation.tools import lifted_gate, program_unitary, permutation_arbitrary
from pyquil.quil import *


# In[14]:


import numpy as np
import math
from math import pi
import random
import copy
from tqdm import tqdm_notebook as tqdm


# In[15]:


from functions import *


# In[17]:


if __name__ == "__main__":
    target_qubits = [0,1]

#     First step choose m and the K_m sequences of 
    m = 1
    k_m = 1 #n. of diff sequences
    n_m = 1  #n. of samples from a certain sequence


# In[18]:


def universal_two_qubits_packs_generator(qmachine, target_qubits:list, num_layer:int):
    list_gates = []
    for index in range(num_layer):
        draft_circuit = give_random_two_qubit_circuit(target_qubits)
        list_gates.extend( qmachine.compiler.quil_to_native_quil(draft_circuit) )
    list_gates = [ ins for ins in list_gates if isinstance(ins, Gate)]
    return list_gates


# In[19]:


def machine_response_rb_universal_two_qubits_conditional(qmachine, target_qubits:list, m:int, k_m, n_m):
    """
    It samples and record the accept or reject of the machine with native gates chosen with conditions for rigetti.
    ::return response_matrix including accepts and rejects in columns
    """
    num_qubits = len(target_qubits)
    response_matrix = np.zeros((k_m,n_m))
    
    for i_sequ in tqdm(range(k_m), desc = 'Sequences'):
        gate_list = universal_two_qubits_packs_generator(qmachine, target_qubits, m)
        prog = Program() #All qubits begin with |0> state
        
        for gate in gate_list:
            prog += gate
        
        #Come back to our initial state
        u_inverse_definition = DefGate('U_inverse', np.linalg.inv(program_unitary(prog, n_qubits=2)))
        U_inverse = u_inverse_definition.get_constructor()
        prog += qmachine.compiler.quil_to_native_quil(Program(u_inverse_definition, U_inverse(*target_qubits)))
        
        #Do not let the quilc to alter the gates by optimization
        prog = Program('PRAGMA PRESERVE_BLOCK') + prog
        prog += Program('PRAGMA END_PRESERVE_BLOCK')
        
        #Measurments
        ro = prog.declare('ro', 'BIT', num_qubits)
        for q in range(num_qubits):
            prog += MEASURE(target_qubits[q], ro[q])
        prog = prog.wrap_in_numshots_loop(n_m)

        #Run the program
        executable = qmachine.compile(prog)
        result = qmachine.run(executable)
        measured_outcome = result.readout_data.get('ro')

        response_matrix[i_sequ,:] = 1 - np.bool_(np.sum(measured_outcome, axis = 1)) # 1 if it is equal to n_zero state
    return prog, response_matrix


# In[27]:


if __name__ == "__main__":
    qc = get_qc( str(2) + 'q-qvm')  # You can make any 'nq-qvm'
#     qc = get_qc("9q-square-noisy-qvm")
    prog, response = machine_response_rb_universal_two_qubits_conditional(qc, [0,1], m, k_m, n_m)


# In[8]:


if __name__ == "__main__":
    get_ipython().system('jupyter nbconvert RB_with_Rigetti_native_gates_conditional_probability_two_qubits.ipynb --to python')


# In[29]:


instructions = [ins for ins in prog.instructions if isinstance(ins, Gate)]
u = program_unitary( Program(instructions), n_qubits=2)
u_inv = np.linalg.inv(u)
u


# In[23]:


np.dot(u, u_inv)


# In[25]:


print(prog)


# In[ ]:




