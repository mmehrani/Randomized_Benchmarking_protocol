#!/usr/bin/env python
# coding: utf-8

# # Benchmark with conditional formating
# All we need in the protocal is the Haar-random unitaries. A random unitary can be represeted by rotation matrices like this:
# 
# In this project we benchmark with those conditional probabilities

# In[93]:


from pyquil import get_qc, Program
from pyquil.gates import *
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare
from pyquil.simulation.tools import lifted_gate, program_unitary
from pyquil.quil import *


# In[94]:


import numpy as np
import math
from math import pi
import random
import copy
from tqdm import tqdm_notebook as tqdm


# In[95]:


from functions import *


# In[96]:


if __name__ == "__main__":
    num_qubits = 2

#     First step choose m and the K_m sequences of 
    m = 2
    k_m = 50 #n. of diff sequences
    n_m = 100  #n. of samples from a certain sequence


# In[ ]:





# In[97]:


def universal_two_qubits_packs_generator(qmachine, num_layer):
    list_gates = []
    for index in range(num_layer):
        draft_circuit = give_random_two_quibt_circuit([0,1])
        list_gates.extend( qmachine.compiler.quil_to_native_quil(draft_circuit) )
    list_gates = [ ins for ins in list_gates if isinstance(ins, Gate)]
    return list_gates


# In[98]:


def machine_response_rb_universal_two_qubits_conditional(qmachine, num_qubits, m, k_m, n_m):
    """
    It samples and record the accept or reject of the machine with native gates chosen with conditions for rigetti.
    ::return response_matrix including accepts and rejects in columns
    """
    response_matrix = np.zeros((k_m,n_m))
    
    for i_sequ in tqdm(range(k_m), desc = 'Sequences'):
        gate_list = universal_two_qubits_packs_generator(qmachine, m)
        prog = Program() #All qubits begin with |0> state
        
        for gate in gate_list:
            prog += gate
        
        #Come back to our initial state
        for gate in reversed(gate_list):
#             prog += copy.deepcopy(gate).dagger() #dagger has replacing operations
            gate_daggered = copy.deepcopy(gate)
            gate_daggered.params[0] *= -1 #make daggered rotation 
            prog += gate_daggered
            
        #Do not let the quilc to alter the gates by optimization
        prog = Program('PRAGMA PRESERVE_BLOCK') + prog
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


# In[ ]:


if __name__ == "__main__":
#     qc = get_qc( str(num_qubits) + 'q-qvm')  # You can make any 'nq-qvm'
    qc = get_qc("9q-square-noisy-qvm")
    response = machine_response_rb_universal_two_qubits_conditional(qc, num_qubits, m, k_m, n_m)


# In[ ]:


if __name__ == "__main__":
    get_ipython().system('jupyter nbconvert RB_with_Rigetti_native_gates_conditional_probability_two_qubits.ipynb --to python')


# In[ ]:





# In[ ]:




