#!/usr/bin/env python
# coding: utf-8

# # Benchmark with conditional formating
# all we need in the protocal is the random unitaries. rigetti does the task by gate decomposition. A random special unitary can be represeted by rotation matrices like this:
# $$
# U = R_Z(\omega) R_Y(\theta) R_Z(\phi)\\
# $$
# Which in terms of the shortest depth of Rigetti native circuits is:
# $$
# U = R_Z(\omega) R_X(\frac{\pi}{2}) R_Z(\theta) R_X( -\frac{\pi}{2}) R_Z(\phi)\\
# $$
# If we take a uniform distribution over U, it induces the following conditional distribution over $\phi, \omega$ and $\theta$:
# $$
# P\{ \theta \} = \frac{\sin \theta}{2\pi}\\
# P\{ \phi | \theta \} = \frac{1}{2\pi} \\
# P\{ \omega | \theta,\phi \} = \frac{1}{2\pi}
# $$

# In this project we benchmark with those conditional probabilities

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
import copy
from tqdm import tqdm_notebook as tqdm


# In[3]:


from functions import *


# In[4]:


if __name__ == "__main__":
    target_qubit = [5]
    num_qubits = len(target_qubit)

#     First step choose m and the K_m sequences of 
    m = 1
    k_m = 1 #n. of diff sequences
    n_m = 1  #n. of samples from a certain sequence


# In[ ]:





# In[5]:


def native_rigetti_packs_generator(qmachine, target_qubit:int, num_layer:int):
    list_gates = []
    angles = np.linspace(0, np.pi, 100)
    
    for index in range(0,num_layer):
        omega, phi = np.random.uniform(0, 2*np.pi, size = 2)
        theta = np.random.choice(angles, p = np.sin(angles) / np.sum( np.sin(angles) ))
        
        draft_circuit = Program( [RZ(phi, qubit = target_qubit),
                                  RY(theta, qubit = target_qubit),
                                  RZ(omega, qubit = target_qubit)])
        
        list_gates.extend(qmachine.compiler.quil_to_native_quil(draft_circuit))
    
    list_gates = [ ins for ins in list_gates if isinstance(ins, Gate)]
    return list_gates


# In[ ]:





# In[ ]:





# In[6]:


def machine_response_rb_native_gate_conditional_single_qubit(qmachine, target_qubit, m, k_m, n_m):
    """
    It samples and record the accept or reject of the machine with native gates chosen with conditions for rigetti.
    ::return response_matrix including accepts and rejects in columns
    """
    if type(target_qubit) == list:
        target_qubit = target_qubit[0]
        
    response_matrix = np.zeros((k_m,n_m))
    
    for i_sequ in tqdm(range(k_m)):
        gate_list = native_rigetti_packs_generator(qmachine, target_qubit, m)
        prog = Program() #All qubits begin with |0> state
        
        for gate in gate_list:
            prog += gate
        
        #Come back to our initial state
#         for gate in reversed(gate_list):
# #             prog += copy.deepcopy(gate).dagger() #dagger has replacing operations
#             gate_daggered = copy.deepcopy(gate)
#             gate_daggered.params[0] *= -1 #make daggered rotation 
#             prog += gate_daggered

        u_inverse_definition = DefGate('U_inverse', np.linalg.inv(program_unitary(prog, n_qubits=1)))
        U_inverse = u_inverse_definition.get_constructor()
        prog += qmachine.compiler.quil_to_native_quil(Program(u_inverse_definition, U_inverse(target_qubit)))
        
        #Do not let the quilc to alter the gates by optimization
        prog = Program('PRAGMA PRESERVE_BLOCK') + prog
        prog += Program('PRAGMA END_PRESERVE_BLOCK')
        
        #Measurments
        ro = prog.declare('ro', 'BIT', 1)
        for q in range(1):
            prog += MEASURE(target_qubit, ro[q])
        prog = prog.wrap_in_numshots_loop(n_m)

        #Run the program
        executable = qmachine.compile(prog)
        result = qmachine.run(executable)
        measured_outcome = result.readout_data.get('ro')

        response_matrix[i_sequ,:] = 1 - np.bool_(np.sum(measured_outcome, axis = 1)) # 1 if it is equal to n_zero state
    return response_matrix


# In[7]:


if __name__ == "__main__":
#     qc = get_qc( str(num_qubits) + 'q-qvm')  # You can make any 'nq-qvm'
    qc = get_qc("9q-square-noisy-qvm")
    response = machine_response_rb_native_gate_conditional_single_qubit(qc, [0], m, k_m, n_m)


# In[8]:


if __name__ == "__main__":
    get_ipython().system('jupyter nbconvert RB_with_Rigetti_native_gates_conditional_probability.ipynb --to python')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




