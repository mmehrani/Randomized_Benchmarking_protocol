#!/usr/bin/env python
# coding: utf-8

# # Standard Randomized Benchmarking with arbitrary random unitary gates

# In[1]:


from pyquil import get_qc, Program
from pyquil.gates import *
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare
from pyquil.simulation.tools import lifted_gate, program_unitary, lifted_gate_matrix
from pyquil.quil import *


# In[2]:


import numpy as np
import random
from functions import averageOfFidelity, qvirtual_machine, qreal_machine
from scipy.stats import unitary_group


# In[ ]:





# In[3]:


if __name__ == "__main__":
    num_qubits = 2

    #First step choose m and the K_m sequences of Clifford group
    m = 1
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


def bring_matrix_to_n(matrix_two_d, n_qubits, qubit_ind):
    matrix_n_d = np.eye(2**(qubit_ind))
    matrix_n_d = np.kron(matrix_n_d, matrix_two_d)
    matrix_n_d = np.kron(matrix_n_d, np.eye(2**(n_qubits - qubit_ind - 1)))
    return matrix_n_d


# In[6]:


def machine_response_standard_bench_random_units(qmachine, num_qubits, m, k_m, n_m):
    """
    It samples and record the accept or reject of the machine.
    ::return response_matrix including accepts and rejects in columns
    """
    response_matrix = np.zeros((k_m,n_m))
    
    for i_sequ in range(k_m):
        prog = Program() #All qubits begin with |0> state
            
        circuit_unitary = np.eye(2**num_qubits)
        # Add some random unitaries to the circuit
        for i in range(m):
            random_unit = unitary_group.rvs(2)
            random_choice_qubit_ind = random.choices(range(num_qubits), k = 1)[0]
            random_unit_on_all_qubits = bring_matrix_to_n(random_unit,
                                                          qubit_ind = random_choice_qubit_ind,
                                                          n_qubits = num_qubits)
            circuit_unitary = random_unit_on_all_qubits.dot(circuit_unitary)
            random_unit_definition = DefGate("U_random_{}".format(i), random_unit)
            U_random_i = random_unit_definition.get_constructor() # Get the gate constructor
            
            prog += Program( random_unit_definition, U_random_i(random_choice_qubit_ind) )
        

        #report the reversed unitary operator of the total transforamtions 
        circuit_unitary_inv = np.linalg.inv( circuit_unitary )
        circuit_unitary_inv_definition = DefGate("U_r", circuit_unitary_inv)
        U_r = circuit_unitary_inv_definition.get_constructor() # Get the gate constructor

        n_tuple = tuple(range(num_qubits))
        prog += Program( circuit_unitary_inv_definition, U_r(*n_tuple) )


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





# In[7]:


# response_matrix = machine_response_standard_bench_random_units(machine_type, num_qubits, m, k_m, n_m)
# response_matrix


# In[8]:


# averageOfFidelity(response_matrix)


# In[9]:


if __name__ == "__main__":
    get_ipython().system('ipython nbconvert --to python RB_standard_verification_with_arbitrary_random_unitaries.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




