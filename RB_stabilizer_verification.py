#!/usr/bin/env python
# coding: utf-8

# # Randomized Benchmarking with Stabilizer Verification

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
from functions import averageOfFidelity, qvirtual_machine, qreal_machine


# In[3]:


if __name__ == "__main__":
    num_qubits = 3
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


def stab_transform(current_stab, gate_in_circuit):
    if gate_in_circuit.name == 'H':
        performing_qubit = gate_in_circuit.qubits[0].index #since it is a single-qubit
        if current_stab[performing_qubit].name == 'Z':
            current_stab[performing_qubit] = X(performing_qubit)
        elif current_stab[performing_qubit].name == 'X':
            current_stab[performing_qubit] = Z(performing_qubit)
        elif current_stab[performing_qubit].name == 'Y':
            current_stab[performing_qubit] = Y(performing_qubit)
        elif current_stab[performing_qubit].name == 'I':
            current_stab[performing_qubit] = I(performing_qubit)
        else:
            print(gate_in_circuit.name, current_stab)
            raise Exception('This is the exception you expect to handle')
    
    elif gate_in_circuit.name == 'S':
        performing_qubit = gate_in_circuit.qubits[0].index #since it is a single-qubit
        if current_stab[performing_qubit].name == 'Z':
            current_stab[performing_qubit] = Z(performing_qubit)
        elif current_stab[performing_qubit].name == 'X':
            current_stab[performing_qubit] = Y(performing_qubit)
        elif current_stab[performing_qubit].name == 'Y':
            current_stab[performing_qubit] = X(performing_qubit)
        elif current_stab[performing_qubit].name == 'I':
            current_stab[performing_qubit] = I(performing_qubit)
        else:
            print(gate_in_circuit.name, current_stab)
            raise Exception('This is the exception you expect to handle')

            
    elif gate_in_circuit.name == 'CNOT':
        performing_qubits = [qubit.index for qubit in gate_in_circuit.qubits] #since it is a single-qubit
        stabs_names = [qubit.name for qubit in current_stab[performing_qubits]]
        if stabs_names == ['Z','I']:
            current_stab[performing_qubits] = [Z(performing_qubits[0]), I(performing_qubits[1])]
        elif stabs_names == ['X','I']:
            current_stab[performing_qubits] = [X(performing_qubits[0]), X(performing_qubits[1])]
        elif stabs_names == ['Y','I']:
            current_stab[performing_qubits] = [Y(performing_qubits[0]), X(performing_qubits[1])]
        elif stabs_names == ['I','I']:
            current_stab[performing_qubits] = [I(performing_qubits[0]), I(performing_qubits[1])]

        elif stabs_names == ['Z','Z']:
            current_stab[performing_qubits] = [I(performing_qubits[0]), Z(performing_qubits[1])]
        elif stabs_names == ['X','Z']:
            current_stab[performing_qubits] = [Y(performing_qubits[0]), Y(performing_qubits[1])]
        elif stabs_names == ['Y','Z']:
            current_stab[performing_qubits] = [X(performing_qubits[0]), Y(performing_qubits[1])]
        elif stabs_names == ['I','Z']:
            current_stab[performing_qubits] = [Z(performing_qubits[0]), Z(performing_qubits[1])]

        elif stabs_names == ['Z','X']:
            current_stab[performing_qubits] = [Z(performing_qubits[0]), X(performing_qubits[1])]
        elif stabs_names == ['X','X']:
            current_stab[performing_qubits] = [X(performing_qubits[0]), I(performing_qubits[1])]
        elif stabs_names == ['Y','X']:
            current_stab[performing_qubits] = [Y(performing_qubits[0]), I(performing_qubits[1])]
        elif stabs_names == ['I','X']:
            current_stab[performing_qubits] = [I(performing_qubits[0]), X(performing_qubits[1])]

        elif stabs_names == ['Z','Y']:
            current_stab[performing_qubits] = [I(performing_qubits[0]), Y(performing_qubits[1])]
        elif stabs_names == ['X','Y']:
            current_stab[performing_qubits] = [Y(performing_qubits[0]), Z(performing_qubits[1])]
        elif stabs_names == ['Y','Y']:
            current_stab[performing_qubits] = [X(performing_qubits[0]), Z(performing_qubits[1])]
        elif stabs_names == ['I','Y']:
            current_stab[performing_qubits] = [Z(performing_qubits[0]), Y(performing_qubits[1])]
        else:
            print(gate_in_circuit.name, current_stab)
            raise Exception('This is the exception you expect to handle')
    else:
        print(gate_in_circuit.name, current_stab)
        raise Exception('This is the exception you expect to handle')
    return current_stab

def update_stabilizer(init_stab, gates_sequence):
    stab = init_stab
    for layer in gates_sequence:
        stab = stab_transform( stab, layer )
    return stab


# In[6]:


def machine_response_stabilizer_bench(qmachine, num_qubits, m, k_m, n_m):
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

        initial_stabilizer = []
        for q_num in range(num_qubits):
            initial_stabilizer.append( np.random.choice([Z(q_num), Z(q_num)]) )
        stabilizer_layer = update_stabilizer( np.array(initial_stabilizer), c_jm )
        
        prog+= Program(*stabilizer_layer)
        
        #Do not let the quilc to alter the gates by optimization
        prog = Program('PRAGMA INITIAL_REWIRING "NAIVE"') + Program('PRAGMA PRESERVE_BLOCK') + prog
        prog += Program('PRAGMA END_PRESERVE_BLOCK')
        
        #Measurments
        ro = prog.declare('ro', 'BIT', num_qubits)
        for q in range(num_qubits):
            prog += MEASURE(q, ro[q])

        prog = prog.wrap_in_numshots_loop(n_m)

        #Run the program
#         qc = get_qc( str(num_qubits) + 'q-qvm')  # You can make any 'nq-qvm'
        executable = qmachine.compile(prog)
        result = qmachine.run(executable)
        measured_outcome = result.readout_data.get('ro')

        response_matrix[i_sequ,:] = 1 - np.bool_(np.sum(measured_outcome, axis = 1)) # 1 if it is equal to n_zero state
    
    return response_matrix
    


# In[7]:


# qc = get_qc( str(num_qubits) + 'q-qvm')  # You can make any 'nq-qvm'
# machine_response_stabilizer_bench(qc,num_qubits, m, k_m, n_m)


# In[8]:


if __name__ == "__main__":
    get_ipython().system('ipython nbconvert --to python RB_stabilizer_verification.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:




