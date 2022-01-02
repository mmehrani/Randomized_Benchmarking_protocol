import numpy as np

# calculate the average fidelity for a given m
def averageOfFidelity(Data_Array, k_m, n_m):
    Pacc_Array = np.array( [sum(i) for i in Data_Array] ) / n_m
    if k_m == sum(Pacc_Array):
        return 1
    R_Array = np.array([int(1/np.log(1/p_jm)+0.5) for p_jm in Pacc_Array] ) #nearest integer is required
    PaccR_Array = [Pacc_Array[i] ** R_Array[i] for i in range(len(Pacc_Array))]
    Lower_Bound_F = [1 - (R_Array[i] * Pacc_Array[i])**(-1) for i in range(len(PaccR_Array))]
    AvergeOf_F = sum(Lower_Bound_F) / k_m
    return AvergeOf_F