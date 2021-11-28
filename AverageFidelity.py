import numpy

# calculate the average fidelity for a given m
def averageOfFidelity(Data_Array, k_m, n_m, R):
    Pacc_Array = [sum(i) for i in Data_Array] / n_m
    PaccR_Array = [x ** R for x in Pacc_Array]
    Lower_Bound_F = [1 / (R * x) for x in PaccR_Array]
    AvergeOf_F = sum(Lower_Bound_F) / k_m
    return AvergeOf_F
