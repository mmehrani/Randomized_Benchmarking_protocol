# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:09:28 2021

@author: mohsen
"""


import numpy as np

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

def averageOfFidelity(Data_Array):
    
    k_m, n_m = Data_Array.shape
    Pacc_Array = np.array( [sum(i) for i in Data_Array] ) / n_m
    
    vfunc = np.vectorize(calculate_lower_bound)
    Lower_Bound_F = vfunc(Pacc_Array)
    AvergeOf_F = sum(Lower_Bound_F) / k_m
    return AvergeOf_F
