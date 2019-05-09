#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:45:07 2017

@author: Allison
"""

#function to compute stat sig for binary case

import numpy as np
from compute_info_measures import compute_info_measures
 
def find_Icrit_binary(n_ones,npts,ntests):
    
    #perform statistical sig test for binary data
    X_data = np.zeros(npts,dtype=np.int)
    X_data[0:n_ones,]=1

    Itot_vector = np.zeros(ntests)
    for j in range(0,ntests):
        
        np.random.shuffle(X_data)
            
        Xs1 = X_data[0:npts-2]
        Xs2 = X_data[1:npts-1]
        Xtar = X_data[2:npts]
            
        Tuple = np.vstack((Xs1,Xs2,Xtar))
        Tuple = np.transpose(Tuple)
            
        pdf,edges = np.histogramdd(Tuple,bins=2)
        pdf = pdf/np.sum(pdf)
                
        info = compute_info_measures(pdf)
        Itot_vector[j]=info['Itotal']

    I_mean=np.average(Itot_vector)
    I_std=np.std(Itot_vector)  
    Icrit = I_mean + 3*I_std
    
    return Icrit