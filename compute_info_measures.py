#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:45:07 2017

@author: Allison
"""

#function to compute information measures from a pdf

import numpy as np

def compute_info_measures(pdf):
    
    
    pdfsize = np.shape(pdf)
    N = pdfsize[1]
    dim = np.size(pdfsize)
    
    
    if dim == 3:
        Hx1 = np.zeros(1) #individual entropies of sources and target
        Hx2 = np.zeros(1)
        Hy = np.zeros(1)
        Ix1x2 = np.zeros(1) #mutual info between sources
        Ix1y = np.zeros(1) #individual information components
        Ix2y = np.zeros(1)
        Ix1x2y = np.zeros(1) #total information from both sources to target
        Is = np.zeros(1) #I_s term to compute redundancy
        R_min = np.zeros(1) #min threshold for redundancy
        R_MMI = np.zeros(1) #max threshold for redundancy
        II = np.zeros(1) #interaction info
        U1 = np.zeros(1)
        U2 = np.zeros(1)
        R = np.zeros(1)
        S = np.zeros(1)
        Imiss = np.zeros(1) #"missing info" or info not provided by sources

     #compute marginal pdfs            
        m_jk = np.sum(pdf,axis=0)
        m_ij = np.sum(pdf,axis=2)
        m_ik = np.sum(pdf,axis=1)
        
        m_i = np.sum(m_ij,axis=1)
        m_j = np.sum(m_ij,axis=0)
        m_k = np.sum(m_jk,axis=0)
        
        H1_vect = -1 * m_i * np.log2(m_i)
        H1_vect[np.isnan(H1_vect)]=0
        Hx1 = np.sum(H1_vect)
        
        H2_vect = -1 * m_j * np.log2(m_j)
        H2_vect[np.isnan(H2_vect)]=0
        Hx2 = np.sum(H2_vect)
        
        H3_vect = -1 * m_k * np.log2(m_k)
        H3_vect[np.isnan(H3_vect)]=0
        Hy = np.sum(H3_vect)

        #compute information measures from marginal pdfs
            
        
        for i in range(0,N):
            for j in range(0,N):
                for k in range(0,N):
                    m_ijk = pdf[i,j,k]
         
                    #I(X1;X2)
                    if k == 0 and m_ij[i,j]>0:
                        Ix1x2 = Ix1x2 + m_ij[i,j]*np.log2(m_ij[i,j]/(m_i[i]*m_j[j]))
                        
                            
                    #I(X1;Y)
                    if j == 0 and m_ik[i,k]>0:
                        Ix1y = Ix1y+ m_ik[i,k]*np.log2(m_ik[i,k]/(m_i[i]*m_k[k]))
        
                    #I(X2;Y)
                    if i == 0 and m_jk[j,k]>0:
                        Ix2y = Ix2y + m_jk[j,k]*np.log2(m_jk[j,k]/(m_j[j]*m_k[k]))
        
        
                    if m_ijk>0:
                        term=m_ijk*np.log2(m_ijk/(m_ij[i,j]*m_k[k]))
                        Ix1x2y=Ix1x2y+term
        
        
        Hvect = np.stack((Hx1,Hx2))
        Hmin = np.min(Hvect)
        if Hmin ==0:
            Is=0
            R_min=0
            R_MMI=0
        else: 
            Is = Ix1x2/Hmin
            R_MMI= np.min(np.stack((Ix1y,Ix2y)))
            R_min = np.max((-Ix1x2y + Ix1y + Ix2y,0.00))
        
        
        R = R_min + Is* (R_MMI-R_min)
        if R <=0:
            R = 0.0
            
        U1= Ix1y - R
        U2 = Ix2y - R
        S = Ix1x2y - U1-U2-R
        Imiss=Hy-Ix1x2y
        
        #update 8/29/17 - compute my own version of cause and effect information
#        UC = np.zeros(N*N) + 1/(N*N) #uniform distribution of sources
#        m_xtar = m_k; #marginal pdf of target
#        m_x1x2 = m_ij; #marginal pdf of sources (2d)
#        
#        #transition matrix
#        TPM = np.reshape(pdf,(N*N, N))
#        TPMsum = np.sum(TPM,axis=1)
#        source_dist = TPMsum
#        tar_dist = np.sum(TPM,axis=0)
#        
#        TPM = TPM.transpose()/TPMsum
#        TPM = np.nan_to_num(TPM)
#        TPMtot = np.sum(np.sum(TPM))
#        sumvect_sources = np.sum(TPM, axis=0)
#        sumvect_target = np.sum(TPM, axis=1)
#        UE = sumvect_target/TPMtot
#        
#        SP = np.zeros((N,N*N))
#        SF = np.zeros((N,N*N))
#        DKLvalSP = SP
#        DKLvalSF = SF
#        
#        nsq = N*N
#        #print(nsq)
#        
#        #for SP|source pairs: need sumvect(1xN) and TPM values
#  
#        for i in range(0,nsq):
#            SF[:,i] = TPM[:,i]/sumvect_sources[i]
#   
#        
#        for j in range(0,N):
#            SP[j,:] = TPM[j,:]/sumvect_target[j]
#     
#           
#        for i in range(0,nsq):
#            DKLvalSP[:,i]=SP[:,i]*np.log2(SP[:,i]/UC[i])
#
#        
#        for j in range(0,N):
#            DKLvalSF[j,:]=SF[j,:]*np.log2(SF[j,:]/UE[j])
#        
#        SP = np.nan_to_num(SP)
#        SF = np.nan_to_num(SF)
#        DKLvalSP = np.nan_to_num(DKLvalSP)
#        DKLvalSF = np.nan_to_num(DKLvalSF)
#        
#        #compute KL Divergence between SP and UC, SF and UE
#        CI_tars = np.sum(DKLvalSP,axis=1)
#        EI_sources = np.sum(DKLvalSF,axis=0)
#        
#        #weighted CI and EI by distribution of sources and target
#        CI_weighted = CI_tars * tar_dist
#        EI_weighted = EI_sources * source_dist
#        
#        EI_total = np.sum(EI_weighted)
#        CI_total = np.sum(CI_weighted)
           
        infodict = {'R': R,'U1':U1, 'U2':U2, 'S':S, 'Imiss':Imiss,
                    'Hxtar':Hy, 'Hxs1':Hx1, 'Hxs2':Hx2, 'Ix1x2':Ix1x2,
                    'Ix1xtar':Ix1y, 'Ix2xtar':Ix2y,'Itotal':Ix1x2y}
                
        return infodict
        
        
