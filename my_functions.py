# -*- coding: utf-8 -*-

import numpy as np
def getSamples(seq_coding1,seq_coding2,min_path):
    N=len(min_path)
    input1=np.zeros([N,112,112,1],np.float32)
    input2=np.zeros([N,112,112,1],np.float32)
    for i in range(0,N):
        
        input1[i,:,:,:]=seq_coding1[min_path[i][0],:,:,:]
        input2[i,:,:,:]=seq_coding2[min_path[i][1],:,:,:]
        
    return input1, input2
