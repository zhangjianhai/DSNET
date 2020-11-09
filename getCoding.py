# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:09:24 2019

@author: Administrator
"""
import numpy as np
#import matplotlib.pyplot as plt
def getCoding(sess, sample, coding, pred_sample, testset, batch_size):


    dim_coding= np.int(coding.shape[1])
    
    # Component : Predition and Reconstruction ---------------------------------------------------------------
#    pred_test=np.zeros([0,testset.shape[1],testset.shape[2],testset.shape[3]])
    coding_test=np.zeros([0,dim_coding])
    if np.mod(testset.shape[0],batch_size)==0:
        epochs_test=np.int(testset.shape[0]/batch_size)
        for i in range(0,epochs_test):
            coding_sess=sess.run(coding, feed_dict={sample: testset[i*batch_size:(i+1)*batch_size,:,:,:]})
            coding_test=np.vstack((coding_test,coding_sess))
#            pred_test=np.vstack((pred_test,sess.run(pred_sample, feed_dict={coding: coding_sess})))
    else:
        epochs_test=np.int(testset.shape[0]/batch_size+1)
        for i in range(0,epochs_test):
            if (i+1)*batch_size<testset.shape[0]:
                coding_sess=sess.run(coding, feed_dict={sample: testset[i*batch_size:(i+1)*batch_size,:,:,:]})
                coding_test=np.vstack((coding_test,coding_sess))
#                pred_test=np.vstack((pred_test,sess.run(pred_sample, feed_dict={coding: coding_sess})))
    
            else:
                if i!=0:
                    temp=sess.run(coding, feed_dict={sample: testset[testset.shape[0]-batch_size:testset.shape[0],:,:,:]})
    #                temp2=sess.run(pred_sample, feed_dict={coding: temp})
                    
                    coding_sess=temp[(batch_size-testset.shape[0]+i*batch_size):,:]
                    coding_test=np.vstack((coding_test,coding_sess))
                else:
                    #????
                    remain=np.int(batch_size/testset.shape[0]+1)
                    N=testset.shape[0]
                    temp=testset
                    for j in range(0,remain):
                        testset=np.vstack((testset,temp))
                        
                    temp=sess.run(coding, feed_dict={sample: testset[0:batch_size,:,:,:]})
    #                temp2=sess.run(pred_sample, feed_dict={coding: temp})
                    
                    coding_sess=temp[0:N,:]
                    coding_test=np.vstack((coding_test,coding_sess))
#                temp3=temp2[(batch_size-testset.shape[0]+i*batch_size):,:,:,:]
#                pred_test=np.vstack((pred_test,temp3))
            
    return coding_test       
     


#import matplotlib.pyplot as plt
##------------------------------------------------
#plt.figure(1)
#plt.imshow(x_test[0,:,:,:].squeeze())
#
#plt.figure(2)
#plt.imshow(pred_test[0,:,:,:].squeeze())
##------------------------------------------------


       