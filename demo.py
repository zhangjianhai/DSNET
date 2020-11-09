# -*- coding: utf-8 -*-

import scipy.io
import utilize_functions as uf
import utilize_functions2 as uf2
import my_functions as mf
import numpy as np
from keras.utils import np_utils
import auto_encoder
import getCoding
import dtw_tool
import random


ske=scipy.io.loadmat('data/ske.mat')
ske=ske['ske']

ske_v=scipy.io.loadmat('data/skeletal_data_validity.mat')
ske_v=ske_v['skeletal_data_validity']

t=6
train_set,test_set=uf.process_data(ske, ske_v, t)

# 生成训练样本
action_seg=uf.getS(t,train_set,ske)
x_train=uf.getImages(action_seg,t)
x_train=x_train[:,:,:,np.newaxis].astype('float32')
train_label=train_set[:,2]
train_label_one_hot=np_utils.to_categorical(train_label, num_classes=10)

# 生成测试样本
action_seg=uf.getS(t,test_set,ske)
x_test=uf.getImages(action_seg,t)
x_test=x_test[:,:,:,np.newaxis].astype('float32')
test_label=test_set[:,2]
test_label_one_hot=np_utils.to_categorical(test_label, num_classes=10)

sess, sample, coding, pred_sample=auto_encoder.AutoEncoderTrain(x_train, x_test, num_epoch=2, batch_size=256, flag_train=False)

#pred_x_test, loss=auto_encoder.AutoEncoderTrain(x_train, x_test, num_epoch=2, batch_size=256, flag_train=False)

#pred_x_test1, loss1=auto_encoder.AutoEncoderTrain(x_train, x_test, num_epoch=0)

#coding=getCoding.getCoding(sess, sample, coding, pred_sample, x_test, batch_size=256)



#------------------------------------------------------------------------------

import tensorflow as tf
import os 

sess_s=tf.InteractiveSession()
#------------------------------------------------------------------------------
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#------------------------------------------------------------------------------
sample1=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
sample2=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
y_sample1 = tf.placeholder(dtype=tf.float32, shape=[None, 10])
y_sample2 = tf.placeholder(dtype=tf.float32, shape=[None, 10])
res_sample1=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
res_sample2=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
y_dis=tf.placeholder(dtype=tf.float32, shape=[None, 1])



#------------------------------------------------------------------------------
# Parameters in CommonNet
W_fc1_CommonNet = weight_variable([1024, 2048])
b_fc1_CommonNet = bias_variable([2048])
W_fc2_CommonNet = weight_variable([2048, 1024])
b_fc2_CommonNet = bias_variable([1024])
W_fc3_CommonNet = weight_variable([1024, 1024])
b_fc3_CommonNet = bias_variable([1024])
# Parameters in IdentityNet
W_fc1_IdentityNet = weight_variable([1024, 2048])
b_fc1_IdentityNet = bias_variable([2048])
W_fc2_IdentityNet = weight_variable([2048, 1024])
b_fc2_IdentityNet = bias_variable([1024])
W_fc3_IdentityNet = weight_variable([1024, 1024])
b_fc3_IdentityNet = bias_variable([1024])
# Parameters in Discriminator
W_fc1_discriminator = weight_variable([1024, 1024])
b_fc1_discriminator = bias_variable([1024])
W_fc2_discriminator = weight_variable([1024, 512])
b_fc2_discriminator = bias_variable([512])
W_fc3_discriminator = weight_variable([512, 10])
b_fc3_discriminator = bias_variable([10])
# Parameters in MetricNet
W_fc1_metricnet = weight_variable([1024, 512])
b_fc1_metricnet = bias_variable([512])
W_fc2_metricnet = weight_variable([512, 128])
b_fc2_metricnet = bias_variable([128])
W_fc3_metricnet = weight_variable([128, 1])
b_fc3_metricnet = bias_variable([1])
# Parameters in MixNet
W_fc1_mix = weight_variable([2048, 2048])
b_fc1_mix = bias_variable([2048])
W_fc2_mix = weight_variable([2048, 1024])
b_fc2_mix = bias_variable([1024])
# Parameters in Decoder
W_fc1_Decoder = weight_variable([1024, 1024])
b_fc1_Decoder = bias_variable([1024])
W_fc2_Decoder = weight_variable([1024, 2048])
b_fc2_Decoder = bias_variable([2048])
W_fc3_Decoder = weight_variable([2048, 1024])
b_fc3_Decoder = bias_variable([1024])





#------------------------------------------------------------------------------

def CommonNet(x):
    # h_fc1: 2048
    h_fc1_CommonNet = tf.nn.relu(tf.matmul(x, W_fc1_CommonNet) + b_fc1_CommonNet)
    # h_fc2: 1024
    h_fc2_CommonNet = tf.nn.relu(tf.matmul(h_fc1_CommonNet, W_fc2_CommonNet) + b_fc2_CommonNet)
    # h_fc3: 1024
    h_fc3_CommonNet = tf.nn.sigmoid(tf.matmul(h_fc2_CommonNet, W_fc3_CommonNet) + b_fc3_CommonNet)
    return h_fc3_CommonNet


def IdentityNet(x):
    # h_fc1: 2048
    h_fc1_IdentityNet = tf.nn.relu(tf.matmul(x, W_fc1_IdentityNet) + b_fc1_IdentityNet)
    # h_fc2: 1024
    h_fc2_IdentityNet = tf.nn.relu(tf.matmul(h_fc1_IdentityNet, W_fc2_IdentityNet) + b_fc2_IdentityNet)
    # h_fc3: 1024
    h_fc3_IdentityNet = tf.nn.sigmoid(tf.matmul(h_fc2_IdentityNet, W_fc3_IdentityNet) + b_fc3_IdentityNet)
    return h_fc3_IdentityNet

def Discriminator(x):
    # h_fc1_discriminator: 1024
    h_fc1_discriminator = tf.nn.relu(tf.matmul(x, W_fc1_discriminator) + b_fc1_discriminator)
    # h_fc2_discriminator: 512
    h_fc2_discriminator = tf.nn.relu(tf.matmul(h_fc1_discriminator, W_fc2_discriminator) + b_fc2_discriminator)
    # h_fc2_discriminator: 10
    h_fc3_discriminator = tf.nn.softmax(tf.matmul(h_fc2_discriminator, W_fc3_discriminator) + b_fc3_discriminator)
    return h_fc3_discriminator

def MetricNet(x):
    # h_fc1_metricnet: 512
    h_fc1_metricnet = tf.nn.relu(tf.matmul(x, W_fc1_metricnet) + b_fc1_metricnet)
    # h_fc2_metricnet: 128
    h_fc2_metricnet = tf.nn.relu(tf.matmul(h_fc1_metricnet, W_fc2_metricnet) + b_fc2_metricnet)
    # h_fc2_metricnet: 1
    h_fc3_metricnet = tf.nn.sigmoid(tf.matmul(h_fc2_metricnet, W_fc3_metricnet) + b_fc3_metricnet)
    return h_fc3_metricnet

def MixNet(x):
    # h_fc1_mix: 2048
    h_fc1_mix = tf.nn.relu(tf.matmul(x, W_fc1_mix) + b_fc1_mix)
    # h_fc2_mix: 1024
    h_fc2_mix = tf.nn.relu(tf.matmul(h_fc1_mix, W_fc2_mix) + b_fc2_mix)

    return h_fc2_mix

def Decoder(x):
    # h_fc1: 1024
    h_fc1_Decoder = tf.nn.relu(tf.matmul(x, W_fc1_Decoder) + b_fc1_Decoder)
    # h_fc2: 2048
    h_fc2_Decoder = tf.nn.relu(tf.matmul(h_fc1_Decoder, W_fc2_Decoder) + b_fc2_Decoder)
    # h_fc3: 1024
    h_fc3_Decoder = tf.nn.sigmoid(tf.matmul(h_fc2_Decoder, W_fc3_Decoder) + b_fc3_Decoder)
    return h_fc3_Decoder


common1=CommonNet(sample1)
common2=CommonNet(sample2)

identity1=IdentityNet(sample1)
identity2=IdentityNet(sample2)

sample1_out=Discriminator(identity1)
sample2_out=Discriminator(identity2)

mix1=tf.concat([common2,identity1],1)
mix2=tf.concat([common1,identity2],1)

coding1=MixNet(mix1)
coding2=MixNet(mix2)

pred_sample1=Decoder(coding1)
pred_sample2=Decoder(coding2)

metric_in=tf.square(identity1-identity2)
dis=MetricNet(metric_in)
#------------------------------------------------------------------------------
# Loss of Discriminator
loss1= -tf.reduce_sum(y_sample1*tf.log(tf.maximum(sample1_out,1e-10)))
loss2= -tf.reduce_sum(y_sample2*tf.log(tf.maximum(sample2_out,1e-10)))
#loss1=tf.reduce_mean(tf.square(y_sample1-sample1_out))
#loss2=tf.reduce_mean(tf.square(y_sample2-sample2_out))
# Loss of Decoder
#loss_pred1=tf.reduce_mean(tf.square(pred_sample1-res_sample1))
#loss_pred2=tf.reduce_mean(tf.square(pred_sample2-res_sample2))
loss_pred1=-tf.reduce_sum(res_sample1*tf.log(tf.maximum(pred_sample1,1e-10)))
loss_pred2=-tf.reduce_sum(res_sample2*tf.log(tf.maximum(pred_sample2,1e-10)))
# Loss of Metric
loss_dis=-tf.reduce_sum(y_dis*tf.log(tf.maximum(dis,1e-10)))

#loss_dis=tf.reduce_mean(tf.square(y_dis-dis))
# Loss of Common
loss_common=-tf.reduce_sum(common1*tf.log(tf.maximum(common2,1e-10)))
#loss_common=tf.reduce_mean(tf.square(common1-common2))
# Overall loss 
loss=loss1+loss2+loss_pred1+loss_pred2+loss_dis+loss_common

lr=1e-6
train= tf.train.AdamOptimizer(lr).minimize(loss)

sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
if os.path.exists('ckpt'):
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,model_file)

#------------------------------------------------------------------------------
# data generalization
    
sample_input1=np.zeros([0,1024],np.float32)
sample_input2=np.zeros([0,1024],np.float32)
sample_input1_y=np.zeros([0,10],np.float32)
sample_input2_y=np.zeros([0,10],np.float32)
sample_metric=np.zeros([0,1],np.float32)


for a in range(0,10):
    for s1 in range(0,10):
        for s2 in range(0,10):
            if ske_v[a,s1,0]==1 and ske_v[a,s2,0]==1:
                print(a,'+',s1, '+' ,s2)
                if ske[a,s1,0].shape[2]>10 and ske[a,s2,0].shape[2]>10:
                    sequence=uf2.process_data(ske, ske_v, t, [a], [s1] ,[0])
                    sequence=uf.getS(t,sequence,ske)  # 用uf和uf2都一样，在这里 
                    sequence_images=uf.getImages(sequence,t)
                    sequence_images1=sequence_images[:,:,:,np.newaxis].astype('float32')
                    
                    sequence=uf2.process_data(ske, ske_v, t, [a], [s2] ,[0])
                    sequence=uf.getS(t,sequence,ske)  # 用uf和uf2都一样，在这里 
                    sequence_images=uf.getImages(sequence,t)
                    sequence_images2=sequence_images[:,:,:,np.newaxis].astype('float32')
                    
                    seq_coding1=getCoding.getCoding(sess, sample, coding, pred_sample, sequence_images1, batch_size=256)
                    seq_coding2=getCoding.getCoding(sess, sample, coding, pred_sample, sequence_images2, batch_size=256)
        
                    min_dis, min_path=dtw_tool.DTW_tool(seq_coding1,seq_coding2)
                    
                    input1,input2=mf.getSamples(seq_coding1,seq_coding2,min_path)
                    
                    input1_y=np.ones([input1.shape[0]])*s1
                    input1_y_one_hot=np_utils.to_categorical(input1_y, num_classes=10)
                    
                    input2_y=np.ones([input2.shape[0]])*s2
                    input2_y_one_hot=np_utils.to_categorical(input2_y, num_classes=10)
        
                    sample_input1=np.vstack((sample_input1,input1))
                    sample_input2=np.vstack((sample_input2,input2))
                    sample_input1_y=np.vstack((sample_input1_y,input1_y_one_hot))
                    sample_input2_y=np.vstack((sample_input2_y,input2_y_one_hot))
                    if s1==s2:
                        dis_temp=np.ones([input1.shape[0],1])
                    else:
                        dis_temp=np.zeros([input1.shape[0],1])
                    sample_metric=np.vstack((sample_metric,dis_temp))
            

N_t=sample_input1.shape[0]
batch_size=256
epochs=np.int(np.floor(N_t/batch_size)+1) 


loss_overall=[]
for repeat in range(0,10):
    
    order_shuffled=random.sample(range(0,N_t),N_t)
    sample_input1=sample_input1[order_shuffled,:]
    sample_input2=sample_input2[order_shuffled,:]
    sample_input1_y=sample_input1_y[order_shuffled,:]
    sample_input2_y=sample_input2_y[order_shuffled,:]
    sample_metric=sample_metric[order_shuffled,:]
    
    
    loss_batch=[]
    for i in range(0,epochs):

        if (i+1)*batch_size<N_t:
            sample_input1_seg=sample_input1[i*batch_size:(i+1)*batch_size,:]
            sample_input2_seg=sample_input2[i*batch_size:(i+1)*batch_size,:]
              
            t_label_one_hot_seg=sample_input1_y[i*batch_size:(i+1)*batch_size,:]
            s_label_one_hot_seg=sample_input2_y[i*batch_size:(i+1)*batch_size,:]
            
            similar_seg=sample_metric[i*batch_size:(i+1)*batch_size,:]
        
            _,b,c,d,e,f,g,h=sess.run([train,loss,loss1,loss2,loss_pred1,loss_pred2,loss_dis,loss_common],feed_dict={sample1: sample_input1_seg, sample2: sample_input2_seg, y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:sample_input1_seg, res_sample2:sample_input2_seg, y_dis:similar_seg})
            loss_batch.append(b)

#        else:
#            sample_input1_seg=sample_input1[N_t-batch_size:N_t,:]
#            sample_input2_seg=sample_input2[N_t-batch_size:N_t,:]
#             
#            t_label_one_hot_seg=sample_input1_y[N_t-batch_size:N_t,:]
#            s_label_one_hot_seg=sample_input2_y[N_t-batch_size:N_t,:]
#        
#            similar_seg=sample_metric[N_t-batch_size:N_t,:]
#        
#            b=sess.run([train,loss,loss1,loss2,loss_pred1,loss_pred2,loss_dis,loss_common],feed_dict={sample1: sample_input1_seg, sample2: sample_input2_seg, y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:sample_input1_seg, res_sample2:sample_input2_seg, y_dis:similar_seg})

    loss_overall.append(np.mean(loss_batch))
        
#        print('Repeat:{},Epoch:{},loss:{}, loss1:{}, loss2:{}, loss_pred1:{}, loss_pred2:{}, loss_dis:{}, loss_common:{}'.format(repeat, i, loss_overall[-1][1],loss_overall[-1][2],loss_overall[-1][3],loss_overall[-1][4],loss_overall[-1][5],loss_overall[-1][6],loss_overall[-1][7]))
#    
    
    test_coding=getCoding.getCoding(sess, sample, coding, pred_sample, x_test, batch_size=256)    
    
    l=(sess.run([sample1_out],feed_dict={sample1: test_coding}))[0]
    pred_label=[]
    for i in range(0,l.shape[0]):
        pred_label.append((l[i,:]==(np.max(l,1))[i]).tolist())
    
    pred_label=np.array(pred_label)
    cr=1-len((np.nonzero(np.sum(np.abs(test_label_one_hot-pred_label),1)))[0])/x_test.shape[0]
    
    
    print('Repeat1:{},loss:{},cr:{}'.format(repeat+1,loss_overall[-1],cr)) 
    
    
    
    
    

    
  #-----------------------
loss_overall=[]
for repeat in range(0,1000):
    
    sample_input1=np.zeros([0,1024],np.float32)
    sample_input2=np.zeros([0,1024],np.float32)
    sample_input1_y=np.zeros([0,10],np.float32)
    sample_input2_y=np.zeros([0,10],np.float32)
    sample_metric=np.zeros([0,1],np.float32)
    
    
    for a in range(0,10):
        print(a)
        for s1 in range(0,10):
            for s2 in range(0,10):
                if ske_v[a,s1,0]==1 and ske_v[a,s2,0]==1:
                    
                    if ske[a,s1,0].shape[2]>10 and ske[a,s2,0].shape[2]>10:
                        sequence=uf2.process_data(ske, ske_v, t, [a], [s1] ,[0])
                        sequence=uf.getS(t,sequence,ske)  # 用uf和uf2都一样，在这里 
                        sequence_images=uf.getImages(sequence,t)
                        sequence_images1=sequence_images[:,:,:,np.newaxis].astype('float32')
                        
                        sequence=uf2.process_data(ske, ske_v, t, [a], [s2] ,[0])
                        sequence=uf.getS(t,sequence,ske)  # 用uf和uf2都一样，在这里 
                        sequence_images=uf.getImages(sequence,t)
                        sequence_images2=sequence_images[:,:,:,np.newaxis].astype('float32')
                        
                        seq_coding1=getCoding.getCoding(sess, sample, coding, pred_sample, sequence_images1, batch_size=256)
                        seq_coding2=getCoding.getCoding(sess, sample, coding, pred_sample, sequence_images2, batch_size=256)
                        
                        c1=(sess.run([common1],feed_dict={sample1: seq_coding1}))[0]
                        c2=(sess.run([common2],feed_dict={sample2: seq_coding2}))[0]
                             
                        min_dis, min_path=dtw_tool.DTW_tool(c1,c2)
                        
                        input1,input2=mf.getSamples(seq_coding1,seq_coding2,min_path)
                        
                        input1_y=np.ones([input1.shape[0]])*s1
                        input1_y_one_hot=np_utils.to_categorical(input1_y, num_classes=10)
                        
                        input2_y=np.ones([input2.shape[0]])*s2
                        input2_y_one_hot=np_utils.to_categorical(input2_y, num_classes=10)
            
                        sample_input1=np.vstack((sample_input1,input1))
                        sample_input2=np.vstack((sample_input2,input2))
                        sample_input1_y=np.vstack((sample_input1_y,input1_y_one_hot))
                        sample_input2_y=np.vstack((sample_input2_y,input2_y_one_hot))
                        if s1==s2:
                            dis_temp=np.ones([input1.shape[0],1])
                        else:
                            dis_temp=np.zeros([input1.shape[0],1])
                        sample_metric=np.vstack((sample_metric,dis_temp))
                  
    N_t=sample_input1.shape[0]
    for repeat2 in range(0,3):
        order_shuffled=random.sample(range(0,N_t),N_t)
        sample_input1=sample_input1[order_shuffled,:]
        sample_input2=sample_input2[order_shuffled,:]
        sample_input1_y=sample_input1_y[order_shuffled,:]
        sample_input2_y=sample_input2_y[order_shuffled,:]
        sample_metric=sample_metric[order_shuffled,:]
    
        
        
        loss_batch=[]
        for i in range(0,epochs):
    
            if (i+1)*batch_size<N_t:
                sample_input1_seg=sample_input1[i*batch_size:(i+1)*batch_size,:]
                sample_input2_seg=sample_input2[i*batch_size:(i+1)*batch_size,:]
                  
                t_label_one_hot_seg=sample_input1_y[i*batch_size:(i+1)*batch_size,:]
                s_label_one_hot_seg=sample_input2_y[i*batch_size:(i+1)*batch_size,:]
                
                similar_seg=sample_metric[i*batch_size:(i+1)*batch_size,:]
            
                _,b,c,d,e,f,g,h=sess.run([train,loss,loss1,loss2,loss_pred1,loss_pred2,loss_dis,loss_common],feed_dict={sample1: sample_input1_seg, sample2: sample_input2_seg, y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:sample_input1_seg, res_sample2:sample_input2_seg, y_dis:similar_seg})
                loss_batch.append(b)
    
        loss_overall.append(np.mean(loss_batch))

        
        
        test_coding=getCoding.getCoding(sess, sample, coding, pred_sample, x_test, batch_size=256)    
        
        l=(sess.run([sample1_out],feed_dict={sample1: test_coding}))[0]
        pred_label=[]
        for i in range(0,l.shape[0]):
            pred_label.append((l[i,:]==(np.max(l,1))[i]).tolist())
        
        pred_label=np.array(pred_label)
        cr=1-len((np.nonzero(np.sum(np.abs(test_label_one_hot-pred_label),1)))[0])/x_test.shape[0]
        
        
        print('Repeat1:{},Repeart2:{},loss:{},cr:{}'.format(repeat+1,repeat2+1,loss_overall[-1],cr))    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# 