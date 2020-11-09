# -*- coding: utf-8 -*-

import scipy.io
import utilize_functions as uf
import utilize_functions2 as uf2
import my_functions as mf
import numpy as np
from keras.utils import np_utils
import getCoding
import dtw_tool
import random
import auto_encoder


ske=scipy.io.loadmat('data/ske.mat')
ske=ske['ske']

ske_v=scipy.io.loadmat('data/skeletal_data_validity.mat')
ske_v=ske_v['skeletal_data_validity']

t=15   # the length of snippet in the paper
train_set,test_set=uf.process_data(ske, ske_v, t)

#  Generating train samples
action_seg=uf.getS(t,train_set,ske)
x_train=uf.getImages(action_seg,t)
x_train=x_train[:,:,:,np.newaxis].astype('float32')
train_label=train_set[:,2]
train_label_one_hot=np_utils.to_categorical(train_label, num_classes=10)

# Generating test samples
action_seg=uf.getS(t,test_set,ske)
x_test=uf.getImages(action_seg,t)
x_test=x_test[:,:,:,np.newaxis].astype('float32')
test_label=test_set[:,2]
test_label_one_hot=np_utils.to_categorical(test_label, num_classes=10)



sess_auto, sample, coding, pred_sample=auto_encoder.AutoEncoderTrain(x_train, x_test, num_epoch=1000, batch_size=256, flag_train=False)
#pred_x_test, loss=auto_encoder.AutoEncoderTrain(x_train, x_test, num_epoch=2, batch_size=256, flag_train=False)

#coding=getCoding.getCoding(sess_auto, sample, coding, pred_sample, x_test, batch_size=256)



#------------------------------------------------------------------------------

import tensorflow as tf
import os 

sess=tf.InteractiveSession()
batch_size=16


#------------------------------------------------------------------------------
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def deconv2d_2x2(x, W, o):
  return tf.nn.conv2d_transpose(x,W,output_shape=o,strides=[1,2,2,1],padding="SAME")

def deconv2d_1x1(x, W, o):
  return tf.nn.conv2d_transpose(x,W,output_shape=o,strides=[1,1,1,1],padding="SAME")

#------------------------------------------------------------------------------
sample1=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
sample2=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
y_sample1 = tf.placeholder(dtype=tf.float32, shape=[None, 10])
y_sample2 = tf.placeholder(dtype=tf.float32, shape=[None, 10])
res_sample1=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
res_sample2=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
y_dis=tf.placeholder(dtype=tf.float32, shape=[None, 1])



#------------------------------------------------------------------------------
# Parameters in IdentityNet
W_conv1_identitynet = weight_variable([5, 5, 1, 32])
b_conv1_identitynet = bias_variable([32])

W_conv2_identitynet = weight_variable([5, 5, 32, 64])
b_conv2_identitynet = bias_variable([64])

W_conv3_identitynet = weight_variable([5, 5, 64, 128])
b_conv3_identitynet = bias_variable([128])

W_fc1_identitynet = weight_variable([14 * 14 * 128, 2048])
b_fc1_identitynet = bias_variable([2048])

W_fc2_identitynet = weight_variable([2048, 1024])
b_fc2_identitynet = bias_variable([1024])
# Parameters in PoseNet
W_conv1_posenet = weight_variable([5, 5, 1, 32])
b_conv1_posenet = bias_variable([32])

W_conv2_posenet = weight_variable([5, 5, 32, 64])
b_conv2_posenet = bias_variable([64])

W_conv3_posenet = weight_variable([5, 5, 64, 128])
b_conv3_posenet = bias_variable([128])

W_fc1_posenet = weight_variable([14 * 14 * 128, 2048])
b_fc1_posenet = bias_variable([2048])

W_fc2_posenet = weight_variable([2048, 1024])
b_fc2_posenet = bias_variable([1024])
# Parameters in Discriminator

W_fc1_discriminator = weight_variable([1024, 2048])
b_fc1_discriminator = bias_variable([2048])

W_fc2_discriminator = weight_variable([2048, 1024])
b_fc2_discriminator = bias_variable([1024])

W_fc3_discriminator = weight_variable([1024, 10])
b_fc3_discriminator = bias_variable([10])
# Parameters in MixNet

W_fc1_mix = weight_variable([2048, 2048])
b_fc1_mix = bias_variable([2048])

W_fc2_mix = weight_variable([2048, 1024])
b_fc2_mix = bias_variable([1024])

# Parameters in DecoderNet

W_fc1_decoder = weight_variable([1024,14*14*128])
b_fc1_decoder = bias_variable([14*14*128])

W_deconv1_decoder=weight_variable([5, 5, 64, 128])
b_deconv1_decoder = bias_variable([64])

W_deconv2_decoder=weight_variable([5, 5, 32, 64])
b_deconv2_decoder = bias_variable([32])

W_deconv3_decoder=weight_variable([5, 5, 16, 32])
b_deconv3_decoder = bias_variable([16])

W_deconv4_decoder = weight_variable([5, 5, 1, 16])
b_deconv4_decoder = bias_variable([1])


# Parameters in MetricNet

W_fc1_metricnet = weight_variable([1024, 512])
b_fc1_metricnet = bias_variable([512])

W_fc2_metricnet = weight_variable([512, 128])
b_fc2_metricnet = bias_variable([128])

W_fc3_metricnet = weight_variable([128, 1])
b_fc3_metricnet = bias_variable([1])
    




#------------------------------------------------------------------------------

def IdentityNet(x):
    # h_conv1: 112*112*32
    h_conv1_identitynet = tf.nn.relu(conv2d(x, W_conv1_identitynet) + b_conv1_identitynet)
    # h_pool1: 56*56*32
    h_pool1_identitynet = max_pool_2x2(h_conv1_identitynet)
    # h_conv2: 56*56*64
    h_conv2_identitynet = tf.nn.relu(conv2d(h_pool1_identitynet, W_conv2_identitynet) + b_conv2_identitynet)
    # h_pool2: 28*28*64
    h_pool2_identitynet = max_pool_2x2(h_conv2_identitynet)
    # h_conv3: 28*28*128
    h_conv3_identitynet = tf.nn.relu(conv2d(h_pool2_identitynet, W_conv3_identitynet) + b_conv3_identitynet)
    # h_pool3: 14*14*128
    h_pool3_identitynet = max_pool_2x2(h_conv3_identitynet)
    # h_pool3_flat: 14*14*128=25088
    h_pool3_flat_identitynet = tf.reshape(h_pool3_identitynet, [-1, 14*14*128])
    # h_fc1: 2048
    h_fc1_identitynet = tf.nn.relu(tf.matmul(h_pool3_flat_identitynet, W_fc1_identitynet) + b_fc1_identitynet)
    # h_fc2: 1024
    h_fc2_identitynet = tf.nn.sigmoid(tf.matmul(h_fc1_identitynet, W_fc2_identitynet) + b_fc2_identitynet)
    
    return h_fc2_identitynet


def PoseNet(x):
    # h_conv1: 112*112*32
    h_conv1_posenet = tf.nn.relu(conv2d(x, W_conv1_posenet) + b_conv1_posenet)
    # h_pool1: 56*56*32
    h_pool1_posenet = max_pool_2x2(h_conv1_posenet)
    # h_conv2: 56*56*64
    h_conv2_posenet = tf.nn.relu(conv2d(h_pool1_posenet, W_conv2_posenet) + b_conv2_posenet)
    # h_pool2: 28*28*64
    h_pool2_posenet = max_pool_2x2(h_conv2_posenet)
    # h_conv3: 28*28*128
    h_conv3_posenet = tf.nn.relu(conv2d(h_pool2_posenet, W_conv3_posenet) + b_conv3_posenet)
    # h_pool3: 14*14*128
    h_pool3_posenet = max_pool_2x2(h_conv3_posenet)
    # h_pool3_flat: 14*14*128=25088
    h_pool3_flat_posenet = tf.reshape(h_pool3_posenet, [-1, 14*14*128])
    # h_fc1: 2048
    h_fc1_posenet = tf.nn.relu(tf.matmul(h_pool3_flat_posenet, W_fc1_posenet) + b_fc1_posenet)
    # h_fc2: 1024
    h_fc2_posenet = tf.nn.sigmoid(tf.matmul(h_fc1_posenet, W_fc2_posenet) + b_fc2_posenet)
    
    return h_fc2_posenet

def Discriminator(x):
    # h_fc1_discriminator: 2048
    h_fc1_discriminator = tf.nn.relu(tf.matmul(x, W_fc1_discriminator) + b_fc1_discriminator)
    # h_fc2_discriminator: 1024
    h_fc2_discriminator = tf.nn.relu(tf.matmul(h_fc1_discriminator, W_fc2_discriminator) + b_fc2_discriminator)
    # h_fc2_discriminator: 10
    h_fc3_discriminator = tf.nn.softmax(tf.matmul(h_fc2_discriminator, W_fc3_discriminator) + b_fc3_discriminator)
    
    return h_fc3_discriminator

def MixNet(x):
    # h_fc1_mix: 2048
    h_fc1_mix = tf.nn.relu(tf.matmul(x, W_fc1_mix) + b_fc1_mix)
    # h_fc2_mix: 1024
    h_fc2_mix = tf.nn.sigmoid(tf.matmul(h_fc1_mix, W_fc2_mix) + b_fc2_mix)

    return h_fc2_mix

def DecoderNet(x):
    h_fc1_decoder = tf.nn.relu(tf.matmul(x, W_fc1_decoder) + b_fc1_decoder)
    h_flat_decoder = tf.reshape(h_fc1_decoder, [-1, 14,14,128])
    h_deconv1_decoder = tf.nn.relu(deconv2d_2x2(h_flat_decoder, W_deconv1_decoder,[batch_size, 28,28,64]) + b_deconv1_decoder)
    h_deconv2_decoder = tf.nn.relu(deconv2d_2x2(h_deconv1_decoder, W_deconv2_decoder,[batch_size, 56,56,32]) + b_deconv2_decoder)
    h_deconv3_decoder = tf.nn.relu(deconv2d_2x2(h_deconv2_decoder, W_deconv3_decoder,[batch_size, 112,112,16]) + b_deconv3_decoder)
    h_deconv4_decoder = tf.nn.sigmoid(deconv2d_1x1(h_deconv3_decoder, W_deconv4_decoder,[batch_size, 112,112,1]) + b_deconv4_decoder)

    return h_deconv4_decoder

def MetricNet(x):
    # h_fc1_metricnet: 512
    h_fc1_metricnet = tf.nn.relu(tf.matmul(x, W_fc1_metricnet) + b_fc1_metricnet)
    # h_fc2_metricnet: 128
    h_fc2_metricnet = tf.nn.relu(tf.matmul(h_fc1_metricnet, W_fc2_metricnet) + b_fc2_metricnet)
    # h_fc2_metricnet: 1
    h_fc3_metricnet = tf.nn.sigmoid(tf.matmul(h_fc2_metricnet, W_fc3_metricnet) + b_fc3_metricnet)
    
    return h_fc3_metricnet


identity1=IdentityNet(sample1)
identity2=IdentityNet(sample2)

pose1=PoseNet(sample1)
pose2=PoseNet(sample2)


sample1_out=Discriminator(identity1)
sample2_out=Discriminator(identity2)



mix1=tf.concat([identity1,pose2],1)
mix2=tf.concat([identity2,pose1],1)

coding1=MixNet(mix1)
coding2=MixNet(mix2)

pred_sample1=DecoderNet(coding1)
pred_sample2=DecoderNet(coding2)

metric_in=tf.square(identity1-identity2)
dis=MetricNet(metric_in)
#------------------------------------------------------------------------------
# Loss of Discriminator
loss1= -tf.reduce_sum(y_sample1*tf.log(tf.maximum(sample1_out,1e-10))+(1-y_sample1)*tf.log(tf.maximum(1-sample1_out,1e-10)))
loss2= -tf.reduce_sum(y_sample2*tf.log(tf.maximum(sample2_out,1e-10))+(1-y_sample2)*tf.log(tf.maximum(1-sample2_out,1e-10)))
#loss1=tf.reduce_mean(tf.square(y_sample1-sample1_out))
#loss2=tf.reduce_mean(tf.square(y_sample2-sample2_out))
# Loss of Decoder
#loss_pred1=tf.reduce_mean(tf.square(pred_sample1-res_sample1))
#loss_pred2=tf.reduce_mean(tf.square(pred_sample2-res_sample2))
loss_pred1=-tf.reduce_sum(res_sample1*tf.log(tf.maximum(pred_sample1,1e-10))+(1-res_sample1)*tf.log(tf.maximum(1-pred_sample1,1e-10)))
loss_pred2=-tf.reduce_sum(res_sample2*tf.log(tf.maximum(pred_sample2,1e-10))+(1-res_sample2)*tf.log(tf.maximum(1-pred_sample2,1e-10)))
# Loss of Metric
loss_dis=-tf.reduce_sum(y_dis*tf.log(tf.maximum(dis,1e-10)))

#loss_dis=tf.reduce_mean(tf.square(y_dis-dis))
# Loss of Common
loss_common=-tf.reduce_sum(pose1*tf.log(tf.maximum(pose2,1e-10)))
#loss_common=tf.reduce_mean(tf.square(common1-common2))
# Overall loss
coef1=1
coef2=1
coef3=0.6
coef4=0.6
coef5=0.6
coef6=0.6


 
loss=coef1*loss1+coef2*loss2+coef3*loss_pred1+coef4*loss_pred2+coef5*loss_dis+coef6*loss_common

lr=1e-6
train= tf.train.AdamOptimizer(lr).minimize(loss)

sess.run(tf.global_variables_initializer())

flag_pretrain=True
flag_results=False
saver = tf.train.Saver()
if os.path.exists('ckpt'):
    if os.path.exists('ckpt_2'):
        model_file=tf.train.latest_checkpoint('ckpt_2/')
        saver.restore(sess,model_file)
        print('loading ckpt_2 success...')
        
    else:
        model_file=tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess,model_file)
        print('loading ckpt success...')
    
    flag_pretrain=False

train_class=20 
train_list=list(range(0,20))

#------------------------------------------------------------------------------
# data generalization
if flag_pretrain:
    sample_input1=np.zeros([0,112,112,1],np.float32)
    sample_input2=np.zeros([0,112,112,1],np.float32)
    sample_input1_y=np.zeros([0,10],np.float32)
    sample_input2_y=np.zeros([0,10],np.float32)
    sample_metric=np.zeros([0,1],np.float32)
    dis_overall=[]
   
    random.shuffle(train_list)
    dis_epoch=[]
    for a in train_list[0:train_class]:
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
                        
                        seq_coding1=getCoding.getCoding(sess_auto, sample, coding, pred_sample, sequence_images1, batch_size=256)
                        seq_coding2=getCoding.getCoding(sess_auto, sample, coding, pred_sample, sequence_images2, batch_size=256)
            
                        min_dis, min_path=dtw_tool.DTW_tool(seq_coding1,seq_coding2)
                        
                        
                        
                        input1,input2=mf.getSamples(sequence_images1,sequence_images2,min_path)
                        
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
                
                        dis_epoch.append(min_dis)
    
    dis_overall.append(np.mean(dis_epoch))
    
    N_t=sample_input1.shape[0]
    epochs=np.int(np.floor(N_t/batch_size)+1) 
    
    if flag_results:   
        loss_overall_b=np.load('results/loss_overall_b_pre.npy') 
        loss_overall_c=np.load('results/loss_overall_c_pre.npy') 
        loss_overall_d=np.load('results/loss_overall_d_pre.npy') 
        loss_overall_e=np.load('results/loss_overall_e_pre.npy') 
        loss_overall_f=np.load('results/loss_overall_f_pre.npy') 
        loss_overall_g=np.load('results/loss_overall_g_pre.npy') 
        loss_overall_h=np.load('results/loss_overall_h_pre.npy') 
        cr_overall=np.load('results/cr_overall_pre.npy') 
        c_overall=np.load('results/c_overall_pre.npy') 
    else:
        loss_overall_b=[]
        loss_overall_c=[]
        loss_overall_d=[]
        loss_overall_e=[]
        loss_overall_f=[]
        loss_overall_g=[]
        loss_overall_h=[]
        cr_overall=[]
        c_overall=[]
    
    
    
    for repeat in range(0,10):
        
        order_shuffled=random.sample(range(0,N_t),N_t)
        sample_input1=sample_input1[order_shuffled,:,:,:]
        sample_input2=sample_input2[order_shuffled,:,:,:]
        sample_input1_y=sample_input1_y[order_shuffled,:]
        sample_input2_y=sample_input2_y[order_shuffled,:]
        sample_metric=sample_metric[order_shuffled,:]
        
        
        loss_b=[]
        loss_c=[]
        loss_d=[]
        loss_e=[]
        loss_f=[]
        loss_g=[]
        loss_h=[]
        
        for i in range(0,epochs):
            
            if (i+1)*batch_size<N_t:
                sample_input1_seg=sample_input1[i*batch_size:(i+1)*batch_size,:,:,:]
                sample_input2_seg=sample_input2[i*batch_size:(i+1)*batch_size,:,:,:]
                  
                t_label_one_hot_seg=sample_input1_y[i*batch_size:(i+1)*batch_size,:]
                s_label_one_hot_seg=sample_input2_y[i*batch_size:(i+1)*batch_size,:]
                
                similar_seg=sample_metric[i*batch_size:(i+1)*batch_size,:]
            
                _,b,c,d,e,f,g,h=sess.run([train,loss,loss1,loss2,loss_pred1,loss_pred2,loss_dis,loss_common],feed_dict={sample1: sample_input1_seg, sample2: sample_input2_seg, y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:sample_input1_seg, res_sample2:sample_input2_seg, y_dis:similar_seg})
                loss_b.append(b)
                loss_c.append(c)
                loss_d.append(d)
                loss_e.append(e)
                loss_f.append(f)
                loss_g.append(g)
                loss_h.append(h)
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
    
        loss_overall_b.append(np.mean(loss_b))
        loss_overall_c.append(np.mean(loss_c))
        loss_overall_d.append(np.mean(loss_d))
        loss_overall_e.append(np.mean(loss_e))
        loss_overall_f.append(np.mean(loss_f))
        loss_overall_g.append(np.mean(loss_g))
        loss_overall_h.append(np.mean(loss_h))
        
    #        print('Repeat:{},Epoch:{},loss:{}, loss1:{}, loss2:{}, loss_pred1:{}, loss_pred2:{}, loss_dis:{}, loss_common:{}'.format(repeat, i, loss_overall[-1][1],loss_overall[-1][2],loss_overall[-1][3],loss_overall[-1][4],loss_overall[-1][5],loss_overall[-1][6],loss_overall[-1][7]))
    #    
        
        
    #    l=(sess.run([sample1_out],feed_dict={sample1: x_test}))[0]
    #    pred_label=[]
    #    for i in range(0,l.shape[0]):
    #        pred_label.append((l[i,:]==(np.max(l,1))[i]).tolist())
    #    
    #    pred_label=np.array(pred_label)
    #    cr=1-len((np.nonzero(np.sum(np.abs(test_label_one_hot-pred_label),1)))[0])/x_test.shape[0]
        
        N_test=x_test.shape[0]
        err=0
        
        pred_test=np.zeros([0,10])
        if np.mod(N_test,batch_size)==0:
            for i in range(0,N_test/batch_size):
                l=sess.run([sample1_out],feed_dict={sample1: x_test[i*batch_size:(i+1)*batch_size,:,:,:]})[0]
                pred_label=[]
                for j in range(0,l.shape[0]):
                    pred_label.append((l[j,:]==(np.max(l,1))[j]).tolist())
                pred_label=np.array(pred_label)
                err_batch=len(np.nonzero(np.sum(np.abs(test_label_one_hot[i*batch_size:(i+1)*batch_size,:]-pred_label),1))[0])
                err=err+err_batch
                pred_test=np.vstack((pred_test,pred_label))
        else:
            for i in range(0,np.int(N_test/batch_size)+1):
                if (i+1)*batch_size<N_test:
                    l=sess.run([sample1_out],feed_dict={sample1: x_test[i*batch_size:(i+1)*batch_size,:,:,:]})[0]
                    pred_label=[]
                    for j in range(0,l.shape[0]):
                        pred_label.append((l[j,:]==(np.max(l,1))[j]).tolist())
                    pred_label=np.array(pred_label)
                    err_batch=len(np.nonzero(np.sum(np.abs(test_label_one_hot[i*batch_size:(i+1)*batch_size,:]-pred_label),1))[0])
                    err=err+err_batch
                    pred_test=np.vstack((pred_test,pred_label))
                else:
                    l=sess.run([sample1_out],feed_dict={sample1: x_test[(N_test-batch_size):,:,:,:]})[0]
                    pred_label=[]
                    for j in range(0,l.shape[0]):
                        pred_label.append((l[j,:]==(np.max(l,1))[j]).tolist())
                    pred_label=np.array(pred_label)
                    err_batch=len(np.nonzero(np.sum(np.abs(test_label_one_hot[i*batch_size:,:]-pred_label[(batch_size-N_test+i*batch_size):,:]),1))[0])
                    err=err+err_batch  
                    pred_test=np.vstack((pred_test,pred_label))
        cr=1-err/N_test   
        cr_overall.append(cr)
        # cr is recognition rate for sequence, c is recognition rate for pose
        c=0
        c_sum=0
        test_set2=np.vstack((test_set,np.zeros([1,6],dtype='int32')))
        score=np.zeros([10,])
        for s in range(0,test_set2.shape[0]-1):
            
            score=score+pred_test[s,:]
            # test_set[i+1,1:4]-test_set[i,1:4] 保证 a，s，r不变，
            if np.sum(np.nonzero(test_set2[s+1,1:4]-test_set2[s,1:4]))!=0:
                label=test_set2[s,2]
                pre_label=np.nonzero(score==np.max(score))[0]
                if (label-pre_label==0)[0]:
                    c=c+1
                score=np.zeros([10,])   
                c_sum=c_sum+1
        c=c/c_sum
        c_overall.append(c)
        print('Repeat1:{},cr:{},c:{},loss:{},loss1:{},loss2:{},loss_pred1:{},loss_pred2:{},loss_dis:{},loss_common:{}'.format(repeat+1,cr,c,loss_overall_b[-1],loss_overall_c[-1],loss_overall_d[-1],loss_overall_e[-1],loss_overall_f[-1],loss_overall_g[-1],loss_overall_h[-1])) 
        
        
    np.save('results/cr_overall_pre.npy',cr_overall)    
    np.save('results/c_overall_pre.npy',c_overall)
    np.save('results/loss_overall_b_pre.npy',loss_overall_b)
    np.save('results/loss_overall_c_pre.npy',loss_overall_c)  
    np.save('results/loss_overall_d_pre.npy',loss_overall_d)  
    np.save('results/loss_overall_e_pre.npy',loss_overall_e)  
    np.save('results/loss_overall_f_pre.npy',loss_overall_f)  
    np.save('results/loss_overall_g_pre.npy',loss_overall_g)  
    np.save('results/loss_overall_h_pre.npy',loss_overall_h)  
    
    np.save('results/dis_overall.npy',dis_overall) 
    saver=tf.train.Saver(max_to_keep=1)
    saver.save(sess,'ckpt/mymodel.ckpt')

 

   
  #-----------------------
if flag_results:   
    loss_overall_b=np.load('results/loss_overall_b.npy') 
    loss_overall_c=np.load('results/loss_overall_c.npy') 
    loss_overall_d=np.load('results/loss_overall_d.npy') 
    loss_overall_e=np.load('results/loss_overall_e.npy') 
    loss_overall_f=np.load('results/loss_overall_f.npy') 
    loss_overall_g=np.load('results/loss_overall_g.npy') 
    loss_overall_h=np.load('results/loss_overall_h.npy') 
    cr_overall=np.load('results/cr_overall_pre.npy') 
    c_overall=np.load('results/c_overall_pre.npy')

else:
    loss_overall_b=[]
    loss_overall_c=[]
    loss_overall_d=[]
    loss_overall_e=[]
    loss_overall_f=[]
    loss_overall_g=[]
    loss_overall_h=[]
    cr_overall=[]
    c_overall=[]
    
dis_overall=np.load('results/dis_overall.npy')


for repeat in range(0,1000):
    
    sample_input1=np.zeros([0,112,112,1],np.float32)
    sample_input2=np.zeros([0,112,112,1],np.float32)
    sample_input1_y=np.zeros([0,10],np.float32)
    sample_input2_y=np.zeros([0,10],np.float32)
    sample_metric=np.zeros([0,1],np.float32)
    dis_epoch=[]
    random.shuffle(train_list)
    for a in train_list[0:train_class]:
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
                        
                        c1=np.zeros([sequence_images1.shape[0],1024],np.float32)
                        if sequence_images1.shape[0]>batch_size:
                            for k in range(0,np.int(sequence_images1.shape[0]/batch_size)+1):
                                if (k+1)*batch_size<sequence_images1.shape[0]:
                                    c1[k*batch_size:(k+1)*batch_size,:]=sess.run([pose1],feed_dict={sample1: sequence_images1[k*batch_size:(k+1)*batch_size,:,:,:]})[0]
                                else:
                                    template1=np.ones([batch_size,112,112,1],np.float32)
                                    template1[0:(sequence_images1.shape[0]-k*batch_size),:,:,:]=sequence_images1[k*batch_size:,:,:,:]
                                    c_temp=sess.run([pose1],feed_dict={sample1: template1})[0]
                                    c1[k*batch_size:,:]=c_temp[0:(sequence_images1.shape[0]-k*batch_size),:]
                        else:
                            template1=np.ones([batch_size,112,112,1],np.float32)
                            template1[0:sequence_images1.shape[0],:,:,:]=sequence_images1
                            template1_temp=sess.run([pose1],feed_dict={sample1: template1})[0]
                            c1[0:sequence_images1.shape[0],:]=template1_temp[0:sequence_images1.shape[0],:]
                            
                        c2=np.zeros([sequence_images2.shape[0],1024],np.float32)
                        if sequence_images2.shape[0]>batch_size:
                            
                            for k in range(0,np.int(sequence_images2.shape[0]/batch_size)+1):
                                if (k+1)*batch_size<sequence_images2.shape[0]:
                                    c2[k*batch_size:(k+1)*batch_size,:]=sess.run([pose2],feed_dict={sample2: sequence_images2[k*batch_size:(k+1)*batch_size,:,:,:]})[0]
                                else:
                                    template2=np.ones([batch_size,112,112,1],np.float32)
                                    template2[0:(sequence_images2.shape[0]-k*batch_size),:,:,:]=sequence_images2[k*batch_size:,:,:,:]
                                    c_temp=sess.run([pose2],feed_dict={sample2: template2})[0]
                                    c2[k*batch_size:,:]=c_temp[0:(sequence_images2.shape[0]-k*batch_size),:]
                        else:
                            template2=np.ones([batch_size,112,112,1],np.float32)
                            template2[0:sequence_images2.shape[0],:,:,:]=sequence_images2
                            template2_temp=sess.run([pose2],feed_dict={sample2: template2})[0]
                            c2[0:sequence_images2.shape[0],:]=template2_temp[0:sequence_images2.shape[0],:]
                            
                            

                        
                        
                             
                        min_dis, min_path=dtw_tool.DTW_tool(c1,c2)
                        
                        input1,input2=mf.getSamples(sequence_images1,sequence_images2,min_path)
                        
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
                        dis_epoch.append(min_dis)
                        
    dis_overall=list(dis_overall)
    dis_overall.append(np.mean(dis_epoch))              
    N_t=sample_input1.shape[0]
    epochs=np.int(np.floor(N_t/batch_size)+1) 
    for repeat2 in range(0,5):
        order_shuffled=random.sample(range(0,N_t),N_t)
        sample_input1=sample_input1[order_shuffled,:,:,:]
        sample_input2=sample_input2[order_shuffled,:,:,:]
        sample_input1_y=sample_input1_y[order_shuffled,:]
        sample_input2_y=sample_input2_y[order_shuffled,:]
        sample_metric=sample_metric[order_shuffled,:]
    
        
        
        loss_b=[]
        loss_c=[]
        loss_d=[]
        loss_e=[]
        loss_f=[]
        loss_g=[]
        loss_h=[]
        for i in range(0,epochs):
    
            if (i+1)*batch_size<N_t:
                sample_input1_seg=sample_input1[i*batch_size:(i+1)*batch_size,:,:,:]
                sample_input2_seg=sample_input2[i*batch_size:(i+1)*batch_size,:,:,:]
                  
                t_label_one_hot_seg=sample_input1_y[i*batch_size:(i+1)*batch_size,:]
                s_label_one_hot_seg=sample_input2_y[i*batch_size:(i+1)*batch_size,:]
                
                similar_seg=sample_metric[i*batch_size:(i+1)*batch_size,:]
            
                _,b,c,d,e,f,g,h=sess.run([train,loss,loss1,loss2,loss_pred1,loss_pred2,loss_dis,loss_common],feed_dict={sample1: sample_input1_seg, sample2: sample_input2_seg, y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:sample_input1_seg, res_sample2:sample_input2_seg, y_dis:similar_seg})
                loss_b.append(b)
                loss_c.append(c)
                loss_d.append(d)
                loss_e.append(e)
                loss_f.append(f)
                loss_g.append(g)
                loss_h.append(h)

        loss_overall_b.append(np.mean(loss_b))
        loss_overall_c.append(np.mean(loss_c))
        loss_overall_d.append(np.mean(loss_d))
        loss_overall_e.append(np.mean(loss_e))
        loss_overall_f.append(np.mean(loss_f))
        loss_overall_g.append(np.mean(loss_g))
        loss_overall_h.append(np.mean(loss_h))

        
        
        N_test=x_test.shape[0]
        err=0
        pred_test=np.zeros([0,10])
        if np.mod(N_test,batch_size)==0:
            for i in range(0,N_test/batch_size):
                l=sess.run([sample1_out],feed_dict={sample1: x_test[i*batch_size:(i+1)*batch_size,:,:,:]})[0]
                pred_label=[]
                for j in range(0,l.shape[0]):
                    pred_label.append((l[j,:]==(np.max(l,1))[j]).tolist())
                pred_label=np.array(pred_label)
                err_batch=len(np.nonzero(np.sum(np.abs(test_label_one_hot[i*batch_size:(i+1)*batch_size,:]-pred_label),1))[0])
                err=err+err_batch
                pred_test=np.vstack((pred_test,pred_label))
        else:
            for i in range(0,np.int(N_test/batch_size)+1):
                if (i+1)*batch_size<N_test:
                    l=sess.run([sample1_out],feed_dict={sample1: x_test[i*batch_size:(i+1)*batch_size,:,:,:]})[0]
                    pred_label=[]
                    for j in range(0,l.shape[0]):
                        pred_label.append((l[j,:]==(np.max(l,1))[j]).tolist())
                    pred_label=np.array(pred_label)
                    err_batch=len(np.nonzero(np.sum(np.abs(test_label_one_hot[i*batch_size:(i+1)*batch_size,:]-pred_label),1))[0])
                    err=err+err_batch
                    pred_test=np.vstack((pred_test,pred_label))
                else:
                    l=sess.run([sample1_out],feed_dict={sample1: x_test[(N_test-batch_size):,:,:,:]})[0]
                    pred_label=[]
                    for j in range(0,l.shape[0]):
                        pred_label.append((l[j,:]==(np.max(l,1))[j]).tolist())
                    pred_label=np.array(pred_label)
                    err_batch=len(np.nonzero(np.sum(np.abs(test_label_one_hot[i*batch_size:,:]-pred_label[(batch_size-N_test+i*batch_size):,:]),1))[0])
                    err=err+err_batch  
                    pred_test=np.vstack((pred_test,pred_label))
        cr=1-err/N_test   
        cr_overall.append(cr)
        c=0
        c_sum=0
        test_set2=np.vstack((test_set,np.zeros([1,6],dtype='int32')))
        score=np.zeros([10,])
        for s in range(0,test_set2.shape[0]-1):
            
            score=score+pred_test[s,:]
            # test_set[i+1,1:4]-test_set[i,1:4] 保证 a，s，r不变，
            if np.sum(np.nonzero(test_set2[s+1,1:4]-test_set2[s,1:4]))!=0:
                label=test_set2[s,2]
                pre_label=np.nonzero(score==np.max(score))[0]
                if (label-pre_label==0)[0]:
                    c=c+1
                score=np.zeros([10,])   
                c_sum=c_sum+1
        c=c/c_sum
        c_overall.append(c)
        
#        l=(sess.run([sample1_out],feed_dict={sample1: test_coding}))[0]
#        
#        for i in range(0,l.shape[0]):
#            pred_label.append((l[i,:]==(np.max(l,1))[i]).tolist())
#        
#        pred_label=np.array(pred_label)
#        cr=1-len((np.nonzero(np.sum(np.abs(test_label_one_hot-pred_label),1)))[0])/x_test.shape[0]
        
        
        print('Repeat1:{},Repeat2:{},cr:{},c:{},loss:{},loss1:{},loss2:{},loss_pred1:{},loss_pred2:{},loss_dis:{},loss_common:{}'.format(repeat+1,repeat2+1,cr,c,loss_overall_b[-1],loss_overall_c[-1],loss_overall_d[-1],loss_overall_e[-1],loss_overall_f[-1],loss_overall_g[-1],loss_overall_h[-1])) 
        np.save('results/cr_overall.npy',cr_overall)    
        np.save('results/c_overall.npy',c_overall)
        np.save('results/loss_overall_b.npy',loss_overall_b)
        np.save('results/loss_overall_c.npy',loss_overall_c)  
        np.save('results/loss_overall_d.npy',loss_overall_d)  
        np.save('results/loss_overall_e.npy',loss_overall_e)  
        np.save('results/loss_overall_f.npy',loss_overall_f)  
        np.save('results/loss_overall_g.npy',loss_overall_g)  
        np.save('results/loss_overall_h.npy',loss_overall_h)  
        np.save('results/dis_overall.npy',dis_overall) 
        saver=tf.train.Saver(max_to_keep=1)
        saver.save(sess,'ckpt_2/mymodel.ckpt') 
    

    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    