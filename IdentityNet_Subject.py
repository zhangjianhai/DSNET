 # -*- coding: utf-8 -*-

from keras.utils import np_utils



import utilize_functions as uf
import tensorflow as tf


import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
import os


ske=scipy.io.loadmat('data/ske.mat')
ske=ske['ske']

ske_v=scipy.io.loadmat('data/skeletal_data_validity.mat')
ske_v=ske_v['skeletal_data_validity']


sess = tf.InteractiveSession()
batch_size=128

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





  

sample1=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
sample2=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])

y_sample1 = tf.placeholder(dtype=tf.float32, shape=[None, 10])
y_sample2 = tf.placeholder(dtype=tf.float32, shape=[None, 10])

res_sample1=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
res_sample2=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])

y_dis=tf.placeholder(dtype=tf.float32, shape=[None, 1])

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
    
    
# -------------------------------------------------

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

loss1= -tf.reduce_sum(y_sample1*tf.log(tf.maximum(sample1_out,1e-10)))
loss2= -tf.reduce_sum(y_sample2*tf.log(tf.maximum(sample2_out,1e-10)))

loss_pred1=-tf.reduce_sum(res_sample1*tf.log(tf.maximum(pred_sample1,1e-10)))
loss_pred2=-tf.reduce_sum(res_sample2*tf.log(tf.maximum(pred_sample2,1e-10)))

loss_dis=-tf.reduce_sum(y_dis*tf.log(tf.maximum(dis,1e-10)))

loss=loss1+loss2+loss_pred1+loss_pred2+loss_dis

lr=1e-6
train= tf.train.AdamOptimizer(lr).minimize(loss)

sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
if os.path.exists('ckpt'):
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,model_file)



# t为设置的局部窗口大小
t=6
# train_set, test_set 为训练集和测试集，为索引

train_set,test_set=uf.process_data(ske, ske_v, t)

# 生成测试样本
action_seg=uf.getS(t,test_set,ske)
x_test=uf.getImages(action_seg,t)
x_test=x_test[:,:,:,np.newaxis].astype('float32')
test_label=test_set[:,2]
test_label_one_hot=np_utils.to_categorical(test_label, num_classes=10)


# 生成训练样本
action_seg=uf.getS(t,train_set,ske)
x_train=uf.getImages(action_seg,t)
x_train=x_train[:,:,:,np.newaxis].astype('float32')
train_label=train_set[:,2]
train_label_one_hot=np_utils.to_categorical(train_label, num_classes=10)



# Pre-train the whole net





        # 第一个输入训练集的最大个数







if os.path.exists('para/pre_loss.npy'):
    pre_loss = np.load('para/pre_loss.npy')
    pre_loss=pre_loss.tolist()
else:
    pre_loss=[]
pre_flag=True
if pre_flag:
    
    for pre in range(0,50):
    
        
        N_t=train_set.shape[0]
                # 生成对应的第一个输出的标签
        t_label=train_set[:,2]
                # 重新排列这些样本
        t_index=train_set
                # 得到图像表示
        action_seg=uf.getS(t,t_index,ske)
        x_t=uf.getImages(action_seg,t)
        x_t=x_t[:,:,:,np.newaxis].astype('float32')
        
                #---------------------------
        order_shuffled=random.sample(range(0,N_t),N_t)
        
        
        s_index=train_set[order_shuffled,:]
                # 生成对应的第二个输出的标签
        s_label=train_set[:,2]
        s_label=s_label[order_shuffled]
                # 得到图像表示
        action_seg=uf.getS(t,s_index,ske)
        x_s=uf.getImages(action_seg,t)
        x_s=x_s[:,:,:,np.newaxis].astype('float32')
        
        
        x_t_temp=np.zeros([0,112,112,1])
        x_s_temp=np.zeros([0,112,112,1])
        t_label_temp=np.zeros([0,])
        s_label_temp=np.zeros([0,])
        x_t_res_temp=np.zeros([0,112,112,1])
        x_s_res_temp=np.zeros([0,112,112,1])
        similar_temp=np.zeros([0,1],np.int32)
        
        for i in range(0,10):
            order_shuffled=random.sample(range(0,N_t),N_t)
            s_label=t_label[order_shuffled]
            x_s=x_t[order_shuffled,:,:,:]
        
            similar=(t_label-s_label)==0
            same_index=np.nonzero(similar)[0]
            x_t_temp=np.vstack((x_t_temp,x_t[same_index,:,:,:]))
            x_s_temp=np.vstack((x_s_temp,x_s[same_index,:,:,:]))
            t_label_temp=np.hstack((t_label_temp,np.int32(t_label[same_index])))
            s_label_temp=np.hstack((s_label_temp,np.int32(s_label[same_index])))
            x_t_res_temp=np.vstack((x_t_res_temp,x_s[same_index,:,:,:]))
            x_s_res_temp=np.vstack((x_s_res_temp,x_t[same_index,:,:,:]))
            similar_temp=np.vstack((similar_temp,np.int32(np.ones([same_index.shape[0],1]))))
        
        x_t=x_t_temp
        x_s=x_s_temp
        t_label=t_label_temp
        s_label=s_label_temp
        x_t_res=x_t_res_temp
        x_s_res=x_s_res_temp
        similar=similar_temp
        
        
        N_t=x_t.shape[0]
        order_shuffled=random.sample(range(0,N_t),N_t)
        x_t=x_t[order_shuffled,:,:,:]
        x_s=x_s[order_shuffled,:,:,:]
        t_label=t_label[order_shuffled]
        s_label=s_label[order_shuffled]
        x_t_res=x_t_res[order_shuffled,:,:,:]
        x_s_res=x_s_res[order_shuffled,:,:,:]
        
        
        
                #------------------------
                # 生成标签的one_hot模式
        t_label_one_hot=np_utils.to_categorical(t_label, num_classes=10)
        s_label_one_hot=np_utils.to_categorical(s_label, num_classes=10)
        
        epochs=np.int(np.floor(N_t/batch_size)+1) 
            
        loss_batch=[]
        for i in range(0,epochs):
            
            if (i+1)*batch_size<N_t:
            
                x_t_seg=x_t[i*batch_size:(i+1)*batch_size,:,:,:]
                x_s_seg=x_s[i*batch_size:(i+1)*batch_size,:,:,:]
                
                x_t_res_seg=x_t[i*batch_size:(i+1)*batch_size,:,:,:]
                x_s_res_seg=x_s[i*batch_size:(i+1)*batch_size,:,:,:]  
                    
                t_label_one_hot_seg=t_label_one_hot[i*batch_size:(i+1)*batch_size,:]
                s_label_one_hot_seg=s_label_one_hot[i*batch_size:(i+1)*batch_size,:]
                
                similar_seg=similar[i*batch_size:(i+1)*batch_size,:]
            
                _,b=sess.run([train,loss],feed_dict={sample1: x_t_seg, sample2: x_s_seg,y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:x_t_res_seg, res_sample2:x_s_res_seg, y_dis:similar_seg})
            
                
                loss_batch.append(b)
        
            else:
                x_t_seg=x_t[N_t-batch_size:N_t,:,:,:]
                x_s_seg=x_s[N_t-batch_size:N_t,:,:,:]
                
                x_t_res_seg=x_t[N_t-batch_size:N_t,:,:,:]
                x_s_res_seg=x_s[N_t-batch_size:N_t,:,:,:]   
                
                
                
                t_label_one_hot_seg=t_label_one_hot[N_t-batch_size:N_t,:]
                s_label_one_hot_seg=s_label_one_hot[N_t-batch_size:N_t,:]
            
                similar_seg=similar[N_t-batch_size:N_t,:]
            
                _,b=sess.run([train,loss],feed_dict={sample1: x_t_seg, sample2: x_s_seg,y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:x_t_res_seg, res_sample2:x_s_res_seg,y_dis:similar_seg})
          
                loss_batch.append(b)
                
            pre_loss.append(np.sum(loss_batch))
            
        print('Pre-train Epoch:{}, pre_loss:{}'.format(pre+1,pre_loss[-1]))
            




if os.path.exists('para/loss_show.npy'):
    loss_show = np.load('para/loss_show.npy')
    loss_show=loss_show.tolist()
else:
    loss_show=[]
    
if os.path.exists('para/cr_show.npy'):
    cr_show = np.load('para/cr_show.npy')
    cr_show=cr_show.tolist()
else:
    cr_show=[] 
    
if os.path.exists('para/cr_show_seq.npy'):
    cr_show_seq = np.load('para/cr_show_seq.npy')
    cr_show_seq=cr_show_seq.tolist()
else:
    cr_show_seq=[]

if os.path.exists('para/cr_show_seq_saliency.npy'):
    cr_show_seq_saliency = np.load('para/cr_show_seq_saliency.npy')
    cr_show_seq_saliency=cr_show_seq_saliency.tolist()
else:
    cr_show_seq_saliency=[]

if os.path.exists('para/epoch.npy'):
    epoch = np.load('para/epoch.npy')
    e=np.int(epoch)
else:
    e=0
    
for epoch in range(e,100000):
    
    N_t=train_set.shape[0]
            # 生成对应的第一个输出的标签
    t_label=train_set[:,2]
            # 重新排列这些样本
    t_index=train_set
            # 得到图像表示
    action_seg=uf.getS(t,t_index,ske)
    x_t=uf.getImages(action_seg,t)
    x_t=x_t[:,:,:,np.newaxis].astype('float32')
    
            #---------------------------
    order_shuffled=random.sample(range(0,N_t),N_t)
    
    
    s_index=train_set[order_shuffled,:]
            # 生成对应的第二个输出的标签
    s_label=train_set[:,2]
    s_label=s_label[order_shuffled]
            # 得到图像表示
    action_seg=uf.getS(t,s_index,ske)
    x_s=uf.getImages(action_seg,t)
    x_s=x_s[:,:,:,np.newaxis].astype('float32')
    
    
    x_t_temp=np.zeros([0,112,112,1])
    x_s_temp=np.zeros([0,112,112,1])
    t_label_temp=np.zeros([0,])
    s_label_temp=np.zeros([0,])
    x_t_res_temp=np.zeros([0,112,112,1])
    x_s_res_temp=np.zeros([0,112,112,1])
    similar_temp=np.zeros([0,1],np.int32)
    
    for i in range(0,10):
        order_shuffled=random.sample(range(0,N_t),N_t)
        s_label=t_label[order_shuffled]
        x_s=x_t[order_shuffled,:,:,:]
    
        similar=(t_label-s_label)==0
        same_index=np.nonzero(similar)[0]
        x_t_temp=np.vstack((x_t_temp,x_t[same_index,:,:,:]))
        x_s_temp=np.vstack((x_s_temp,x_s[same_index,:,:,:]))
        t_label_temp=np.hstack((t_label_temp,np.int32(t_label[same_index])))
        s_label_temp=np.hstack((s_label_temp,np.int32(s_label[same_index])))
        # 来自同一个人的处理办法
        x_t_res_temp=np.vstack((x_t_res_temp,x_s[same_index,:,:,:]))
        x_s_res_temp=np.vstack((x_s_res_temp,x_t[same_index,:,:,:]))
        similar_temp=np.vstack((similar_temp,np.int32(np.ones([same_index.shape[0],1]))))





    order_shuffled=random.sample(range(0,N_t),N_t)
    s_label=t_label[order_shuffled]
    x_s=x_t[order_shuffled,:,:,:]
    # 来自不同的人 
    dissimilar=(t_label-s_label)!=0
    dissimilar_index=np.nonzero(dissimilar)[0]
    
    
    pose_t=np.zeros([N_t,1024])
    pose_s=np.zeros([N_t,1024])
    epochs=np.int(np.floor(N_t/batch_size)+1) 
    for i in range(0,epochs):
        if (i+1)*batch_size<N_t:
            x_t_seg=x_t[i*batch_size:(i+1)*batch_size,:,:,:]
            x_s_seg=x_s[i*batch_size:(i+1)*batch_size,:,:,:]
    
            pose_t[i*batch_size:(i+1)*batch_size,:],pose_s[i*batch_size:(i+1)*batch_size,:]=sess.run([pose1,pose2],feed_dict={sample1: x_t_seg, sample2: x_s_seg})

        else:
            x_t_seg=x_t[N_t-batch_size:N_t,:,:,:]
            x_s_seg=x_s[N_t-batch_size:N_t,:,:,:]
    
            pose_t[N_t-batch_size:N_t,:],pose_s[N_t-batch_size:N_t,:]=sess.run([pose1,pose2],feed_dict={sample1: x_t_seg, sample2: x_s_seg})
    
    
    x_t_dis=np.zeros([0,112,112,1])
    x_s_dis=np.zeros([0,112,112,1])
    t_label_dis=np.zeros([0,])
    s_label_dis=np.zeros([0,])
    x_t_res_dis=np.zeros([0,112,112,1])
    x_s_res_dis=np.zeros([0,112,112,1])
    
    
    
    
    
    for i in range(0,N_t):
        
        if (i+1)%1000==0:
            print('Processing {} samples...'.format(i+1))
        
        if (t_label[i]-s_label[i])!=0:
            # pose_t 与第i个不同身份的索引
            
            x_t_dis=np.vstack((x_t_dis,x_t[i,:,:,:].reshape([1,112,112,1])))
            x_s_dis=np.vstack((x_s_dis,x_s[i,:,:,:].reshape([1,112,112,1])))
            
            t_label_dis=np.hstack((t_label_dis,t_label[i]))
            s_label_dis=np.hstack((s_label_dis,s_label[i]))      
            identity_t_index=(np.nonzero(t_label!=t_label[i]))[0]
            dis=np.sum(np.square(np.dot(np.ones([N_t,1]),pose_s[i,:].reshape([1,1024]))-pose_t),axis=1)
            dis[identity_t_index]=20000
            min_index=np.nonzero((dis==np.min(dis)))[0]
            
            x_t_res_dis=np.vstack((x_t_res_dis,x_t[min_index,:,:,:]))
            
            # pose_s 与第i个不同身份的索引
            identity_s_index=(np.nonzero(s_label!=s_label[i]))[0]
            dis=np.sum(np.square(np.dot(np.ones([N_t,1]),pose_t[i,:].reshape([1,1024]))-pose_s),axis=1)
            dis[identity_s_index]=20000
            min_index=np.nonzero((dis==np.min(dis)))[0]
            
            x_s_res_dis=np.vstack((x_s_res_dis,x_s[min_index,:,:,:]))

    x_t=np.vstack((x_t_dis,x_t_temp))
    x_s=np.vstack((x_s_dis,x_s_temp))
    t_label=np.float32(np.hstack((t_label_dis,t_label_temp)))
    s_label=np.float32(np.hstack((s_label_dis,s_label_temp)))
    x_t_res=np.vstack((x_t_res_dis,x_t_res_temp))
    x_s_res=np.vstack((x_s_res_dis,x_s_res_temp))
    similar_res=np.vstack((np.zeros([x_t_dis.shape[0],1]),similar_temp))
    
    N_t=x_t.shape[0]
    order_shuffled=random.sample(range(0,N_t),N_t)
    x_t=x_t[order_shuffled,:,:,:]
    x_s=x_s[order_shuffled,:,:,:]
    t_label=t_label[order_shuffled]
    s_label=s_label[order_shuffled]
    x_t_res=x_t_res[order_shuffled,:,:,:]
    x_s_res=x_s_res[order_shuffled,:,:,:]
    similar_res=similar_res[order_shuffled,:]
    
    
            #------------------------
            # 生成标签的one_hot模式
    t_label_one_hot=np_utils.to_categorical(t_label, num_classes=10)
    s_label_one_hot=np_utils.to_categorical(s_label, num_classes=10)
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    loss_batch=[]
    
    for k in range(0,10):
        for i in range(0,epochs):
            
            if (i+1)*batch_size<N_t:
        
                x_t_seg=x_t[i*batch_size:(i+1)*batch_size,:,:,:]
                x_s_seg=x_s[i*batch_size:(i+1)*batch_size,:,:,:]
                
                t_label_one_hot_seg=t_label_one_hot[i*batch_size:(i+1)*batch_size,:]
                s_label_one_hot_seg=s_label_one_hot[i*batch_size:(i+1)*batch_size,:]
                
                x_t_res_seg=x_t_res[i*batch_size:(i+1)*batch_size,:,:,:]
                x_s_res_seg=x_s_res[i*batch_size:(i+1)*batch_size,:,:,:]

                similar_res_seg=similar_res[i*batch_size:(i+1)*batch_size,:]
            
                _,b=sess.run([train,loss],feed_dict={sample1: x_t_seg, sample2: x_s_seg,y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:x_t_res_seg, res_sample2:x_s_res_seg, y_dis:similar_res_seg})
    
        
                loss_batch.append(b)
        
                
        
            else:
                x_t_seg=x_t[N_t-batch_size:N_t,:,:,:]
                x_s_seg=x_s[N_t-batch_size:N_t,:,:,:]
                
                t_label_one_hot_seg=t_label_one_hot[N_t-batch_size:N_t,:]
                s_label_one_hot_seg=s_label_one_hot[N_t-batch_size:N_t,:]
                
                x_t_res_seg=x_t_res[i*batch_size:(i+1)*batch_size,:,:,:]
                x_s_res_seg=x_s_res[i*batch_size:(i+1)*batch_size,:,:,:]
        

                similar_res_seg=similar_res[N_t-batch_size:N_t,:]
            
                _,b=sess.run([train,loss],feed_dict={sample1: x_t_seg, sample2: x_s_seg,y_sample1: t_label_one_hot_seg,y_sample2: s_label_one_hot_seg,res_sample1:x_t_res_seg, res_sample2:x_s_res_seg, y_dis:similar_res_seg})
                loss_batch.append(b)
    
        loss_show.append(loss_batch[-1])        
            

        if epoch==epoch:   #(epoch+1) % 100 ==0
            cr=0
            pred_test=np.zeros([0,10])
            epochs_test=np.int(np.floor(x_test.shape[0]/batch_size)+1)
            correct_prediction = tf.equal(tf.argmax(sample1_out,1), tf.argmax(y_sample1,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            for i in range(0,epochs_test):
                if (i+1)*batch_size<x_test.shape[0]:
            
                    cr_temp=np.sum(sess.run(correct_prediction, feed_dict={sample1: x_test[i*batch_size:(i+1)*batch_size,:,:,:], y_sample1: test_label_one_hot[i*batch_size:(i+1)*batch_size,:]}))
                    pred_test=np.vstack((pred_test,sess.run(sample1_out, feed_dict={sample1: x_test[i*batch_size:(i+1)*batch_size,:,:,:]})))
                    cr=cr+cr_temp
                else:
                    cr_temp=np.sum(sess.run(correct_prediction, feed_dict={sample1: x_test[i*batch_size:,:,:,:], y_sample1: test_label_one_hot[i*batch_size:,:]}))
                    pred_test=np.vstack((pred_test,sess.run(sample1_out, feed_dict={sample1: x_test[i*batch_size:(i+1)*batch_size,:,:,:]})))           
                    cr=cr+cr_temp
            
            cr_show.append(cr/x_test.shape[0])
            # c 为序列的识别率
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
                    if label-pre_label==0:
                        c=c+1
                    score=np.zeros([10,])   
                    c_sum=c_sum+1
            c=c/c_sum
            cr_show_seq.append(c)
            #-------------------------
            c=0
            c_sum=0
            test_set2=np.vstack((test_set,np.zeros([1,6],dtype='int32')))
            score=np.zeros([0,10])
            
            for s in range(0,test_set2.shape[0]-1):
    
                
                score=np.vstack((score,pred_test[s,:]))
                #
                if np.sum(np.nonzero(test_set2[s+1,1:4]-test_set2[s,1:4]))!=0:
                    label=test_set2[s,2]
                    score=score+1e-10
                    entropy=-1/np.sum(np.log(score)*score,axis=1)
                    entropy=np.diag(entropy/np.sum(entropy))
                    saliency_score=np.sum(np.dot(entropy,score),axis=0)
                    
                    
                    
                    pre_label=saliency_score==np.max(saliency_score)
                    
                    
                    if pre_label[label]==True:
                        c=c+1
                    score=np.zeros([10,])   
                    c_sum=c_sum+1
            c=c/c_sum
            cr_show_seq_saliency.append(c)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        print('Epoch #{}, loss: {}, cr_patch:{}, cr_seq:{}, cr_seq_saliency:{}'.format(epoch+1,loss_show[-1],cr_show[-1],cr_show_seq[-1],cr_show_seq_saliency[-1]))   




plt.figure(1)
plt.hold(True)
loss_show=loss_show/np.max(loss_show)
plt.plot(loss_show,label='loss',color ='red')
plt.hold(True)
plt.plot(cr_show,label='cr_patch',color ='blue')
plt.hold(True)
plt.plot(cr_show_seq,label='cr_seq',color ='green')
plt.hold(True)
plt.plot(cr_show_seq_saliency,label='cr_seq_saliency',color ='purple')




plt.legend(loc='upper right')     
plt.title('Loss Function(lr='+str()+')')
plt.xlabel('interation epoch')
#plt.ylabel('loss+rc')
plt.savefig('loss_cr.png')


print('maximal cr_patch:{}, maximal cr_seg:{}, maximal cr_seg_saliency:{}'.format(np.max(cr_show),np.max(cr_show_seq),np.max(cr_show_seq_saliency)))

np.save('para/loss_show.npy',loss_show)
np.save('para/cr_show.npy',cr_show)
np.save('para/cr_show_seq.npy',cr_show_seq)
np.save('para/cr_show_seq_saliency.npy',cr_show_seq_saliency)
np.save('para/epoch.npy',epoch)
np.save('para/pre_loss.npy',pre_loss)

saver=tf.train.Saver(max_to_keep=1)
saver.save(sess,'ckpt/mymodel.ckpt')







