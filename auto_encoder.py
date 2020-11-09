# -*- coding: utf-8 -*-
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import random
import numpy as np

def AutoEncoderTrain(trainset, testset, num_epoch=1000, batch_size=256, flag_train=True, lr=1e-6):
    #sess = tf.InteractiveSession()
    sess = tf.InteractiveSession()
    #------------------------------------------------------------------------------
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    #------------------------------------------------------------------------------
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
    def deconv2d_2x2(x, W, o):
      return tf.nn.conv2d_transpose(x,W,output_shape=o,strides=[1,2,2,1],padding="SAME")
    
    def deconv2d_1x1(x, W, o):
      return tf.nn.conv2d_transpose(x,W,output_shape=o,strides=[1,1,1,1],padding="SAME")
    
    sample=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
    res_sample=tf.placeholder(dtype=tf.float32,shape=[None,112,112,1])
    
    #------------------------------------------------------------------------------
    # Parameters in encoder
    W_conv1_encoder = weight_variable([5, 5, 1, 32])
    b_conv1_encoder = bias_variable([32])
    
    W_conv2_encoder = weight_variable([5, 5, 32, 64])
    b_conv2_encoder = bias_variable([64])
    
    W_conv3_encoder = weight_variable([5, 5, 64, 128])
    b_conv3_encoder = bias_variable([128])
    
    W_fc1_encoder = weight_variable([14 * 14 * 128, 2048])
    b_fc1_encoder = bias_variable([2048])
    
    W_fc2_encoder = weight_variable([2048, 1024])
    b_fc2_encoder = bias_variable([1024])
    
    
    # Parameters in Decoder
    
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
    
    
    #------------------------------------------------------------------------------
    def Encoder(x):
        # h_conv1: 112*112*32
        h_conv1_encoder = tf.nn.relu(conv2d(x, W_conv1_encoder) + b_conv1_encoder)
        # h_pool1: 56*56*32
        h_pool1_encoder = max_pool_2x2(h_conv1_encoder)
        # h_conv2: 56*56*64
        h_conv2_encoder = tf.nn.relu(conv2d(h_pool1_encoder, W_conv2_encoder) + b_conv2_encoder)
        # h_pool2: 28*28*64
        h_pool2_encoder = max_pool_2x2(h_conv2_encoder)
        # h_conv3: 28*28*128
        h_conv3_encoder = tf.nn.relu(conv2d(h_pool2_encoder, W_conv3_encoder) + b_conv3_encoder)
        # h_pool3: 14*14*128
        h_pool3_encoder = max_pool_2x2(h_conv3_encoder)
        # h_pool3_flat: 14*14*128=25088
        h_pool3_flat_encoder = tf.reshape(h_pool3_encoder, [-1, 14*14*128])
        # h_fc1: 2048
        h_fc1_encoder = tf.nn.relu(tf.matmul(h_pool3_flat_encoder, W_fc1_encoder) + b_fc1_encoder)
        # h_fc2: 1024
        h_fc2_encoder = tf.nn.sigmoid(tf.matmul(h_fc1_encoder, W_fc2_encoder) + b_fc2_encoder)
        
        return h_fc2_encoder
    
    
    def Decoder(x):
        h_fc1_decoder = tf.nn.relu(tf.matmul(x, W_fc1_decoder) + b_fc1_decoder)
        h_flat_decoder = tf.reshape(h_fc1_decoder, [-1, 14,14,128])
        h_deconv1_decoder = tf.nn.relu(deconv2d_2x2(h_flat_decoder, W_deconv1_decoder,[batch_size, 28,28,64]) + b_deconv1_decoder)
        h_deconv2_decoder = tf.nn.relu(deconv2d_2x2(h_deconv1_decoder, W_deconv2_decoder,[batch_size, 56,56,32]) + b_deconv2_decoder)
        h_deconv3_decoder = tf.nn.relu(deconv2d_2x2(h_deconv2_decoder, W_deconv3_decoder,[batch_size, 112,112,16]) + b_deconv3_decoder)
        h_deconv4_decoder = tf.nn.sigmoid(deconv2d_1x1(h_deconv3_decoder, W_deconv4_decoder,[batch_size, 112,112,1]) + b_deconv4_decoder)
    
        return h_deconv4_decoder
    
    #------------------------------------------------------------------------------
    coding=Encoder(sample)
    pred_sample=Decoder(coding)
    
    #------------------------------------------------------------------------------
#    loss=-tf.reduce_sum(res_sample*tf.log(tf.maximum(pred_sample,1e-10)))
    loss=tf.reduce_mean(tf.square(res_sample-pred_sample))
    tr=tf.train.AdamOptimizer(lr).minimize(loss)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if os.path.exists('auto-encoder-model'):
        model_file=tf.train.latest_checkpoint('auto-encoder-model/')
        saver.restore(sess,model_file)
        print('loading auto-encoder-model success...')
    if os.path.exists('auto-encoder-model/loss_show.npy'):
        loss_show = np.load('auto-encoder-model/loss_show.npy')
        loss_show=loss_show.tolist()
    else:
        loss_show=[] 


    if flag_train==True:
        N_t=trainset.shape[0]
        for w in range(0,num_epoch):
            order_shuffled=random.sample(range(0,N_t),N_t)
            trainset_shuffled=trainset[order_shuffled,:,:,:]
            
            epochs=np.int(np.floor(N_t/batch_size)+1)
            for i in range(0,epochs):
                if (i+1)*batch_size<N_t:
                    trainset_batch=trainset_shuffled[i*batch_size:(i+1)*batch_size,:,:,:]
                    _,b=sess.run([tr, loss],feed_dict={sample: trainset_batch, res_sample: trainset_batch})
                else:
                    trainset_batch=trainset_shuffled[N_t-batch_size:N_t,:,:,:]
                    _,b=sess.run([tr, loss],feed_dict={sample: trainset_batch, res_sample: trainset_batch})
                print('Epoch', w+1,', loss = ', b)
            loss_show.append(b)
            
            
            
        saver=tf.train.Saver(max_to_keep=1)
        saver.save(sess, 'auto-encoder-model/mymodel.ckpt')
        np.save('auto-encoder-model/loss_show.npy', loss_show)
    
## Component : Predition and Reconstruction 预测编码和重构 ----------------------------------------------------------------

    
#    batch_size=256
#    testset=x_test
#    pred_test=np.zeros([0,testset.shape[1],testset.shape[2],testset.shape[3]])
#    if np.mod(testset.shape[0],batch_size)==0:
#        epochs_test=np.int(testset.shape[0]/batch_size)
#        for i in range(0,epochs_test):
#            pred_test=np.vstack((pred_test,sess_auto.run(pred_sample, feed_dict={sample: testset[i*batch_size:(i+1)*batch_size,:,:,:]})))
#    else:
#        epochs_test=np.int(testset.shape[0]/batch_size+1)
#        for i in range(0,epochs_test):
#            if (i+1)*batch_size<testset.shape[0]:
#                
#                pred_test=np.vstack((pred_test,sess_auto.run(pred_sample, feed_dict={sample: testset[i*batch_size:(i+1)*batch_size,:,:,:]})))
#    
#            else:
#                temp=sess_auto.run(pred_sample, feed_dict={sample: testset[testset.shape[0]-batch_size:testset.shape[0],:,:,:]})
#                pred_test=np.vstack((pred_test,temp[(batch_size-testset.shape[0]+i*batch_size):,:,:,:]))           
#


    return sess, sample, coding, pred_sample

 



