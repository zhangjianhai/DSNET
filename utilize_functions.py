# -*- coding: utf-8 -*-

import numpy as np

def process_data(ske, ske_v, t):
    # t 局部片段的长度
    c=0
    segments_train=np.zeros([0,6],np.int)
    segments_test=np.zeros([0,6],np.int)
    #segments=np.zeros([1,6],np.int)
    for a in range(0,20):
        for s in range(0,10):
            r_n=0
            for r in [0,1,2]:
                r_n=r_n+1
                if ske_v[a,s,r]==1:
                    if ske[a,s,r].shape[2]>1:
                        if r_n==1:
                            ske_tmp=ske[a,s,r]
                            length=ske_tmp.shape[2]
                            
                            for step in range(0,1):
                                for k in range(0,length):
                                    if (step+1)*(t-1)+k<length or (step+1)*(t-1)+k<length:
                                        segments_tmp=np.zeros([1,6],np.int)
                                        segments_tmp[0,0]=c
                                        segments_tmp[0,1]=a
                                        segments_tmp[0,2]=s
                                        segments_tmp[0,3]=r
                                        segments_tmp[0,4]=step
                                        segments_tmp[0,5]=k
                                        segments_train=np.vstack((segments_train, segments_tmp))
                                        c=c+1
                        else:
                        
                            ske_tmp=ske[a,s,r]
                            length=ske_tmp.shape[2]
                            
                            for step in range(0,1):
                                for k in range(0,length):
                                    if (step+1)*(t-1)+k<length or (step+1)*(t-1)+k<length:
                                        segments_tmp=np.zeros([1,6],np.int)
                                        segments_tmp[0,0]=c
                                        segments_tmp[0,1]=a
                                        segments_tmp[0,2]=s
                                        segments_tmp[0,3]=r
                                        segments_tmp[0,4]=step
                                        segments_tmp[0,5]=k
                                        segments_test=np.vstack((segments_test, segments_tmp))
                                        c=c+1   
#    segments_train=np.delete(segments_train,0,axis=0)         #
#    segments_test=np.delete(segments_test,0,axis=0)
    return segments_train, segments_test

# getS --> 通过segments_index 索引，得到ske的截取特征
def getS(t,segment_index,ske):
    # segment_index的行的个数，i.e.也就是样本的个数，
    segment_index_num=len(segment_index)  
    # 3,20是ske数据中的格式，3代表xyz坐标，20代表jionts，
    action_seg=np.zeros([segment_index_num,3,20,t])
    # i对每一个样本循环处理，
    for i in range(0,segment_index_num):
        # 要截取得到的片段上的时间索引，
        seg_index=np.zeros([t],np.int)
        for w in range(0,t):
            seg_index[w]=segment_index[i,5]+segment_index[i,4]*w+w
            action_seg[i,:,:,:]=ske[segment_index[i,1],segment_index[i,2],segment_index[i,3]][:,:,seg_index]    
    return action_seg

# getImages --> 通过getS得到的action_seg 得到图像表示，
def getImages(action_seg, t):
    n_a=action_seg.shape[0]
        #imgs=np.zeros([n,224,224])
    imgs=np.zeros([n_a,112,112])
    for i in range(0,n_a):
        imgs[i,:,:]=getImage(action_seg[i,:,:,:], t)
    return imgs
        
# getImages --> 通过getS得到的action_seg 得到图像表示，
def getImage(action_seg, t):
    import math
    #That's because imresize has been removed from scipy since v1.3.0rc1

    from scipy.misc import imresize
    joint_pairs=[[3,6],[2,3],[19,2],[7,0],[8,1],[9,7],[10,8],[13,4],[14,5],[15,13],[16,14]]
    n=len(joint_pairs)
    jnts_angle=np.zeros([np.int(n*(n-1)/2),t])
    for s in range(0,t):
        k=0
        for i in range(0,n):
            for j in range(i,n):
                if i!=j:                   
                    bone_child=action_seg[:,joint_pairs[i][0],s]-action_seg[:,joint_pairs[i][1],s]
                    bone_child=bone_child/np.sqrt(np.dot(bone_child,bone_child))
                    bone_parent=action_seg[:,joint_pairs[j][0],s]-action_seg[:,joint_pairs[j][1],s]
                    bone_parent=bone_parent/np.sqrt(np.dot(bone_parent,bone_parent))
                    joint_dot=np.dot(bone_child,bone_parent)
                    jnts_angle[k,s]=math.acos(joint_dot)/math.pi #转变为角度且归一化
                    k=k+1
#        img=jnts_angle/255
#        img=imresize(jnts_angle,(2*self.n,2*self.t))/255
        img=imresize(jnts_angle,(112,112))/255
        #print(img)
        return img
