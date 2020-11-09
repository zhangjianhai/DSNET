# -*- coding: utf-8 -*-


import numpy as np

def dis_tab(aug1,aug2):
    n_aug1=aug1.shape[0]
    n_aug2=aug2.shape[0]
    dis_t=np.zeros([n_aug1,n_aug2],np.float64)
    for i in range(0,n_aug1):
        for j in range(0,n_aug2):
            dis_t[i,j]=np.sum(np.square(aug1[i,:]-aug2[j,:]))
    return dis_t

def path_index(dis_table):
    m,n=dis_table.shape
    # initializing
    dp_dis=np.zeros([m,n,3],np.float64)
    dp_dis[0,0,:]=np.array([0,0,2*dis_table[0,0]])
    for i in range(1,m):
        dp_dis[i,0,:]=np.array([i-1,0,dp_dis[i-1,0,2]+dis_table[i,0]])           
    for j in range(1,n):
        dp_dis[0,j,:]=np.array([0,j-1,dp_dis[0,j-1,2]+dis_table[0,j]])                             
        
    for i in range(1,m):
        for j in range(1,n):
            temp=[dp_dis[i,j-1,2]+dis_table[i,j], dp_dis[i-1,j-1,2]+2*dis_table[i,j], dp_dis[i-1,j,2]+dis_table[i,j]]
            val=min(temp)
            index=np.where(temp==min(temp))
            index=index[0]
            order=np.array([[i,j-1],[i-1,j-1],[i-1,j]])
            
            dp_dis[i,j,:]=np.array([order[index,:][0][0],order[index,:][0][1],val])
            
            min_path_table=dp_dis
            flag=True
            min_path=[]
            min_path.append([m-1,n-1])
            count=0
            while flag:
                temp=min_path_table[min_path[count][0],min_path[count][1]][0:2]
                count=count+1
                min_path.append([np.int(temp[0]),np.int(temp[1])])
                if np.sum(temp)==0:
                    break
                
    return min_path, min_path_table         
                








def DTW_tool(aug1,aug2):
    dis_table=dis_tab(aug1,aug2)
    min_path,min_path_table=path_index(dis_table)
    min_dis=min_path_table[-1,-1,2]
    return min_dis, min_path

