#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:00:44 2020

@author: Ryan
"""
import numpy as np
import copy

###################### Outer product of some given vectors to make a tensor
###################### ###################### ###################### ###################### 
class tenop(object):
    def o_product_vec(x): # x is a list of vectors (1 d array), the output is the resutlting tensor
        num_vectors=len(x)

        len_vectors=np.zeros(num_vectors)
        
        for i in range(num_vectors):
            len_vectors[i]=len(x[i])
            
        tensor_shape=[]
        for i in range(num_vectors):
            tensor_shape.append(int(len_vectors[i]))
            #tensor_shape.append(int(len_vectors[num_vectors-i-1]))
       
        
        memory=[]
        counter=np.zeros(num_vectors)
        
        progress=True
        
        while progress:
            product=1
            for i in range (num_vectors):
                product*=x[num_vectors-i-1][int(counter[num_vectors-i-1])]
                #product*=x[i][int(counter[i])]
            memory.append(product)
            if counter[num_vectors-1]<len_vectors[num_vectors-1]-1:
            #if counter[0]<len_vectors[0]-1:
                counter[num_vectors-1]+=1
                #counter[0]+=1
            else:
                counter[num_vectors-1]=0
                #counter[0]=0
                turn=True
                j=num_vectors-1
                #j=0
                while turn:
                    if j!=0:
                    #if j!=num_vectors-1:
                        if counter[j-1]<len_vectors[j-1]-1:
                        #if counter[j+1]<len_vectors[j+1]-1:
                            counter[j-1]+=1
                            #counter[j+1]+=1
                            turn=False
                        else:
                            counter[j-1]=0
                            #counter[j+1]=0
                            j-=1
                            #j+=1
                    else:
                        turn=False
                        progress=False
        array=np.array(memory)
        tensor=array.reshape(tensor_shape)
        
        
        return tensor
                
###################### unfolding 
###################### ###################### ###################### ######################               
    def unfold(x,n):

        return np.rollaxis(x, n, 0).reshape(x.shape[n], -1)
        
######################  Khatri_rao product  
###################### ###################### ###################### ######################                      
    def khatri_rao(a,b):
        
        a_shape=a.shape
        b_shape=b.shape
        if len(a_shape)!=2 or len(b_shape)!=2:
            raise('input is not matrix')
        if a_shape[1]!=b_shape[1]:
            raise('for khatri rao product matrices must have the same column number')
        product=np.zeros(shape=(a_shape[1],a_shape[0]*b_shape[0]))
        a=np.transpose(a)
        b=np.transpose(b)
        for i in range(a_shape[1]):
            product[i]=np.kron(a[i],b[i])
        product=np.transpose(product)
        return product
    
        
######################  Reshape similar to Matlab reshape function
###################### ###################### ###################### ######################                              
        
    def matrix_reshape(x,out_shape): #out_shape=[num of columns,num of rows]
        if len(out_shape)!=2:
            raise('length shape must be two')
        xt=np.transpose(x)
        y=xt.reshape((int(out_shape[0]),int(out_shape[1])))
        yt=np.transpose(y)
        return yt
######################  Reshape similar to Matlab reshape function for folding to 3D tensor
###################### ###################### ###################### ######################              
    
    def folding_3d(x,out_shape): #out_shape=[depth, #columns, #rows] in matlab this is reverse in order
        if len(out_shape)!=3:
            raise('length shape must be two')
        xt=np.transpose(x)
        s=out_shape
        tensor=xt.reshape((int(s[0]),int(s[1]),int(s[2])))
        shap=tensor.shape
        tensor_t=np.zeros(shape=(shap[0],shap[2],shap[1]))
        for i in range(0,shap[0]):
            tensor_t[i,:,:]=np.transpose(tensor[i,:,:])
            
        return tensor_t
        
######################  Truncated SVD which return u,r and v transpose based on a given threshold of accuracy
###################### ###################### ###################### ######################              
        

        
        

    def truncated_svd(x,threshold): #calcualte the r accroding to threshold and output the left and right matrices based on the rank
        
        shape_x=x.shape
        r_max=shape_x[0]

        u, s, vh = np.linalg.svd(x,full_matrices=False)

        s=np.diag(s)
        us=np.dot(u,s)
        
        
        r=0

        sigma=2
        while sigma>threshold and r<r_max:
 
            r+=1
            
           
        
            x_test=np.zeros(shape=shape_x)
            
            #for i in range(0,r):
            
            x_test=np.dot(us[:,:r],vh[:r,:])

            #product=s[:r]*product
          
            
            error=x-x_test
            
            norm_error=np.linalg.norm(error,'fro')
            
            sigma=norm_error/np.linalg.norm(x,'fro')
            
        #for j in range(0,r):
        
        left_matrix=u[:,:r]
         
        product=np.dot(s[:r,:r],vh[:r,:])
        right_matrix=product
           
            

        
        return r,left_matrix,right_matrix
            

        
        
        
        
        
        
                
        
            
        
        
        

        
    
        
            