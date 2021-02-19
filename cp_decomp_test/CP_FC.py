#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:53:56 2020

@author: Ryan Solgi
"""

##### Imports ###################

import numpy as np
from tensor_operation import tenop as tp

####### Reading Data (Scipy.io is used for reading matlab dataset)###########

import scipy.io
mat = scipy.io.loadmat('tensorized_weights/cp_fc_layer.mat')
print(mat.keys())
tensor = mat['A']

####### ####### ####### ####### ####### ####### ####### ####### ####### 
tensor_shape = tensor.shape
print(tensor_shape)

####### CP decompostion
result = []
rank_list = []
for rank in range(12, 13):
    rank_list.append(int(rank))
    repeat = True
    tensor_dim = len(tensor_shape)
    factors = []

    for i in range(tensor_dim):
        #random initialization
        factors.append(np.random.rand(tensor_shape[i], rank))

        #eigenvector Initialization
        '''
        factors.append(np.zeros(shape = (tensor_shape[i], rank)))
        tensor_i = tp.unfold(tensor, i)

        u,  s, vh  =  np.linalg.svd(tensor_i)

        if rank>tensor_shape[i]:
            factors[i][:, :tensor_shape[i]] = copy.deepcopy(u)
        else:
            factors[i] = copy.deepcopy(u[:, :rank])
        '''

    t = 0
    while repeat:

        t += 1

        for n in range(tensor_dim):

            v = np.ones(shape=(rank, rank))  # v is updated as elementwise product of factors
            for i in range(tensor_dim):
                if i != n:
                    product = np.dot(np.transpose(factors[i]), factors[i])
                    v = np.multiply(product, v)

            rao_product = np.ones(shape=(1, rank))  # rao_proudct will be updated
            for i in range(tensor_dim):
                if i != n:
                    rao_product = tp.khatri_rao(rao_product, factors[i])

            tensor_n = tp.unfold(tensor, n)

            tensor_rao_product = np.dot(tensor_n, rao_product)

            inverse = np.linalg.pinv(v)

            factors[n] = np.dot(tensor_rao_product, inverse)

            tensor_hat = np.dot(factors[n], np.transpose(rao_product))
            error = tensor_n-tensor_hat

            norm_error = np.linalg.norm(error, 'fro')
            norm_tensor_n = np.linalg.norm(tensor_n, 'fro')
            obj = norm_error/norm_tensor_n

        # if int(t/10) = =0:
        #     print('obj', obj)
        if t == 50:
            repeat = False

    factors = np.array(factors)
    # print(len(factors))
    print('Rank ' + str(rank) + ': ',  obj)
    ret = {'error': obj}
    for i in range(len(tensor_shape)):
        ret[chr(i + 65)] = factors[i]
    scipy.io.savemat('factors_' + str(rank) + '.mat',  ret)
    result.append(obj)

rank_array = np.array(rank_list)
result_array = np.array(result)

#############################    Report Graph    ##############

# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# ax  =  plt.figure().gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer = True))
# plt.plot(rank_array, result_array)
# plt.title('CP-FC Result')
# plt.xlabel('Rank')
# plt.ylabel('Fitness')
# plt.savefig('result/CP_FC_Graph.pdf')
# plt.show()

