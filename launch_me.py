#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:04:31 2017

@author: alex
"""

import tensorflow as tf, numpy as np
import scipy
from tensorflow.contrib.layers import batch_norm

class ConvLayer2D():
    def __init__(self, scope, kernel_shape = [2,2,3,3], num_filters = 1, \
                 padding = 'SAME', strides = [1,1,1,1], nonlin = 'relu'):
        self.weights = []
        self.padding = padding
        self.nonlin = nonlin
        self.strides = strides
        with tf.variable_scope(scope):
            for i in range(num_filters):
                with tf.variable_scope(str(i)):
                    #print(i)
                    self.weights.append(\
                            [tf.get_variable('weights', shape = kernel_shape, \
                            initializer = tf.glorot_normal_initializer()), \
                            tf.get_variable('bias', shape = kernel_shape[-1], \
                            initializer = tf.zeros_initializer())])
                    #print(self.weights)
    
    def transform(self, x):
        ans = tf.concat([tf.nn.conv2d(x, self.weights[i][0], padding = self.padding, \
                        strides = self.strides) + self.weights[i][1] \
                        for i in range(len(self.weights))], axis = 0)
        """
        conv = tf.concat([tf.nn.conv2d(x, self.weights[i][0], padding = self.padding, \
                strides = self.strides) + \
                self.weights[i][1] for i in range(len(self.weights))],\
                axis = 0)
        """
        if self.nonlin == 'relu':
            return tf.nn.relu(ans)
        elif self.nonlin == 'sigmoid':
            return tf.nn.sigmoid(ans)
        else:
            return tf.nn.relu(ans)

    def __call__(self, x):
        return self.transform(x)

tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32, shape=(1, 12, 12, 3))
Y = tf.placeholder(dtype=tf.float32, shape=(1, 24, 24, 3))

#========= first l-1 layers
def LayersBlock(X, scope):
    #patches learning
    X = tf.nn.sigmoid(ConvLayer2D('conv1', kernel_shape = [3,3,3,6], num_filters = 6)(X))
    X = tf.nn.relu(batch_norm(X))
    X = tf.nn.max_pool(X, [1,2,2,1], [1,1,1,1], 'VALID')
    print(X.get_shape().as_list())
    X = tf.nn.sigmoid(ConvLayer2D('conv2', kernel_shape = [3,3,6,12], num_filters = 10)(X))
    X = tf.nn.relu(batch_norm(X))
    X = tf.nn.max_pool(X, [1,2,2,1], [1,1,1,1], 'VALID')
    print(X.get_shape().as_list())
    X = tf.nn.sigmoid(ConvLayer2D('conv3', kernel_shape = [3,3,12,18], num_filters = 10)(X))
    X = tf.nn.relu(batch_norm(X))
    X = tf.nn.max_pool(X, [1,2,2,1], [1,1,1,1], 'VALID')
    print(X.get_shape().as_list())
    X = tf.nn.sigmoid(ConvLayer2D('conv4', kernel_shape = [3,3,18,32], num_filters = 10)(X))
    X = tf.nn.relu(batch_norm(X))
    X = tf.nn.max_pool(X, [1,2,2,1], [1,1,1,1], 'VALID')
    print(X.get_shape().as_list())
    X = tf.nn.sigmoid(ConvLayer2D('conv5', kernel_shape = [3,3,32,9], num_filters = 3)(X))
    print(X.get_shape().as_list())
    X = batch_norm(X)
    #X = tf.nn.sigmoid(X)
    #X = batch_norm(X)
    X = tf.concat([tf.reduce_sum(batch_norm(tf.slice(X, [0,0,0,0],[6000,-1,-1,-1])), axis = 0), \
                  tf.reduce_sum(batch_norm(tf.slice(X, [6000,0,0,0],[6000,-1,-1,-1])), axis = 0), \
                  tf.reduce_sum(batch_norm(tf.slice(X, [12000,0,0,0],[6000,-1,-1,-1])), axis = 0), \
                  ], axis = 2)
    X = tf.nn.sigmoid(X)
    print(X.get_shape().as_list())
    X = tf.reshape(X, shape = [1, 24, 24, 3])
    return X
    
ans = LayersBlock(X, scope='block')
loss = tf.reduce_mean(tf.multiply(ans-Y, ans-Y))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

import os, matplotlib.pyplot as plt
select_list = os.listdir('/home/alex/Downloads/images/')
n = int(len(select_list)*0.8)
train, test = select_list[:n], select_list[n:]
loss_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i0,i in enumerate(train[:100]):
        base_img = np.load(open('/home/alex/Downloads/images/'+i, 'rb'))
        entry, out_result = [], []
        #for k in range(4):
        pos_small_x, pos_small_y = np.random.randint(low=0,high=48,size=1)[0],\
                    np.random.randint(low=0,high=48,size=1)[0]
        pos_big_x, pos_big_y = pos_small_x*2, pos_small_y*2
        entry=np.expand_dims(scipy.misc.imresize(base_img, (60,60,3))*1./255, \
                axis=0)[:,pos_small_x:pos_small_x+12, pos_small_y:pos_small_y+12,:]
        out_result=np.expand_dims(base_img,axis=0)\
                    [:,pos_big_x:pos_big_x+24, pos_big_y:pos_big_y+24,:]
        #entry = np.concatenate([k for k in entry], axis = 0)
        #out_result = np.concatenate([k for k in out_result], axis = 0)
        for j in range(5):
            train_step.run(feed_dict = {X : entry, Y : out_result})
                #base_img[pos_big_x:pos_big_x+24, pos_big_y:pos_big_y+24,:]})
        loss_list.append(sess.run(loss, feed_dict = {X : entry, Y : out_result}))
                    #Y:base_img[pos_big_x:pos_big_x+24, pos_big_y:pos_big_y+24,:]}))
        print(i0, len(train), loss_list[-1])
    
    pict = []
    for i0,i in enumerate(test[:50]):
        base_img = np.load(open('/home/alex/Downloads/images/'+i, 'rb'))
        pos_small_x, pos_small_y = np.random.randint(low=0,high=48,size=1)[0],\
                                    np.random.randint(low=0,high=48,size=1)[0]
        pos_big_x, pos_big_y = pos_small_x*2, pos_small_y*2
        entry = np.expand_dims(scipy.misc.imresize(base_img, (60,60,3))*1./255, \
                    axis=0)[:,pos_small_x:pos_small_x+12, pos_small_y:pos_small_y+12,:]
        pict.append([entry,sess.run(ans, feed_dict={X:entry})])    
        
        
        
        