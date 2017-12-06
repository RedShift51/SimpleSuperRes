#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

        if self.nonlin == 'relu':
            return tf.nn.relu(ans)
        elif self.nonlin == 'sigmoid':
            return tf.nn.sigmoid(ans)
        else:
            return tf.nn.relu(ans)


    def __call__(self, x):
        return self.transform(x)

sequence_pos = [np.arange(0,24,3), np.arange(1,25,3), np.arange(2,26,3)]
#sec_seq = np.arange(1,26,3)
#thr_seq = np.arange(2,27,3)
sequence_pos = [[np.meshgrid(i,j) for j in sequence_pos] for i in sequence_pos]
np.meshgrid(fir_seq, sec_seq)    

def reshaping(X):
    
    #nine points
    #cond = lambda i : tf.logical_and(tf.equal(tf.truncatemod(i,3), 0), \
    #                                 tf.less_equal(i, 25))
    x = tf.range(0,25,3)
    #print(x.get_shape().as_list())
    #cond = lambda i : tf.equal(tf.truncatemod(i+1,3), 0)
    slicing = lambda i : tf.slice(X, [0,0,i], [-1,-1,3])
    #with tf.Session() as sess:
    #    print(sess.run(cond([1,2,3,4,5])))
    res = tf.map_fn(slicing, x)
    print(res[0].get_shape().as_list())
    """
    X1 = tf.slice(X, [0,0,0], [-1,-1,3])
    X2 = tf.slice(X, [0,0,3], [-1,-1,3])
    X3 = tf.slice(X, [0,0,6], [-1,-1,3])
    X4 = tf.slice(X, [0,0,9], [-1,-1,3])
    X5 = tf.slice(X, [0,0,12], [-1,-1,3])
    X6 = tf.slice(X, [0,0,15], [-1,-1,3])
    X7 = tf.slice(X, [0,0,18], [-1,-1,3])
    X8 = tf.slice(X, [0,0,21], [-1,-1,3])
    X9 = tf.slice(X, [0,0,24], [-1,-1,3])
    """
    fir_seq = tf.range(0,25,3)
    sec_seq = tf.range(1,26,3)
    thr_seq = tf.range(2,27,3)
    
    X1 = tf.SparseTensor(indices = , values = X1, dense_shape = [24,24,3])
    
    
    
tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32, shape=(1, 12, 12, 3))
Y = tf.placeholder(dtype=tf.float32, shape=(1, 24, 24, 3))

#========= first l-1 layers
def LayersBlock(X, scope):
    #patches learning
    X = tf.nn.sigmoid(ConvLayer2D('conv1', kernel_shape = [3,3,3,6], num_filters = 6)(X))
    X = batch_norm(X)
    X = tf.nn.max_pool(X, [1,2,2,1], [1,1,1,1], 'VALID')
    print(X.get_shape().as_list())
    X = tf.nn.sigmoid(ConvLayer2D('conv2', kernel_shape = [3,3,6,12], num_filters = 10)(X))
    X = batch_norm(X)
    X = tf.nn.max_pool(X, [1,2,2,1], [1,1,1,1], 'VALID')
    print(X.get_shape().as_list())
    X = tf.nn.sigmoid(ConvLayer2D('conv3', kernel_shape = [3,3,12,18], num_filters = 10)(X))
    X = batch_norm(X)
    X = tf.nn.max_pool(X, [1,2,2,1], [1,1,1,1], 'VALID')
    print(X.get_shape().as_list())
    X = tf.nn.sigmoid(ConvLayer2D('conv4', kernel_shape = [3,3,18,32], num_filters = 10)(X))
    X = batch_norm(X)
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
    X = reshaping(X)#, shape = [1, 24, 24, 3])
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
        
        
        




def body(x):
    a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
    b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    c = a + b
    return tf.nn.relu(x + c)

def condition(x):
    return tf.reduce_sum(x) < 100

x = tf.Variable(tf.constant(0, shape=[2, 2]))

with tf.Session():
    tf.initialize_all_variables().run()
    result = tf.while_loop(condition, body, [x])
    print(result.eval())