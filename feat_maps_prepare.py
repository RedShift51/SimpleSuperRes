import cv2, numpy as np, os
from resnet_model import *

PATH = '/home/alex/IntLab/faces'
addrs = []
for i in os.listdir(PATH):
    addrs += [os.path.join(PATH, i, j) for j in os.listdir(os.path.join(PATH, i))]

X0 = tf.placeholder(dtype=tf.float32, shape=(1, 250, 250, 3))
res_model = Model(resnet_size=50, bottleneck=True, num_classes=1024,
                  num_filters=64, kernel_size=7, conv_stride=2,
                  first_pool_size=3, first_pool_stride=2, block_sizes=[3,4,6,3],
                  block_strides=[1, 1, 1, 1], final_size=1000, resnet_version=1,
                  data_format='channels_last')
Y = res_model(X0, training=False)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '/home/alex/IntLab/variables')
    for i0,i in enumerate(os.listdir(PATH)):
        cur_img = [os.path.join(PATH, i, j) for j in os.listdir(os.path.join(PATH, i)) if j.find('npy') == -1]
        for j0,j in enumerate(cur_img):
            a = sess.run(Y, feed_dict={X0: np.expand_dims(cv2.imread(j), 0)})
            np.save(cur_img[j0][:-4]+'.npy', a)
            #print(a.shape)
        print(cur_img, i0, 'of', len(os.listdir(PATH)))

"""
addrs = np.array(addrs)
np.random.shuffle(addrs)
addrs = list(addrs)
print(addrs)
ans = np.unique(np.array([str(cv2.imread(i).shape) for i in addrs]))
print(ans)
"""



