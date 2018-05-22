""" Single image super resolution problem solution, loss function includes feature maps """
""" We multiply resolution by 4 """
import tensorflow as tf, numpy as np, os, cv2
from resnet_model import *

PATH = '/home/alex/IntLab/faces'
addrs, flag = [], 0
for i in os.listdir(PATH):
    addrs += [(os.path.join(PATH, i, j)[:-4]+'.jpg', os.path.join(PATH, i, j)[:-4]+'.npy') for j in os.listdir(os.path.join(PATH, i))]
addrs = {i0: i for i0,i in enumerate(addrs)}

def gram_func(Xinp):
    print(tf.map_fn(lambda x: tf.matmul(tf.transpose(tf.reshape(x, [63 * 63, 2048])),
                             tf.reshape(x, [63 * 63, 2048])), Xinp))
    """
    1/0
    tf.expand_dims(tf.matmul(tf.transpose(tf.reshape(x, [12 * 30, 2048])),
                             tf.reshape(x, [12 * 30, 2048])), axis=0)
    tf.matmul(tf.transpose(tf.reshape(x, [x.get_shape().as_list()[0] * x.get_shape().as_list()[1],
                                          x.get_shape().as_list()[2]])),
              tf.reshape(x, [x.get_shape().as_list()[0] * x.get_shape().as_list()[1],
                             x.get_shape().as_list()[2]]))
    """
    #Xinp = tf.map_fn(lambda x: tf.transpose(tf.map_fn(lambda y: tf.squeeze(y), tf.transpose(x, perm=[2, 0, 1])),[1,2,0]), Xinp)
    #Xinp = tf.map_fn(lambda x: tf.matmul(tf.transpose(x), x), Xinp)

    return Xinp

start_size = (125, 125)
batch_size = 3

X0 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 125, 125, 3])
Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 63, 63, 2048])

""" Solving architecture """
with tf.variable_scope('1'):
    X = tf.layers.conv2d(inputs=X0, filters=32, kernel_size=(3, 3), strides=(1, 1),
                     padding='SAME', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                     data_format='channels_last')
X = tf.nn.relu(X)
with tf.variable_scope('2'):
    X1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=(3, 3), strides=(1, 1),
                     padding='SAME', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                     data_format='channels_last')
X1 = tf.nn.relu(X1)
with tf.variable_scope('3'):
    X1 = tf.layers.conv2d(inputs=X1, filters=32, kernel_size=(3, 3), strides=(1, 1),
                     padding='SAME', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                     data_format='channels_last')
X1 = tf.nn.relu(X1)
with tf.variable_scope('4'):
    X1 = tf.layers.batch_normalization(inputs=X1, name='bn1')
X = X + X1
X = tf.image.resize_images(images=X, size=[i*2 for i in start_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
with tf.variable_scope('5'):
    X = tf.layers.conv2d(inputs=X, filters=64, kernel_size=(3, 3), strides=(1, 1),
                     padding='SAME', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                     data_format='channels_last')
X = tf.nn.relu(X)
"""
with tf.variable_scope('6'):
    X1 = tf.layers.conv2d(inputs=X, filters=64, kernel_size=(3, 3), strides=(1, 1),
                     padding='SAME', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                     data_format='channels_last')
X1 = tf.nn.relu(X1)
with tf.variable_scope('7'):
    X1 = tf.layers.conv2d(inputs=X1, filters=64, kernel_size=(3, 3), strides=(1, 1),
                     padding='SAME', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                     data_format='channels_last')
X1 = tf.nn.relu(X1)
with tf.variable_scope('8'):
    X1 = tf.layers.batch_normalization(inputs=X1, name='bn2')
X = X + X1
#X = tf.image.resize_images(images=X, size=[i*4 for i in start_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
"""
with tf.variable_scope('9'):
    X = tf.layers.conv2d(inputs=X, filters=32, kernel_size=(3, 3), strides=(1, 1),
                     padding='SAME', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                     data_format='channels_last')
X = tf.nn.relu(X)
with tf.variable_scope('10'):
    X = tf.layers.conv2d(inputs=X, filters=3, kernel_size=(3, 3), strides=(1, 1),
                     padding='SAME', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                     data_format='channels_last')


""" ======================================================================================================= """
res_model = Model(resnet_size=50, bottleneck=True, num_classes=1024,
                  num_filters=64, kernel_size=7, conv_stride=2,
                  first_pool_size=3, first_pool_stride=2, block_sizes=[3,4,6,3],
                  block_strides=[1, 1, 1, 1], final_size=1000, resnet_version=1,
                  data_format='channels_last')

vars = tf.trainable_variables()
diff = res_model(X, training=False)
#with tf.variable_scope('feat_map'):
#with tf.variable_scope('resnet_model') as scope:
#    scope.reuse_variables()
#    feat_map = res_model(X, training=False)

vars = tf.trainable_variables()

loss = tf.reduce_mean((diff - Y) ** 2) + \
    0.002 * tf.add_n([tf.nn.l2_loss(i) for i in vars if str(i).find('resnet') == -1]) #+ \
    #tf.reduce_mean((gram_func(diff) - gram_func(Y)) ** 2)

opti = [tf.train.AdamOptimizer(0.001/(3.*i)).minimize(loss) for i in range(1, 7)]

saver_resnet = tf.train.Saver([i for i in vars if str(i).find('resnet') != -1])
#saver_other = tf.train.Saver([i for i in vars if str(i).find('feat_map') != -1])
saver_model = tf.train.Saver([i for i in vars if str(i).find('resnet') == -1])

print(type(addrs))
addrs = {i: addrs[i] for i in range(int(len(list(addrs.keys()))-len(list(addrs.keys())) % 3))}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_resnet.restore(sess, '/home/alex/IntLab/variables')

    batch = [[addrs[batch_size*i+j] for j in range(batch_size)] for i in range(int(len(addrs)/batch_size))]
    for e in range(6):
        for i0, i in enumerate(batch):
            Xz = [cv2.imread(j[0]) for j in i]
            Xz = np.concatenate([np.expand_dims(cv2.resize(j, (125, 125)), 0) for j in Xz], 0)
            Yz = np.concatenate([np.asarray(np.load(j[1])) for j in i], 0)
            #print(Xz.shape, Yz.shape)
            _, cur_loss = sess.run([opti[0], loss], feed_dict={X0: Xz, Y: Yz})
            print(i0, 'of', len(batch), e, cur_loss)








