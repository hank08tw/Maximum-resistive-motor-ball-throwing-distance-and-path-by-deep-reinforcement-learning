"""
import numpy as np
a = np.array([1,2,3])
b = np.array([1,2,3])[np.newaxis,:]
print (a.shape,'\n',a)
print (b.shape,'\n',b)
fd = open('./data1.txt','w',encoding='utf-8')
fd.write('cool\n')
fd.write('cool2')
"""
#import random
#print(random.random())
"""
import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b
# 通过log_device_placement参数来输出运行每一个运算的设备。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
"""
import tensorflow as tf

# 通过tf.device将运算指定到特定的设备上。
with tf.device('/cpu:0'):
   a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
   b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:1'):
    c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

