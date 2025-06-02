# cnn 1 
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load data
mnist = input_data.read_data_sets('data',one_hot=True)
# input
imageInput = tf.placeholder(tf.float32,[None,784])
labelInput = tf.placeholder(tf.float32,[None,10]) #label

# data reshape
imageInputReshape = tf.reshape(imageInput,[-1,28,28,1]) # M * 28 * 28 * 1

# W0 = knerl 5 * 5 oup 32 in:1
w0 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b0 = tf.Variable(tf.constant(0.1,shape=[32]))
# layer1
layer1 = tf.nn.relu(tf.nn.conv2d(imageInputReshape,w0,strides=[1,1,1,1],padding='SAME') + b0)
# M * 28 * 28 * 32

# pool sample data reduce M * 28 * 28 * 32 => M * 7 * 7 * 32
layer1_pool = tf.nn.max_pool(layer1,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')

# layer2 out
w1 = tf.Variable(tf.truncated_normal([7*7*32,1024],stddev=0.1))
b1 = tf.Variable(tf.constant(0.1,shape=[1024]))

h_reshape = tf.reshape(layer1_pool,[-1,7*7*32]) #N*N1
h1 = tf.nn.relu(tf.matmul(h_reshape,w1) + b1)


# softmax
w2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b2 = tf.Variable(tf.constant(0.1,shape=[10]))
pred = tf.nn.softmax(tf.matmul(h1,w2) + b2)

# N * 10

loss0 = labelInput * tf.log(pred)
loss1 = 0
for m in range(0,100):
    for n in range(0,10):
        loss1 = loss1 - loss0[m,n]
loss = loss1/100

# train
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        images,labels = mnist.train.next_batch(500)
        sess.run(train,feed_dict={imageInput:images,labelInput:labels})

        pred_test = sess.run(pred,feed_dict={imageInput:mnist.test.images,labelInput:labels})

        acc = tf.equal(tf.arg_max(pred_test,1),tf.arg_max(mnist.test.labels,1))
        acc_float = tf.reduce_mean(tf.cast(acc,tf.float32))
        acc_result = sess.run(acc_float,feed_dict={imageInput:mnist.test.images,labelInput:mnist.test.labels})
        print(acc_result)