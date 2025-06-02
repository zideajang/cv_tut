import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load data
mnist = input_data.read_data_sets('data',one_hot=True)
# attributes
trainNum = 55000
testNum = 10000
trainSize = 500
testSize = 5
k = 4
# data random range 0 - trainNum replace no repeat
trainIndex = np.random.choice(trainNum,trainSize,replace=False)
testIndex = np.random.choice(testNum,testSize,replace=False)
trainData = mnist.train.images[trainIndex] # train images
trainLabel = mnist.train.labels[trainIndex] #
testData = mnist.test.images[testIndex]
testLabel = mnist.test.labels[testIndex]

print('trainData.shape=',trainData.shape)
print('trainLabel.shape=',trainLabel.shape)
print('testData.shape=',testData.shape)
print('testLabel.shape=',testLabel.shape)

# tf input
trainDataInput = tf.placeholder(shape=[None,784],dtype=tf.float32)
trainLabelInput = tf.placeholder(shape=[None,10],dtype=tf.float32)

testDataInput = tf.placeholder(shape=[None,784],dtype=tf.float32)
testLabelInput = tf.placeholder(shape=[None,10],dtype=tf.float32)

# knn distance 5 * 784 5 * 1 * 784
# 5 500 784 2500 * 784 
f1 = tf.expand_dims(testDataInput,1) # dimension expand
f2 = tf.subtract(trainDataInput,f1) # 784 sum()
# 5 * 500
f3 = tf.reduce_sum(tf.abs(f2),reduction_indices=2) #

# 5 * 500
f5 = tf.negative(f3)
# f5 maxvalue 4 value
f6,f7 = tf.nn.top_k(f5,k=4)

# index->pictures
f8 = tf.gather(trainLabelInput,f7)

f9 = tf.reduce_sum(f8,reduction_indices=1)
f10 = tf.argmax(f9,dimension=1)

with tf.Session() as sess:
    # f1 testData
    p1 = sess.run(f1,feed_dict={testDataInput:testData[0:5]})
    print('p1=',p1.shape)
    p2 = sess.run(f2,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print('p2=',p2.shape)
    # print('p2=',p2[1][100])
    p3 = sess.run(f3,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print('p3=',p3.shape)
    print('',p3[0][0])
    p5 = sess.run(f5,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print('p5=',p5.shape)
    print('',p5[0][0])

    p6,p7 = sess.run((f6,f7),feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print('p6=',p6.shape) # 
    print('p7=',p7.shape)
    print('p6[0][0]',p6[0][0])
    print('p7[0][0]',p7[0][0]) #index

    p8 = sess.run(f8,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print('p8=',p8.shape)
    print('p8[]',p8)
    p9 = sess.run(f9,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print('p9=',p9.shape)
    print('p9[]',p9)

    p10 = sess.run(f10,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print('p10=',p10.shape)
    print('p10[]',p10)

    expect = np.argmax(testLabel[0:5],axis=1)
    print('expect',expect)
j = 0
for i in range(0,5):
    if expect[i] == p10[i]:
        j = j+1
print('ac =', j* 100/5)
