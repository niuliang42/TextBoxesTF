'''
Author: Liang Niu ln932@nyu.edu
'''

from __future__ import print_function

import tensorflow as tf

class TextBoxes():
    def __init__(self):
        # Network Parameters
        n_width = 700
        n_height = 700
        n_input = n_width * n_height

        # tf Graph input
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # weights
        W = { 'conv1_1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
              'conv1_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
              'conv2_1': tf.Variable(tf.random_normal([3, 3, 64, 128])),
              'conv2_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
              'conv3_1': tf.Variable(tf.random_normal([3, 3, 128, 256])),
              'conv3_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
              'conv3_3': tf.Variable(tf.random_normal([3, 3, 256, 256])),
              'conv4_1': tf.Variable(tf.random_normal([3, 3, 256, 512])),
              'conv4_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
              'conv4_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),
              'out': tf.Variable(tf.random_normal([1024, n_classes])) }
        biases = {'conv1_1': tf.Variable(tf.random_normal([64])),
                  'conv1_2': tf.Variable(tf.random_normal([64])),
                  'conv2_1': tf.Variable(tf.random_normal([128])),
                  'conv2_2': tf.Variable(tf.random_normal([128])),
                  'conv3_1': tf.Variable(tf.random_normal([256])),
                  'conv3_2': tf.Variable(tf.random_normal([256])),
                  'conv3_3': tf.Variable(tf.random_normal([256])),
                  'conv4_1': tf.Variable(tf.random_normal([512])),
                  'conv4_2': tf.Variable(tf.random_normal([512])),
                  'conv4_3': tf.Variable(tf.random_normal([512])),
                  'out': tf.Variable(tf.random_normal([1024])),}

    def conv2d(self, x, W, b, strides=1):
        ''' Conv2D wrapper, with bias and relu activation '''
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(self, x, k=2):
        ''' MaxPool2D wrapper '''
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='SAME')

    def net(self):
        # VGG16 (conv1_1 to conv4_3)
        conv1_1 = conv2d(x, W['conv1_1'], biases['conv1_1'])
        conv1_2 = conv2d(conv1_1, W['conv1_2'], biases['conv1_2'])
        conv1 = maxpool2d(conv1_2)

        conv2_1 = conv2d(conv1, W['conv2_1'], biases['conv2_1'])
        conv2_2 = conv2d(conv2_1, W['conv2_2'], biases['conv2_2'])
        conv2 = maxpool2d(conv2_2)

        conv3_1 = conv2d(conv2, W['conv3_1'], biases['conv3_1'])
        conv3_2 = conv2d(conv3_1, W['conv3_2'], biases['conv3_2'])
        conv3_3 = conv2d(conv3_2, W['conv3_3'], biases['conv3_3'])
        conv3 = maxpool2d(conv3_3)

        conv4_1 = conv2d(conv3, W['conv4_1'], biases['conv4_1'])
        conv4_2 = conv2d(conv4_1, W['conv4_2'], biases['conv4_2'])
        conv4_3 = conv2d(conv4_2, W['conv4_3'], biases['conv4_3'])
        conv4 = maxpool2d(conv4_3)

    def train(self):
        pass
