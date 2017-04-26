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
              'conv5_1': tf.Variable(tf.random_normal([3, 3, 512, 512])),
              'conv5_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
              'conv5_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),
              'fc6': tf.Variable(tf.random_normal([3, 3, 512, 1024])),
              'fc7': tf.Variable(tf.random_normal([1, 1, 1024, 1024])),
              'conv6_1': tf.Variable(tf.random_normal([1, 1, 1024, 256])),
              'conv6_2': tf.Variable(tf.random_normal([3, 3, 256, 512])),
              'conv7_1': tf.Variable(tf.random_normal([1, 1, 512, 128])),
              'conv7_2': tf.Variable(tf.random_normal([3, 3, 128, 256])),
              'conv8_1': tf.Variable(tf.random_normal([1, 1, 256, 128])),
              'conv8_2': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        }
        # Biases
        B = {'conv1_1': tf.Variable(tf.random_normal([64])),
                  'conv1_2': tf.Variable(tf.random_normal([64])),
                  'conv2_1': tf.Variable(tf.random_normal([128])),
                  'conv2_2': tf.Variable(tf.random_normal([128])),
                  'conv3_1': tf.Variable(tf.random_normal([256])),
                  'conv3_2': tf.Variable(tf.random_normal([256])),
                  'conv3_3': tf.Variable(tf.random_normal([256])),
                  'conv4_1': tf.Variable(tf.random_normal([512])),
                  'conv4_2': tf.Variable(tf.random_normal([512])),
                  'conv4_3': tf.Variable(tf.random_normal([512])),
                  'conv5_1': tf.Variable(tf.random_normal([512])),
                  'conv5_2': tf.Variable(tf.random_normal([512])),
                  'conv5_3': tf.Variable(tf.random_normal([512])),
                  'fc6': tf.Variable(tf.random_normal([1024])),
                  'fc7': tf.Variable(tf.random_normal([1024])),
                  'conv6_1': tf.Variable(tf.random_normal([256])),
                  'conv6_2': tf.Variable(tf.random_normal([512])),
                  'conv7_1': tf.Variable(tf.random_normal([128])),
                  'conv7_2': tf.Variable(tf.random_normal([256])),
                  'conv8_1': tf.Variable(tf.random_normal([128])),
        }

    def conv2d(self, x, W, b, strides=1, dilation=1):
        ''' Conv2D wrapper, with bias and relu activation '''
        if dilation == 1:
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        else:
            x = tf.nn.atrous_conv2d(x, W, rate=dilation, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def globalavgpool2d(self, x):
        ''' Global Avg Pool 2D wrapper '''
        return tf.nn.avg_pool(x, [n_height, n_width], [n_height, n_width])

    def maxpool2d(self, x, k=2, strides=-1):
        ''' MaxPool2D wrapper '''
        if strides==-1:
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        else:
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='SAME')

    def net(self):
        conv1_1 = conv2d(x, W['conv1_1'], B['conv1_1'])
        conv1_2 = conv2d(conv1_1, W['conv1_2'], B['conv1_2'])
        conv1 = maxpool2d(conv1_2)

        conv2_1 = conv2d(conv1, W['conv2_1'], B['conv2_1'])
        conv2_2 = conv2d(conv2_1, W['conv2_2'], B['conv2_2'])
        conv2 = maxpool2d(conv2_2)

        conv3_1 = conv2d(conv2, W['conv3_1'], B['conv3_1'])
        conv3_2 = conv2d(conv3_1, W['conv3_2'], B['conv3_2'])
        conv3_3 = conv2d(conv3_2, W['conv3_3'], B['conv3_3'])
        conv3 = maxpool2d(conv3_3)

        conv4_1 = conv2d(conv3, W['conv4_1'], B['conv4_1'])
        conv4_2 = conv2d(conv4_1, W['conv4_2'], B['conv4_2'])
        conv4_3 = conv2d(conv4_2, W['conv4_3'], B['conv4_3'])
        conv4 = maxpool2d(conv4_3)

        conv5_1 = conv2d(conv4, W['conv5_1'], B['conv5_1'])
        conv5_2 = conv2d(conv5_1, W['conv5_2'], B['conv5_2'])
        conv5_3 = conv2d(conv5_2, W['conv5_3'], B['conv5_3'])
        conv5 = maxpool2d(conv5_3, k=3, strides=1)

        fc6 = conv2d(conv5, W['fc6'], B['fc6'], dilation=6)
        fc7 = conv2d(fc6, W['fc7'], B['fc7'])

        conv6_1 = conv2d(fc7, W['conv6_1'], B['conv6_1'])
        conv6_2 = conv2d(conv6_1, W['conv6_2'], B['conv6_2'], strides=2)

        conv7_1 = conv2d(conv6_2, W['conv7_1'], B['conv7_1'])
        conv7_2 = conv2d(conv7_1, W['conv7_2'], B['conv7_2'], strides=2)

        conv8_1 = conv2d(conv7_2, W['conv8_1'], B['conv8_1'])
        conv8_2 = conv2d(conv8_1, W['conv8_2'], B['conv8_2'], strides=2)
        pool6 = globalavgpool2d(conv8_2)

    def train(self, train_data):
        print "Training is done."

    def test(self, test_data):
        print "Accuracy is NaN"
