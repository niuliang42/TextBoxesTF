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
        # n_classes = 10 # MNIST total classes (0-9 digits)
        dropout = 0.75 # Dropout, probability to keep units
        W = { 'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
                'out': tf.Variable(tf.random_normal([1024, n_classes]))
                }


