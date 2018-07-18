#!/usr/bin/env python3

# test_conv_mnist.py 
# 
# Copyright (c) 2010-2018 Wave Computing, Inc. and its applicable licensors.   
# All rights reserved; provided, that any files identified as open source shall
# be governed by the specific open source license(s) applicable to such files. 
#
# For any files associated with distributions under the Apache 2.0 license, 
# full attribution to The Apache Software Foundation is given via the license 
# below.
#
# PURPOSE
#      Run the MNIST example with custom operations.
# 
# Author          : Ken Shiring
# Created On      : 02/13/2018
# 
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 # 
 #     http://www.apache.org/licenses/LICENSE-2.0
 # 
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.


import sys
import pytest
import progressbar as pb
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

import waveflow



def test_cmnist_tf():
    a = 'tf'
    print("Running Conv-MNIST with arithmetic=%s" % (a))
    acc = run_conv_mnist(use_tensorboard=False, use_trace=False, arithmetic_type=a)
    assert acc > 0.85
    return True

'''
# Too slow to run these in the regression (for now).
def test_cmnist_wf():
    a = 'wf'
    print("Running Conv-MNIST with arithmetic=%s" % (a))
    return run_conv_mnist(use_tensorboard=False, use_trace=False, arithmetic_type=a)

def test_cmnist_dfx():
    a = 'dfx'
    print("Running Conv-MNIST with arithmetic=%s" % (a))
    return run_conv_mnist(use_tensorboard=False, use_trace=False, arithmetic_type=a)

def test_cmnist_int():
    a = 'int'
    print("Running Conv-MNIST with arithmetic=%s" % (a))
    return run_conv_mnist(use_tensorboard=False, use_trace=False, arithmetic_type=a)
'''



def op_list(g):
  result = set()
  for o in g.get_operations():
    if type(o.op_def) is not type(None):
        print("Got opdef: %s" % (str(o.op_def)))
        # print('node_def name: ', o.op_def.name)
        result.add(o.op_def.name)
  return result


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # variable_summaries(W_conv1)
    cv_conv1 = tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1)
    h_conv1 = tf.nn.relu(cv_conv1)

  # Pooling layer - downsamples to 14x14
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    cv_conv2 = tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2)
    h_conv2 = tf.nn.relu(cv_conv2)

  # Second pooling layer - downsamples to 7x7
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    mm_fc1 = tf.nn.bias_add(tf.matmul(h_pool2_flat, W_fc1), b_fc1)
    h_fc1 = tf.nn.relu(mm_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.bias_add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  # return wavecomp_ops_module.wave_conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_w(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  # return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  return wavecomp_ops_module.wave_conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def run_conv_mnist(use_tensorboard=False, use_trace=False, arithmetic_type='tf'):
    ''' Run MNIST using the Wave custom matmul operator. 
    '''
    waveflow.waveflow_arithmetic = arithmetic_type

    tf.reset_default_graph()
    # Import data
    mnist = input_data.read_data_sets('./mnist_data')

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)


    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
    # outputs of 'y', and then average across the batch.
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
        avg_entropy = tf.reduce_mean(cross_entropy)
        if use_tensorboard: tf.summary.scalar("avg_loss", avg_entropy, family='Accuracy')

    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    with tf.name_scope('adam_optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_step = optimizer.minimize(avg_entropy)
        # tf.summary.scalar("learning_rate", optimizer._lr_t)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        if use_tensorboard: tf.summary.scalar("accuracy", accuracy, family='Accuracy')

    '''
    with tf.name_scope('metrics'):
        y_prob = tf.one_hot(y_, 10)
        _, recall = tf.metrics.recall(y_conv, y_prob)
        _, precision = tf.metrics.precision(y_conv, y_prob)

        tf.summary.scalar('recall', recall)
        tf.summary.scalar('precision', precision)
    '''

    tb_log = waveflow.TF_TBLogger(log_dir='./tb_conv_mnist_log/', enable_tb=use_tensorboard, 
        enable_trace=use_trace, unified_trace=use_trace, arith_type=arithmetic_type)

    # print("vars:")
    if use_tensorboard:
        for v in tf.trainable_variables():
            m_v = optimizer.get_slot(v, 'm')
            # print("var: %s, data: %s" % (v.name, m_v))
            v_mean = tf.reduce_mean(m_v)
            tf.summary.scalar("%s" % (v.name), v_mean, family='Momentum')


    # print('nodes with trainng')
    # print('op list: ', op_list(tf.get_default_graph()))

    batch_size = 128
    total_batches = 100
    report_interval = 1

    widgets = ["Training: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=total_batches)
    # pbar.start()

    config = tf.ConfigProto(device_count={"CPU": 16},
                            inter_op_parallelism_threads=2,
                            intra_op_parallelism_threads=16)

    with tf.Session('', config=config) as sess:
        print("Running model ...")
        tb_log.init_session(sess)

        tf.global_variables_initializer().run()

        # Train
        for i in range(total_batches):
            # pbar.update(i)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            tb_log.run_session(iter=i, ops=train_step, feed={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

            if i % report_interval == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

        # pbar.finish()
        tb_log.close()

        # Compute training set accuracy
        print("Evaluating full training set ...")
        sys.stdout.flush()
        train_accuracy = accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0})
        print("Train set accuracy: %s" % (train_accuracy))

        # Compute test set accuracy
        print("Evaluating full test set ...")
        sys.stdout.flush()
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("Test set accuracy: %s" % (test_accuracy))

    return test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run convolutional-MNIST test')
    parser.add_argument("-tb", "--tensorboard", action='store_true', help='Generate Tensorboard data')
    parser.add_argument("-tr", "--trace", action='store_true', help='Generate execution trace')
    parser.add_argument("-a",  "--arith", type=str, default='tf', help='Arithmetic type')

    args = parser.parse_args()

    run_conv_mnist(use_tensorboard=args.tensorboard, use_trace=args.trace, arithmetic_type=args.arith)
