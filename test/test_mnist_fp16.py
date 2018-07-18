#!/usr/bin/env python3

# test_mnist_fp16.py 
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
import os

import progressbar as pb
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import waveflow


def test_mnist_fp16(use_tensorboard=False, arithmetic_type='tf'):
    ''' Run MNIST using the Wave custom matmul operator. 
    '''

    tf.reset_default_graph()

    waveflow.waveflow_arithmetic = arithmetic_type

    # Import data
    mnist = input_data.read_data_sets('./mnist_data')

    # Create the model
    x = tf.placeholder(tf.float16, [None, 784])
    W = tf.Variable(tf.zeros([784, 10], dtype=tf.float16))
    b = tf.Variable(tf.zeros([10], dtype=tf.float16))

    y_l1 = tf.cast(tf.matmul(x, W), tf.float16)
    y = tf.cast(tf.nn.bias_add(y_l1, b), tf.float16)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

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
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        avg_entropy = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(avg_entropy)
        if use_tensorboard: tf.summary.scalar("avg_loss", avg_entropy, family='Accuracy')

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        if use_tensorboard: tf.summary.scalar("accuracy", accuracy, family='Accuracy')


    if use_tensorboard:
        tb_log_dir = './tb_mnist_log/' + arithmetic_type
        print("Logging data to tensorboard, directory: %s" % (tb_log_dir))


    total_batches = 100

    widgets = ["Training: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=total_batches)
    pbar.start()

    # for shape debugging
    shape_op = tf.shape(y)

    with tf.Session('') as sess:
        print("Running model ...")
        if use_tensorboard: 
            tb_writer = tf.summary.FileWriter(tb_log_dir, sess.graph)
            tb_merged = tf.summary.merge_all()

            ops_for_train = [train_step, tb_merged]
        else:
            ops_for_train = train_step

        tf.global_variables_initializer().run()

        # Train
        for i in range(total_batches):
            pbar.update(i)
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # batch_xs = batch_xs.astype(tf.float16)
            train_results = sess.run(ops_for_train, feed_dict={x: batch_xs, y_: batch_ys})
            
            if use_tensorboard:
                tb_summary = train_results[1]
                tb_writer.add_summary(tb_summary, i)
                tb_writer.flush()


        pbar.finish()
        if use_tensorboard: tb_writer.close()

        # Compute training set accuracy
        print("Evaluating full training set ...")
        sys.stdout.flush()
        train_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        print("Train set accuracy: %s" % (train_accuracy))

        # Compute test set accuracy
        print("Evaluating full test set ...")
        sys.stdout.flush()
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Test set accuracy: %s" % (test_accuracy))


if __name__ == "__main__":
    test_mnist_fp16(use_tensorboard=True, arithmetic_type='tf')
