#!/usr/bin/env python3

# test_lstm.py 
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
#      Unit test for FP LSTM cell
# 
# Author          : Ken Shiring
# Created On      : 04/16/2018
# 
# This test was taken from Monik Pamecha's online LSTM example:
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# It has been rewritten for clarity.
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

from collections import namedtuple
from random import shuffle
import progressbar as pb
import argparse


import numpy as np
import tensorflow as tf


import waveflow

def test_lstm_tf():
    a = 'tf'
    print("Running LSTM with arithmetic=%s" % (a))
    err = run_lstm(use_tensorboard=False, use_trace=False, arithmetic_type=a)
    assert err < 0.15
    return True

def test_lstm_wf():
    a = 'wf'
    print("Running LSTM with arithmetic=%s" % (a))
    err = run_lstm(use_tensorboard=False, use_trace=False, arithmetic_type=a)
    assert err < 0.15
    return True

def test_lstm_dfx():
    a = 'dfx'
    print("Running LSTM with arithmetic=%s" % (a))
    err = run_lstm(use_tensorboard=False, use_trace=False, arithmetic_type=a)
    assert err < 0.15
    return True

def test_lstm_int():
    a = 'int'
    print("Running LSTM with arithmetic=%s" % (a))
    err = run_lstm(use_tensorboard=False, use_trace=False, arithmetic_type=a)
    assert err < 0.15
    return True


def generate_data(gen_examples, input_len=20):
    print("Generating data ...")

    rand_ints = np.arange(2**input_len, dtype=np.int32)
    np.random.shuffle(rand_ints)
    # Fill up the desired array if the input_len is small
    desired_data = 2 * gen_examples
    if len(rand_ints) < desired_data:
        rep = max(int(desired_data / len(rand_ints)), 2)
        train_ints = np.tile(rand_ints, rep)
    else:
        train_ints = rand_ints
    print("Generating %d samples" % (len(train_ints)))

    train_input = np.zeros(shape=(len(train_ints), input_len, 1), dtype=np.float32)
    for i in train_ints:
        for j in range(input_len):
            train_input[i, j, 0] = float(int((i >> j) & 1))

    # Onehot outputs representing bit-count
    train_output = np.zeros(shape=(len(train_ints), input_len+1), dtype=np.float32)
    for i in train_ints:
        bits = int(bin(i).count("1"))
        train_output[i, bits] = 1.

    test_input = train_input[gen_examples:]
    test_output = train_output[gen_examples:]
    train_input = train_input[:gen_examples]
    train_output = train_output[:gen_examples]

    print("test and training data loaded")
    return (train_input, train_output, test_input, test_output)


class ProgressTrain(pb.Widget):

    def __init__(self):
        self._ep = 0
        self._err = 1.0

    def update(self, pbar):
        return "Epoch: %d, Error: %.4f" % (self._ep, self._err)



def run_lstm(use_tensorboard=False, use_trace=False, arithmetic_type='tf'):
    ''' Run tests on the Wave custom lstm operator. 
    '''
    waveflow.waveflow_arithmetic = arithmetic_type

    tf.reset_default_graph()

    NUM_EXAMPLES = 10000
    input_len = 14
    train_input, train_output, test_input, test_output = generate_data(NUM_EXAMPLES, input_len)

    # print("Data:")
    # print("x: %s" % (train_input[:16]))
    # print("y: %s" % (train_output[:16]))

    data = tf.placeholder(tf.float32, [None, input_len, 1]) #Number of examples, number of input, dimension of each input
    target = tf.placeholder(tf.float32, [None, input_len+1])
    num_hidden = 8
    num_outputs = int(target.get_shape()[1])

    with tf.name_scope('lstm_layer'):
        cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        lstm_out = tf.nn.batch_normalization(last, mean=0., variance=1., offset=None, scale=None, variance_epsilon=1e-5)

    if True:
        with tf.name_scope('fc_layer'):
            weight = tf.Variable(tf.truncated_normal([num_hidden, num_outputs]))
            bias = tf.Variable(tf.constant(0.1, shape=[num_outputs]))
            logits = tf.matmul(lstm_out, weight) + bias
    else:
        # Experiment to see if outputs can be generated from just the LSTM.
        logits = lstm_out

    with tf.name_scope('loss'):
        prediction = tf.nn.softmax(logits)
        pred_clip = tf.clip_by_value(prediction,1e-10,1.0)
        cross_entropy = -tf.reduce_sum(target * tf.log(pred_clip))
        # cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=target, logits=pred_clip)

    with tf.name_scope('loss_optimizer'):
        optimizer = tf.train.AdamOptimizer()
        minimize = optimizer.minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.global_variables_initializer()

    tb_log = waveflow.TF_TBLogger(log_dir='./tb_lstm_log/', enable_tb=use_tensorboard, 
        enable_trace=use_trace, unified_trace=False, arith_type=arithmetic_type)

    batch_size = 32
    print("Input set: %s" % (str(train_input.shape)))
    no_of_batches = int(train_input.shape[0] / batch_size)
    epoch = 50

    print("Batch size: %d, Num batches: %d, epochs: %d" % (batch_size, no_of_batches, epoch))
    print("Got op list: %s" % (waveflow.op_list(tf.get_default_graph())))

    pb_train = ProgressTrain()

    widgets = ["Training lstm; ", pb_train, ' ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]

    pbar = pb.ProgressBar(widgets=widgets, maxval=epoch)
    pbar.start()

    # CPU is faster on this small example
    # dev = '/device:GPU:0'
    dev = '/cpu:0'

    with tf.Session() as sess:
        with sess.graph.device(dev):
            print("Running model ...")
            tb_log.init_session(sess)

            sess.run(init_op)

            for i in range(epoch):
                pbar.update(i)
                ptr = 0
                for j in range(no_of_batches):
                    inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
                    ptr += batch_size
                    train_entropy = sess.run(minimize, feed_dict={data: inp, target: out})
                    # KS: This call is bugged and I can't figure out why. The error doesn't describe
                    # the actual problem.
                    # tb_log.run_session(iter=j, ops=minimize, feed={data: inp, target: out})

                # Every epoch, show the current error on the training set.
                incorrect = sess.run(error, feed_dict={data: inp, target: out})
                pb_train._ep = i
                pb_train._err = incorrect
                # print("Epoch %d, error: %.4f" % (i, incorrect))

            pbar.finish()
            tb_log.close()
            '''
            no_of_test_batches = int(len(test_input) / batch_size)
            ptr = 0
            for i in range(no_of_test_batches):
                inp, out = test_input[ptr:ptr+batch_size], test_output[ptr:ptr+batch_size]
                ptr += batch_size
                incorrect = sess.run(error, feed_dict={data: inp, target: out})
                print('Test set error: %3.1f' % (100 * incorrect))
            '''

            test_len = 1000
            inp, out = test_input[:test_len], test_output[:test_len]
            incorrect = sess.run(error, feed_dict={data: inp, target: out})
            print('Test set error: %3.1f' % (100 * incorrect))

            sess.close()

    return incorrect



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simple LSTM test')
    parser.add_argument("-tb", "--tensorboard", action='store_true', help='Generate Tensorboard data')
    parser.add_argument("-tr", "--trace", action='store_true', help='Generate execution trace')
    parser.add_argument("-a",  "--arith", type=str, default='tf', help='Arithmetic type')

    args = parser.parse_args()

    run_lstm(use_tensorboard=args.tensorboard, use_trace=args.trace, arithmetic_type=args.arith)

