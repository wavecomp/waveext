#!/usr/bin/env python3

# sqnet_train.py 
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
#      Train Squeezenet on CIFAR10.
# 
# Author          : Ken Shiring
# Created On      : 03/05/2018
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

import os
import sys
import tarfile
import pickle
import argparse

from six.moves import urllib


import progressbar as pb

import numpy as np
import tensorflow as tf

import waveflow



def op_list(g):
  result = set()
  for o in g.get_operations():
    if type(o.op_def) is not type(None):
      print('node_def name: ', o.op_def.name)
      result.add(o.op_def.name)
  return result


def maybe_download_and_extract(data_dir, data_url):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def get_data_set(name="train", cifar=10):
    x = None
    y = None
    l = None

    print("Loading cifar10 dataset ...")
    sys.stdout.flush()
    # folder_name = "cifar_10" if cifar == 10 else "cifar_100"
    folder_name = "cifar-10-batches-py"

    f = open('./cifar10_data/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open('./cifar10_data/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./cifar10_data/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    # return x, dense_to_one_hot(y), l
    return x, y, l


def get_batch(train_x, train_y, bsize):
    randidx = np.random.randint(len(train_x), size=bsize)
    batch_xs = train_x[randidx]
    batch_ys = train_y[randidx]
    return batch_xs, batch_ys


def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
           name=None):
  return tf.layers.conv2d(
      inputs,
      filters,
      kernel_size,
      strides,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      activation=tf.nn.relu,
      name=name,
      padding="same")


def fire_module(inputs, squeeze_depth, expand_depth, name):
    """Fire module: squeeze input filters, then apply spatial convolutions."""
    with tf.variable_scope(name, "fire", [inputs]):
        squeezed = conv2d(inputs, squeeze_depth, [1, 1], name="squeeze")
        e1x1 = conv2d(squeezed, expand_depth, [1, 1], name="e1x1")
        e3x3 = conv2d(squeezed, expand_depth, [3, 3], name="e3x3")
        return tf.concat([e1x1, e3x3], axis=3)


def squeezenet(images, num_classes=1001):
    """Squeezenet 1.0 model."""
    x_image = tf.reshape(images, [-1, 32, 32, 3])
    net = conv2d(x_image, 96, [7, 7], strides=(2, 2), name="conv1")
    net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool1")
    net = fire_module(net, 16, 64, name="fire2")
    net = fire_module(net, 16, 64, name="fire3")
    net = fire_module(net, 32, 128, name="fire4")
    net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool4")
    net = fire_module(net, 32, 128, name="fire5")
    net = fire_module(net, 48, 192, name="fire6")
    # CIFAR10 is too small for these layers
    # net = fire_module(net, 48, 192, name="fire7")
    # net = fire_module(net, 64, 256, name="fire8")
    # net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool8")
    # net = fire_module(net, 64, 256, name="fire9")
    drop_rate = tf.placeholder(tf.float32)
    net = tf.layers.dropout(net, rate=drop_rate, name="drop9")
    net = conv2d(net, num_classes, [1, 1], strides=(1, 1), name="conv10")
    net = tf.layers.average_pooling2d(net, pool_size=(3, 3), strides=(1, 1))
    logits = tf.layers.flatten(net)
    return logits, drop_rate


def train_sqnet(use_tensorboard=False, use_trace=False, arithmetic_type='tf'):
    ''' Run Squeezenet
    '''
    batch_size = 128

    waveflow.waveflow_arithmetic = arithmetic_type

    tf.reset_default_graph()
    # Import data
    data_dir = './cifar10_data'
    maybe_download_and_extract(data_dir=data_dir, data_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    train_x, train_y, labels = get_data_set(name="train")

    # Create the model
    image_size = 32
    # x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    x = tf.placeholder(tf.float32, [None, image_size*image_size*3])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv, drop_rate = squeezenet(x, num_classes=10)

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


    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        if use_tensorboard: tf.summary.scalar("dropout_accuracy", accuracy, family='Accuracy')


    tb_log = waveflow.TF_TBLogger(log_dir='./tb_sqnet_cf10_log', enable_tb=use_tensorboard, 
        enable_trace=use_trace, unified_trace=True, arith_type=arithmetic_type)


    # if use_tensorboard:
    if False:
        # Here we add additional logging data for visualization. This captures the average
        # momentum for each trainable param when Momentum optimizers are used, which shows
        # how networks are converging over time.
        for v in tf.trainable_variables():
            m_v = optimizer.get_slot(v, 'm')
            # print("var: %s, data: %s" % (v.name, m_v))
            with tf.name_scope('visualization'):
                v_mean = tf.reduce_mean(m_v)
                tf.summary.scalar("%s" % (v.name), v_mean, family='Momentum')


    # print('nodes with trainng')
    # print('op list: ', op_list(tf.get_default_graph()))


    total_batches = 100
    report_interval = 10
    epochs = len(train_x) / total_batches

    print("Batch size: %d, Num batches: %d, epochs: %.1f" % (batch_size, total_batches, epochs))

    widgets = ["Training: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=total_batches)
    # pbar.start()

    # Turn off graph-rewriting optimizations
    # config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))


    # with tf.Session('', config=config) as sess:
    with tf.Session('') as sess:
        print("Running model ...")
        tb_log.init_session(sess)

        tf.global_variables_initializer().run()

        # Train
        for i in range(total_batches):
            # pbar.update(i)
            batch_xs, batch_ys = get_batch(train_x, train_y, batch_size)
            tb_log.run_session(iter=i, ops=train_step, feed={x: batch_xs, y_: batch_ys, drop_rate: 0.5})

            if i % report_interval == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, drop_rate: 1.0})
                print('step %d, training accuracy %.4f' % (i, train_accuracy))
                if use_tensorboard: tf.summary.scalar("accuracy", train_accuracy, family='Accuracy')


        # pbar.finish()
        tb_log.close()

        # Compute training set accuracy
        print("Evaluating full training set ...")
        sys.stdout.flush()
        train_accuracy = accuracy.eval(feed_dict={x: train_x, y_: train_y, drop_rate: 1.0})
        print("Train set accuracy: %s" % (train_accuracy))
        

        print("Evaluating full test set ...")
        sys.stdout.flush()
        test_x, test_y, labels = get_data_set(name="test")
        # Compute test set accuracy
        test_accuracy = accuracy.eval(feed_dict={x: test_x, y_: test_y, drop_rate: 1.0})
        print("Test set accuracy: %s" % (test_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Squeezenet on CIFAR10 example')
    parser.add_argument("-tb", "--tensorboard", action='store_true', help='Generate Tensorboard data')
    parser.add_argument("-tr", "--trace", action='store_true', help='Generate execution trace')
    parser.add_argument("-a",  "--arith", type=str, default='tf', help='Arithmetic type')

    args = parser.parse_args()

    train_sqnet(use_tensorboard=args.tensorboard, use_trace=args.trace, arithmetic_type=args.arith)

