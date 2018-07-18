#!/usr/bin/env python3

# test_avgpool2d.py
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
#      Unit test for BP AvgPool2d
#
# Author          : Nemanja Popov
# Created On      : 05/15/2018
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

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import nn_ops

import waveflow


def compare_tensor(z1, z2, msg):
    ''' Run a compare on 2 tensors for equality. Report failure details.
    '''
    assert z1.shape == z2.shape, msg
    rtol = 1e-4
    if not np.allclose(z1, z2, atol=rtol):
        print("\n\n")
        d = ~np.isclose(z1, z2, atol=rtol)
        print("z1 mismatch: %s" % (z1[d]))
        print("z2 mismatch: %s" % (z2[d]))
        print("at: %s" % (str(np.where(d))))
        print("Failure: %s" % (msg))
        return False

    return True


def test_avgpool2d():
    ''' Run tests on the Wave custom avgpool2d operator.
    '''
    tf.reset_default_graph()
    # Turn off graph-rewriting optimizations
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

    iterations = 100

    for i in range(iterations):
        tf.reset_default_graph()

        # NCHW
        t_n = 1
        t_h = 64
        t_w = 64
        t_c = 2

        # window
        w_n = 1
        w_h = 2
        w_w = 2
        w_c = 1

        #strides
        s_n = 1
        s_h = 2
        s_w = 2
        s_c = 1

        # N H W C
        max_in = tf.get_variable("a", [t_n, t_h, t_w, t_c], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))

        t_init = tf.global_variables_initializer()

        # SAME variant
        with tf.Session('', config=config) as sess:
            t_init.run()

            # print("Wave Kernel:\n-------------------------------------------------")

            z_op = waveflow.wavecomp_ops_module.wave_avg_pool_dfx(
                max_in,
                ksize=[w_n, w_h, w_w, w_c],
                strides=[s_n, s_h, s_w, s_c],
                padding='SAME',
                data_format='NHWC')

            # Base tensorflow. Only supports NHWC.
            z2_op = nn_ops.avg_pool(
                max_in,
                ksize=[w_n, w_h, w_w, w_c],
                strides=[s_n, s_h, s_w, s_c],
                padding='SAME',
                data_format='NHWC')

            # z = z_op.eval()
            # z2 = z2_op.eval()
            z, z2 = sess.run([z_op, z2_op])

            # print("\nTF:\n-------------------------------------------------")

            assert_str = "Failure on i: %d, mode: SAME" % (i)
            if not compare_tensor(z, z2, assert_str):
                print("z: shape: %s, %s" % (z.shape, z))
                print("z (np): shape: %s, %s" % (z2.shape, z2))
                print("\n\n")
                assert False

        # Valid variant
        with tf.Session('', config=config) as sess:
            t_init.run()

            # print("Wave Kernel:\n-------------------------------------------------")

            z_op = waveflow.wavecomp_ops_module.wave_avg_pool_dfx(
                max_in,
                ksize=[w_n, w_h, w_w, w_c],
                strides=[s_n, s_h, s_w, s_c],
                padding='VALID',
                data_format='NHWC')

            # Base tensorflow. Only supports NHWC.
            z2_op = nn_ops.avg_pool(
                max_in,
                ksize= [w_n, w_h, w_w, w_c],
                strides=[s_n, s_h, s_w, s_c],
                padding='VALID',
                data_format='NHWC')


            z, z2 = sess.run([z_op, z2_op])
            # print("\nTF:\n-------------------------------------------------")

            assert_str = "Failure on i: %d, mode: VALID" % (i)
            if not compare_tensor(z, z2, assert_str):
                print("z: shape: %s, %s" % (z.shape, z))
                print("z (np): shape: %s, %s" % (z2.shape, z2))
                print("\n\n")
                assert False

    return True


if __name__ == "__main__":
    test_avgpool2d()