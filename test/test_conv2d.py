#!/usr/bin/env python3

# test_conv2d.py 
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
#      Unit test for FP Conv2D
# 
# Author          : Ken Shiring
# Created On      : 02/26/2018
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

import progressbar as pb

import waveflow


def compare_tensor(z1, z2, msg):
    ''' Run a compare on 2 tensors for equality. Report failure details.
    '''
    # assert z1.shape == z2.shape, msg
    if z1.shape != z2.shape:
        print(msg)
        print("z1 shape: %s, z2 shape: %s" % (str(z1.shape), str(z2.shape)))
        return False

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


def conv2d_test(config, t_init, i, p, activations, c2d_wts, stride, padding):
    with tf.Session('', config=config) as sess:
        t_init.run()
        
        # print("Wave Kernel (NN):\n-------------------------------------------------")
        z_op = waveflow.wavecomp_ops_module.wave_conv2d(activations, c2d_wts, strides=[1, stride, stride, 1], padding=padding)
        
        # Base tensorflow. Only supports NHWC.
        z2_op = tf.nn.conv2d(activations, c2d_wts,
            strides=[1, stride, stride, 1], padding=padding, data_format='NHWC', use_cudnn_on_gpu=False)

        z, z2, act_val, wts_val = sess.run([z_op, z2_op, activations, c2d_wts])
        # print("\nTF:\n-------------------------------------------------")

        assert_str = "Failure on i: %d, mode: SAME, params: %s" % (i, str(p))
        if not compare_tensor(z, z2, assert_str):
            print("activations: %s" % (act_val))
            print("c2d_wts: %s" % (wts_val))
            print("\n\n")
            assert False


def test_conv2d():
    ''' Run tests on the Wave custom conv2d operator. 
    '''
    tf.reset_default_graph()

    # Turn off graph-rewriting optimizations
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

    iterations = 10

    widgets = ["conv2d tests: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=iterations)
    pbar.start()


    # Interesting kernel variants to cycle through
    kernel_params = [
        {'t_n':100, 't_ci':1,  't_co':32,  't_h':28,  't_w':28,  'w_k':5},
        {'t_n':4,   't_ci':32, 't_co':32,  't_h':15,  't_w':15,  'w_k':3},
        {'t_n':1,   't_ci':4,  't_co':64,  't_h':16,  't_w':16,  'w_k':3},
        {'t_n':128, 't_ci':64, 't_co':128, 't_h':7,   't_w':7,   'w_k':5},
        {'t_n':4,   't_ci':8,  't_co':4,   't_h':224, 't_w':224, 'w_k':7},
        {'t_n':100, 't_ci':1,  't_co':32,  't_h':28,  't_w':28,  'w_k':1},
        {'t_n':1,   't_ci':1,  't_co':2,   't_h':4,   't_w':4,   'w_k':1}
    ]

    for i in range(iterations):
        pbar.update(i)
        tf.reset_default_graph()

        # NCHW
        p = kernel_params[i % len(kernel_params)]
        t_n = p['t_n']
        t_ci = p['t_ci']
        t_co = p['t_co']
        t_h = p['t_h']
        t_w = p['t_w']
        w_k = p['w_k']

        # N H W C
        activations = tf.get_variable("a", [t_n, t_h, t_w, t_ci], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        # K K Ci Co
        c2d_wts =     tf.get_variable("b", [w_k, w_k, t_ci, t_co], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))

        t_init = tf.global_variables_initializer()


        # SAME variant, stride = 1
        conv2d_test(config, t_init, i, p, activations, c2d_wts, stride=1, padding='SAME')

        # Valid variant, stride = 1
        conv2d_test(config, t_init, i, p, activations, c2d_wts, stride=1, padding='VALID')

        # SAME variant, stride = 2
        # conv2d_test(config, t_init, i, p, activations, c2d_wts, stride=2, padding='SAME')

        # Valid variant, stride = 2
        conv2d_test(config, t_init, i, p, activations, c2d_wts, stride=2, padding='VALID')

    pbar.finish()
    return True


if __name__ == "__main__":
    test_conv2d()

