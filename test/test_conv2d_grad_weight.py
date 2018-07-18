#!/usr/bin/env python3

# test_conv2d_grad_weight.py
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
# Author          : Djordje Kovacevic
# Created On      : 03/13/2018
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
import pytest

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

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


def test_conv2d_grad_weight():
    ''' Run tests on the Wave custom conv2d operator. 
    '''
    tf.reset_default_graph()

    # Turn off graph-rewriting optimizations
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

    # NCHW
    t_n = 1
    t_ci = 2
    t_co = 4
    t_h = 4
    t_w = t_h

    w_k = 3

    t_ho = t_h - (w_k - 1)
    t_wo = t_w - (w_k - 1)

    if True:
        # N H W C
        gradient_same  = tf.get_variable("gs", [t_n, t_h, t_w, t_co], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        gradient_valid = tf.get_variable("gv", [t_n, t_ho, t_wo, t_co], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        # K K Ci Co
        c2d_in =  tf.get_variable("f", [t_n, t_h, t_w, t_ci], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        total_iters = 100

    else:
        total_iters = 100
        '''
        # N H W C
        gradient_same  = tf.get_variable("gs", dtype=tf.float32, 
            initializer=[[[[ 0.20130304,  0.27111197, -0.23086302, -0.23035139],
                           [-0.09796654, -0.25246394, -0.37301826, -0.04510078],
                           [ 0.23488244,  0.11107023, -0.06509414, -0.10155579],
                           [-0.02653619,  0.25628388,  0.14519225, -0.03400368]],
                          [[-0.2737653,  -0.09805538,  0.2395286,  -0.05114909],
                           [ 0.25615498, -0.00413944, -0.11665037,  0.41589662],
                           [-0.2739116,  -0.25920913, -0.09962016,  0.09206288],
                           [ 0.20151283,  0.17594334, -0.07801611,  0.18326335]],
                          [[ 0.41815385, -0.2302951,   0.04820603,  0.28555122],
                           [-0.07512673,  0.03257893,  0.26206508, -0.23212446],
                           [ 0.11086888, -0.14072362,  0.0384812,  -0.3119425 ],
                           [ 0.12364136,  0.20889276,  0.21803159,  0.23013732]],
                          [[ 0.14886299,  0.2521119,   0.09665762, -0.0950445 ],
                           [-0.02764839, -0.22908632, -0.21170329, -0.01173863],
                           [-0.01734301, -0.09034941, -0.36093128, -0.10678421],
                           [ 0.22199751, -0.07065484, -0.20354767,  0.35294214]]]])

        # K K Ci Co
        c2d_in =  tf.get_variable("f", dtype=tf.float32, 
            initializer=[[[[ 0.05964299, -0.05988665],
                           [ 0.1176498,  -0.03819368],
                           [ 0.5281046,  -0.23744191],
                           [ 0.11645658, -0.19929294]],
                          [[-0.23003536,  0.00503353],
                           [-0.13039683,  0.07515113],
                           [ 0.27454942,  0.07771762],
                           [-0.30660862,  0.0843183 ]],
                          [[ 0.45593834, -0.10620885],
                           [-0.2692576,  -0.21969746],
                           [-0.0423032,  -0.27153978],
                           [-0.12364867,  0.4992814 ]],
                          [[ 0.199746,   -0.36852396],
                           [ 0.06857206, -0.5132366 ],
                           [-0.3468758,  -0.11179192],
                           [-0.00362369,  0.24938436]]]])
        '''
        # N H W C
        gradient_valid  = tf.get_variable("gv", dtype=tf.float32, 
            initializer=[[[[-0.46859148, -0.47966853,  0.20002109, -0.56815886],
                           [ 0.28490868, -0.34714422, -0.02198983, -0.16817461]],
                          [[ 0.55926424,  0.43453974, -0.14650254, -0.35478827],
                           [ 0.33911255, -0.13839267,  0.50347936, -0.24987353]]]])

        # K K Ci Co
        c2d_in =  tf.get_variable("f", dtype=tf.float32, 
            initializer=[[[[-0.02360273,  0.11748651],
                           [-0.06789866, -0.09688393],
                           [ 0.19889985, -0.06614035],
                           [ 0.1964069,  -0.49319836]],
                          [[-0.11062168,  0.10862893],
                           [-0.11725109, -0.25628847],
                           [-0.03851736, -0.04931202],
                           [ 0.21630755, -0.22763665]],
                          [[ 0.22559226,  0.38029733],
                           [ 0.10525792, -0.36071476],
                           [ 0.05696308, -0.12789722],
                           [ 0.07823437,  0.0835037 ]],
                          [[ 0.2075723,   0.14272317],
                           [ 0.50318813,  0.3311627 ],
                           [-0.0821431,   0.15861501],
                           [-0.0524208,  -0.01618423]]]])

    wgt_shape = tf.get_variable("s", dtype=tf.int32, initializer=[w_k, w_k, t_ci, t_co])

    t_init = tf.global_variables_initializer()

    for i in range(total_iters):
        
        # SAME variant
        with tf.Session('', config=config) as sess:
            t_init.run()
            
            # print("Wave Kernel:\n-------------------------------------------------")
            
            #z_op = nn_ops.conv2d_backprop_filter(
            z_op = waveflow.wavecomp_ops_module.wave_conv2d_gradient_weight(
                c2d_in,
                wgt_shape,
                gradient_same,
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_cudnn_on_gpu=False,
                data_format='NHWC')

            # Base tensorflow. Only supports NHWC.
            z2_op = nn_ops.conv2d_backprop_filter(
            #z2_op = wavecomp_ops_module.wave_conv2d_gradient_weight(
                c2d_in,
                wgt_shape,
                gradient_same,
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_cudnn_on_gpu=False,
                data_format='NHWC')

            # z = z_op.eval()
            # z2 = z2_op.eval()
            z, z2, grad_val, in_val = sess.run([z_op, z2_op, gradient_same, c2d_in])

            # print("\nTF:\n-------------------------------------------------")

            assert_str = "Failure on i: %d, mode: SAME" % (i)
            if not compare_tensor(z, z2, assert_str):
                print("gradients: %s" % (grad_val))
                print("c2d_in: %s" % (in_val))
                print("z: shape: %s, %s" % (z.shape, z))
                print("z (np): shape: %s, %s" % (z2.shape, z2))
                print("\n\n")
                assert False
        
        # Valid variant
        with tf.Session('', config=config) as sess:
            t_init.run()
            
            # print("Wave Kernel:\n-------------------------------------------------")
            
            #z_op = nn_ops.conv2d_backprop_filter(
            z_op = waveflow.wavecomp_ops_module.wave_conv2d_gradient_weight(
                c2d_in,
                wgt_shape,
                gradient_valid,
                strides=[1, 1, 1, 1],
                padding='VALID',
                use_cudnn_on_gpu=False,
                data_format='NHWC')


            # Base tensorflow. Only supports NHWC.
            z2_op = nn_ops.conv2d_backprop_filter(
            #z2_op = wavecomp_ops_module.wave_conv2d_gradient_weight(
                c2d_in,
                wgt_shape,
                gradient_valid,
                strides=[1, 1, 1, 1],
                padding='VALID',
                use_cudnn_on_gpu=False,
                data_format='NHWC')


            z, z2, grad_val, in_val = sess.run([z_op, z2_op, gradient_valid, c2d_in])
            # print("\nTF:\n-------------------------------------------------")

            assert_str = "Failure on i: %d, mode: VALID" % (i)
            if not compare_tensor(z, z2, assert_str):
                print("gradients: %s" % (grad_val))
                print("c2d_in: %s" % (in_val))
                print("z: shape: %s, %s" % (z.shape, z))
                print("z (np): shape: %s, %s" % (z2.shape, z2))
                print("\n\n")
                assert False

    return True


if __name__ == "__main__":
    test_conv2d_grad_weight()

