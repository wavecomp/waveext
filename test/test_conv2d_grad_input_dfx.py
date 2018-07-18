#!/usr/bin/env python3

# test_conv2d_grad_input_dfx.py 
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
#      Unit test for BP Conv2D
# 
# Author          : Ken Shiring
# Created On      : 03/07/2018
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
    rtol = 1e-3
    if not np.allclose(z1, z2, atol=rtol):
        print("\n\n")
        d = ~np.isclose(z1, z2, atol=rtol)
        print("z1 mismatch: %s" % (z1[d]))
        print("z2 mismatch: %s" % (z2[d]))
        print("at: %s" % (str(np.where(d))))
        print("Failure: %s" % (msg))
        return False

    return True


def test_conv2d_grad_input_dfx():
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
        c2d_wts =  tf.get_variable("f", [w_k, w_k, t_ci, t_co], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        total_iters = 100
    else:
        total_iters = 100
        '''
        # N H W C
        gradient_same  = tf.get_variable("gs", dtype=tf.float32, 
            initializer=[[[[ 0.08031779, -0.25303698,  0.03623476, -0.18058969],
                           [-0.13975245, -0.1637536,   0.16259798, -0.02949391],
                           [ 0.09451582,  0.22659606,  0.05693823,  0.4917419 ],
                           [-0.30809653,  0.07311761,  0.3780873,  -0.08786422]],
                          [[ 0.2738534,   0.04332319,  0.17183594,  0.05189532],
                           [ 0.12897372, -0.04585831, -0.22999597, -0.34978592],
                           [-0.30032733, -0.24595816,  0.1805963,  -0.15278156],
                           [ 0.32070962, -0.08978839, -0.22666173,  0.2911243 ]],
                          [[-0.08074081, -0.39459208,  0.21127546, -0.4269226 ],
                           [ 0.08455674, -0.0359996,   0.2437691,   0.32546043],
                           [-0.00876724, -0.3661593,   0.11359364, -0.02061039],
                           [-0.08034749,  0.10190688, -0.02636097,  0.04948684]],
                          [[ 0.17580973,  0.01204642, -0.19307376,  0.02388505],
                           [ 0.09260377,  0.07462457,  0.37491015,  0.16807063],
                           [-0.26408812,  0.06460509,  0.01770028,  0.4702053 ],
                           [-0.07011604,  0.00892794, -0.04604778,  0.2620218 ]]]])

        # K K Ci Co
        c2d_wts =  tf.get_variable("f", dtype=tf.float32, 
            initializer=[[[[ 0.06656972, -0.10714861, -0.38451153,  0.23074387],
                           [-0.01944306, -0.30447152,  0.04994076, -0.09367681]],
                          [[ 0.05957646,  0.19590905,  0.11895119,  0.00895185],
                           [ 0.2502987,   0.20522587,  0.12625359,  0.01663731]],
                          [[-0.11965699, -0.0604148,   0.11648465,  0.36048207],
                           [-0.30445528,  0.09261203, -0.0333834,   0.24425095]]],
                         [[[ 0.16690508, -0.30831957, -0.07823245, -0.06891213],
                           [ 0.04043804,  0.12982817, -0.08034446,  0.12222644]],
                          [[-0.18581082, -0.12341253, -0.01331879,  0.05503126],
                           [-0.27605322,  0.13493368, -0.13817346, -0.05097049]],
                          [[-0.09993836,  0.20109807, -0.13612682, -0.20353511],
                           [-0.04943075,  0.07663928,  0.35151458,  0.06485093]]],
                         [[[-0.20775227, -0.16490227, -0.2690038,  -0.02779585],
                           [-0.10147694,  0.2807299,  -0.01072734, -0.18013969]],
                          [[ 0.08905439, -0.16782533,  0.06677146,  0.00378867],
                           [ 0.0749261,  -0.05467231,  0.05659405,  0.15372248]],
                          [[-0.05901923, -0.15167546, -0.06820064,  0.2985376 ],
                           [-0.08696859, -0.15774204,  0.27479723,  0.07990441]]]])
        '''
        # N H W C
        gradient_valid  = tf.get_variable("gs", dtype=tf.float32, 
            initializer=[[[[ 0.03840168,  0.10001929, -0.73023874, -0.64396137],
                         [ 0.06620534,  0.08522084,  0.2871255,  -0.2253074 ]],

                        [[ 0.36911452, -0.38547048,  0.03067151, -0.07689341],
                         [-0.00224663,  0.3763262,   0.18161671,  0.239477  ]]]])

        # K K Ci Co
        c2d_wts =  tf.get_variable("f", dtype=tf.float32, 
            initializer=[[[[ 0.05997244, -0.19243374,  0.2180913,   0.10070436],
                         [ 0.13730897,  0.03690678,  0.18295528, -0.17694373]],

                        [[-0.00731155,  0.20696399, -0.1208116,  -0.09631037],
                         [-0.05803618, -0.16626428, -0.06250718, -0.08830667]],

                        [[-0.01097434,  0.16296767, -0.31571156,  0.26823783],
                         [ 0.22869335,  0.11004149,  0.11700328, -0.20618626]]],


                       [[[-0.2435487,   0.11153392, -0.07607887,  0.09280396],
                         [ 0.03443901,  0.08875315,  0.31825522, -0.05234769]],

                        [[-0.03120394,  0.06814163,  0.1016055,  -0.02458281],
                         [ 0.26557085,  0.0025273,   0.09956491,  0.26658794]],

                        [[-0.11202502, -0.1806068,   0.08336216,  0.01869717],
                         [ 0.09206906, -0.01017215, -0.15024644, -0.25189745]]],


                       [[[-0.11102232,  0.12463175,  0.131392,    0.03081301],
                         [ 0.07039325, -0.20609103, -0.13402021,  0.2547113 ]],

                        [[ 0.14444682, -0.35085383,  0.271559,   -0.15192555],
                         [ 0.08196701, -0.16252175,  0.09922236, -0.04675364]],

                        [[ 0.27735794, -0.07617409,  0.08599395,  0.09511787],
                         [-0.17027737, -0.09131872,  0.24946164, -0.05494743]]]])


    in_shape = tf.get_variable("s", dtype=tf.int32, initializer=[t_n, t_h, t_w, t_ci])

    t_init = tf.global_variables_initializer()

    for i in range(total_iters):


        # SAME variant
        with tf.Session('', config=config) as sess:
            t_init.run()
            
            # print("Wave Kernel:\n-------------------------------------------------")
            
            # z_op = wavecomp_ops_module.wave_conv2d_gradient_input(
            # z_op = nn_ops.conv2d_backprop_input(
            z_op = waveflow.wavecomp_ops_module.wave_conv2d_dfx_gradient_input(
                in_shape,
                c2d_wts,
                gradient_same,
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_cudnn_on_gpu=False,
                data_format='NHWC')

            # Base tensorflow. Only supports NHWC.
            z2_op = nn_ops.conv2d_backprop_input(
                in_shape,
                c2d_wts,
                gradient_same,
                strides=[1, 1, 1, 1],
                padding='SAME',
                use_cudnn_on_gpu=False,
                data_format='NHWC')

            # z = z_op.eval()
            # z2 = z2_op.eval()
            z, z2, grad_val, wts_val = sess.run([z_op, z2_op, gradient_same, c2d_wts])
            # print("z: shape: %s, %s" % (z.shape, z))

            # print("\nTF:\n-------------------------------------------------")

            assert_str = "Failure on i: %d, mode: SAME" % (i)
            if not compare_tensor(z, z2, assert_str):
                print("gradients: %s" % (grad_val))
                print("c2d_wts: %s" % (wts_val))
                print("z: shape: %s, %s" % (z.shape, z))
                print("z (np): shape: %s, %s" % (z2.shape, z2))
                print("\n\n")
                assert False



        # Valid variant
        with tf.Session('', config=config) as sess:
            t_init.run()
            
            # print("Wave Kernel:\n-------------------------------------------------")
            
            # z_op = nn_ops.conv2d_backprop_input(
            z_op = waveflow.wavecomp_ops_module.wave_conv2d_dfx_gradient_input(
                in_shape,
                c2d_wts,
                gradient_valid,
                strides=[1, 1, 1, 1],
                padding='VALID',
                use_cudnn_on_gpu=False,
                data_format='NHWC')


            # Base tensorflow. Only supports NHWC.
            z2_op = nn_ops.conv2d_backprop_input(
                in_shape,
                c2d_wts,
                gradient_valid,
                strides=[1, 1, 1, 1],
                padding='VALID',
                use_cudnn_on_gpu=False,
                data_format='NHWC')


            z, z2, grad_val, wts_val = sess.run([z_op, z2_op, gradient_valid, c2d_wts])
            # print("\nTF:\n-------------------------------------------------")

            assert_str = "Failure on i: %d, mode: VALID" % (i)
            if not compare_tensor(z, z2, assert_str):
                print("gradients: %s" % (grad_val))
                print("c2d_wts: %s" % (wts_val))
                print("z: shape: %s, %s" % (z.shape, z))
                print("z (np): shape: %s, %s" % (z2.shape, z2))
                print("\n\n")
                assert False



    return True


if __name__ == "__main__":
    test_conv2d_grad_input_dfx()

