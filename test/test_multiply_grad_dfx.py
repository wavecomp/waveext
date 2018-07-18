#!/usr/bin/env python3

# test_multiply_grad_dfx.py 
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
#      Test multiply gradients in Dynamic Fixed Point.
#
# Author          : Djordje Kovacevic
# Created On      : 05/23/2018
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

import numpy as np
import tensorflow as tf

import waveflow


def _wave_mul_grad_cc(op_x, op_y, grad):
    tuple_op = namedtuple('dummy_op', ['inputs'])
    d = tuple_op(inputs=[op_x, op_y])
    return waveflow._wave_mul_grad_dfx_cc(d, grad)


def test_mul_grad():
    ''' Run tests on the Wave custom matmul gradient operator.
    '''
    tf.reset_default_graph()

    #   .Input("grad: float32")
    #   .Input("x: float32")
    #   .Input("y: float32")
    #   .Output("grad_x: float32")
    #   .Output("grad_y: float32");
    debug = False

    t_grad = tf.get_variable("grad", [3, 2], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    t_x = tf.get_variable("x", [3, 2], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    t_y = tf.get_variable("y", [3, 2], dtype=tf.float32, initializer=tf.glorot_normal_initializer())


    t_init = tf.global_variables_initializer()

    for i in range(100):
        with tf.Session('') as sess:
            t_init.run()

            z_op = _wave_mul_grad_cc(t_x, t_y, t_grad)
            v_x, v_y, v_grad, t_z = sess.run([t_x, t_y, t_grad, z_op])
            v_z_x, v_z_y = t_z
            if debug:
                print("Iter: %d" % (i))
                print("Wave Kernel:\n-------------------------------------------------")
                print("grad: %s" % (t_grad.eval()))
                print("x: %s" % (t_x.eval()))
                print("y: %s" % (t_y.eval()))
                print("z_x: %s" % (v_z_x))
                print("z_y: %s" % (v_z_y))

            # Convert to numpy
            grad_np = np.array(v_grad)
            x_np = np.array(v_x)
            y_np = np.array(v_y)
            z_np_x = np.multiply(grad_np, y_np)
            z_np_y = np.multiply(grad_np, x_np)
            if debug:
                print("\nNumpy:\n-------------------------------------------------")
                print("grad (np): %s" % (grad_np))
                print("x (np): %s" % (x_np))
                print("y (np): %s" % (y_np))
                print("z_x (np): %s" % (z_np_x))
                print("z_y (np): %s" % (z_np_y))

            assert np.allclose(v_z_x, z_np_x, atol=1e-3)
            assert np.allclose(v_z_y, z_np_y, atol=1e-3)

    return True


if __name__ == "__main__":
    test_mul_grad()

