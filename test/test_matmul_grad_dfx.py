#!/usr/bin/env python3

# test_matmul_grad_dfx.py 
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
#      Test matmul gradients in Dynamic Fixed Point.
# 
# Author          : Ken Shiring
# Created On      : 03/06/2018
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


def _wave_mat_mul_dfx_grad_cc(op_input, op_weights, grad):
    # KS: needs Dynamic Fixed Point specialization.
    # return _wave_mat_mul_grad_cc(op, grad)

    # grad_inputs = waveflow.wavecomp_ops_module.wave_mat_mul_dfx(grad, op_weights, transpose_b=True)
    # grad_weights = tf.transpose(waveflow.wavecomp_ops_module.wave_mat_mul_dfx(grad, op_input, transpose_a=True))
    # return grad_inputs, grad_weights
    tuple_op = namedtuple('dummy_op', ['inputs'])
    d = tuple_op(inputs=[op_input, op_weights])
    return waveflow._wave_mat_mul_dfx_grad_cc(d, grad)


def test_matmul_grad_dfx():
    ''' Run tests on the Wave custom matmul gradient operator. 
    '''
    tf.reset_default_graph()

    #   .Input("grad: float32")
    #   .Input("input: float32")
    #   .Input("weights: float32")
    #   .Output("grad_input: float32")
    #   .Output("grad_weights: float32");
    debug = True

    t_grad = tf.get_variable("grad", [3, 2], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    t_input = tf.get_variable("input", [3, 4], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    t_weights = tf.get_variable("weights", [4, 2], dtype=tf.float32, initializer=tf.glorot_normal_initializer())

    t_init = tf.global_variables_initializer()

    for i in range(100):
        with tf.Session('') as sess:
            t_init.run()
            # print("op: %s" % (str(wavecomp_ops_module.wave_mat_mul_grad)))
            # t_z_in, t_z_wt = _wave_mat_mul_dfx_grad_cc(t_input, t_weights, t_grad)
            # z_in = t_z_in.eval()
            # # (4, 2)
            # z_wt = t_z_wt.eval()
            z_op = _wave_mat_mul_dfx_grad_cc(t_input, t_weights, t_grad)
            v_input, v_weights, v_grad, t_z = sess.run([t_input, t_weights, t_grad, z_op])
            v_z_in, v_z_wt = t_z
            if debug:
                print("Wave Kernel:\n-------------------------------------------------")
                print("grad: %s" % (t_grad.eval()))
                print("input: %s" % (t_input.eval()))
                print("weights: %s" % (t_weights.eval()))
                print("z_in: %s" % (v_z_in))
                print("z_wt: %s" % (v_z_wt))

            
            # Convert to numpy
            # grad_np = np.array(t_grad.eval())
            # input_np = np.array(t_input.eval())
            # weights_np = np.array(t_weights.eval())
            grad_np = np.array(v_grad)
            input_np = np.array(v_input)
            weights_np = np.array(v_weights)
            z_np_in = np.matmul(grad_np, weights_np.T)
            # (2, 3) x (3, 4) = (2, 4)
            z_np_wt = np.matmul(grad_np.T, input_np).T
            if debug:
                print("\nNumpy:\n-------------------------------------------------")
                print("grad (np): %s" % (grad_np))
                print("input (np): %s" % (input_np))
                print("weights (np): %s" % (weights_np))
                print("z_in (np): %s" % (z_np_in))
                print("z_wt (np): %s" % (z_np_wt))

            print("Iter: %d" % (i))
            assert np.allclose(v_z_in, z_np_in, atol=1e-3)
            assert np.allclose(v_z_wt, z_np_wt, atol=1e-3)
        

    return True


if __name__ == "__main__":
    test_matmul_grad_dfx()

