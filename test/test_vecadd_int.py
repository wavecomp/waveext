#!/usr/bin/env python3

# test_vecadd_int.py
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
#      Test vector add
#
# Author          : Djordje Kovacevic
# Created On      : 05/28/2018
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

from tensorflow.python.ops import gen_nn_ops

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



def test_vecadd_int():
    ''' Run tests on the Wave custom vector add operator.
    '''
    tf.reset_default_graph()

    v_a1 = tf.get_variable("a1", [30, 100], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    v_a2 = tf.get_variable("a2", [2, 10, 30, 100], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    v_b = tf.get_variable("b", [100], dtype=tf.float32, initializer=tf.glorot_normal_initializer())

    t_init = tf.global_variables_initializer()
    t_debug = False

    for i in range(100):

        with tf.Session('') as sess:
            t_init.run()
            if t_debug: print("dims: a: %s, b: %s" % (v_a1.shape, v_b.shape))
            if t_debug: print("Wave Kernel:\n-------------------------------------------------")
            z_op = waveflow.wavecomp_ops_module.wave_vec_add_int(v_a1, v_b)
            z, a, b = sess.run([z_op, v_a1, v_b])
            if t_debug:
                print("a: %s" % (a))
                print("b: %s" % (b))
                print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(a)
            b_np = np.array(b)
            # Numpy's intrinsic broadcast rules should do the right thing here.
            z_np = np.add(a_np, b_np)
            if t_debug:
                print("\nNumpy:\n-------------------------------------------------")
                print("a (np): %s" % (a_np))
                print("b (np): %s" % (b_np))
                print("z (np): %s" % (z_np))

            assert_str = "Failure on i: %d, params: %s" % (i, str(v_a1.shape))
            if not compare_tensor(z, z_np, assert_str):
                print("a: %s" % (a))
                print("b: %s" % (b))
                print("z: %s" % (z))
                print("z (np): %s" % (z_np))

                print("\n\n")
                assert False


        with tf.Session('') as sess:
            t_init.run()
            if t_debug: 
                print("dims: a: %s, b: %s" % (v_a2.shape, v_b.shape))
                print("Wave Kernel:\n-------------------------------------------------")
            z2_op = waveflow.wavecomp_ops_module.wave_vec_add_int(v_a2, v_b)
            z2, a, b = sess.run([z2_op, v_a2, v_b])
            if t_debug:
                print("a: %s" % (a))
                print("b: %s" % (b))
                print("z: %s" % (z2))

            # Convert to numpy
            a_np = np.array(a)
            b_np = np.array(b)
            # Numpy's intrinsic broadcast rules should do the right thing here.
            z2_np = np.add(a_np, b_np)
            if t_debug:
                print("\nNumpy:\n-------------------------------------------------")
                print("a (np): %s" % (a_np))
                print("b (np): %s" % (b_np))
                print("z (np): %s" % (z2_np))

            assert_str = "Failure on i: %d, params: %s" % (i, str(v_a2.shape))
            if not compare_tensor(z2, z2_np, assert_str):
                print("a: %s" % (a))
                print("b: %s" % (b))
                print("z: %s" % (z2))
                print("z (np): %s" % (z2_np))

                print("\n\n")
                assert False

    return True

import time
if __name__ == "__main__":
    test_vecadd_int()
