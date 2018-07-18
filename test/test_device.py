#!/usr/bin/env python3

# test_device.py 
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
# Author          : Ken Shiring
# Created On      : 05/08/2018
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

import waveflow


# def my_custom_dev(n):
#     print("Got dev arg: %s\n%s" % (type(n), str(n)))
#     return "/device:DPU:0"


def test_device():
    ''' Run tests on the Wave custom vector add operator. 
    '''
    tf.reset_default_graph()

    v_a1 = tf.get_variable("a1", [2, 100], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    v_a2 = tf.get_variable("a2", [2, 10, 30, 100], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    v_b = tf.get_variable("b", [100], dtype=tf.float32, initializer=tf.glorot_normal_initializer())

    t_init = tf.global_variables_initializer()
    t_debug = False

    z_op = waveflow.wavecomp_ops_module.wave_vec_add(v_a1, v_b)

    for i in range(1):

        with tf.Session('') as sess:
            with sess.graph.device(waveflow.wf_custom_dev):
                t_init.run()
                if t_debug: print("dims: a: %s, b: %s" % (v_a1.shape, v_b.shape))

                if t_debug: print("Wave Kernel:\n-------------------------------------------------")
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

                assert np.allclose(z, z_np)


    return True


if __name__ == "__main__":
    test_device()

