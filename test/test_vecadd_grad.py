#!/usr/bin/env python3

# test_vecadd_grad.py 
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
#      Test vector add gradeints
# 
# Author          : Ken Shiring
# Created On      : 02/16/2018
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


def test_vecadd_grad():
    ''' Run tests on the Wave custom vector add operator. 
    '''
    tf.reset_default_graph()

    v_a1 = tf.get_variable("a1", [10, 5], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    v_a2 = tf.get_variable("a2", [3, 2, 10, 6], dtype=tf.float32, initializer=tf.glorot_normal_initializer())

    t_init = tf.global_variables_initializer()
    t_debug = False

    for i in range(100):
        with tf.Session('') as sess:
            t_init.run()

            if t_debug:
                print("Wave Kernel:\n-------------------------------------------------")
                print("dims: a: %s" % (v_a1.shape))
            z_op = waveflow.wavecomp_ops_module.wave_vec_add_grad(v_a1)
            z_tf_op = gen_nn_ops.bias_add_grad(v_a1)
            z, z_tf, a1 = sess.run([z_op, z_tf_op, v_a1])
            if t_debug: 
                print("a: %s" % (a1))
                print("z: %s" % (z))
                print("z (tf): %s" % (z_tf))

            assert np.allclose(z, z_tf)


        with tf.Session('') as sess:
            t_init.run()

            if t_debug:
                print("Wave Kernel:\n-------------------------------------------------")
                print("dims: a: %s" % (v_a2.shape))
            z2_op = waveflow.wavecomp_ops_module.wave_vec_add_grad(v_a2)
            z2_tf_op = gen_nn_ops.bias_add_grad(v_a2)
            z2, z2_tf, a2 = sess.run([z2_op, z2_tf_op, v_a2])
            if t_debug: 
                print("a: %s" % (a2))
                print("z: %s" % (z2))
                print("z (tf): %s" % (z2_tf))

            assert np.allclose(z2, z2_tf)

    return True


if __name__ == "__main__":
    test_vecadd_grad()

