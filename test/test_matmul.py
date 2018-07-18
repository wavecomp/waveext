#!/usr/bin/env python3

# test_matmul.py 
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
#      Basic matmul test.
# 
# Author          : Ken Shiring
# Created On      : 02/09/2018
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

import progressbar as pb


import numpy as np
import tensorflow as tf

import waveflow


def test_matmul():
    ''' Run tests on the Wave custom matmul operator. 
    '''
    tf.reset_default_graph()

    a = tf.get_variable("a", [2, 3], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
    b = tf.get_variable("b", [3, 4], dtype=tf.float32, initializer=tf.glorot_normal_initializer())

    t_init = tf.global_variables_initializer()
    debug = False

    iters = 100
    widgets = ["matmul test: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=iters)
    pbar.start()

    for i in range(100):
        pbar.update(i)

        # NN variant
        with tf.Session(''):
            t_init.run()
            if debug: print("Wave Kernel (NN):\n-------------------------------------------------")
            if debug: print("a: %s" % (a.eval()))
            if debug: print("b: %s" % (b.eval()))
            # (2, 3) * (3, 4) = (2, 4)
            z = waveflow.wavecomp_ops_module.wave_mat_mul(a, b).eval()
            if debug: print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(a.eval())
            b_np = np.array(b.eval())
            z2 = np.matmul(a_np, b_np)
            if debug: print("\nNumpy:\n-------------------------------------------------")
            if debug: print("a (np): %s" % (a_np))
            if debug: print("b (np): %s" % (b_np))
            if debug: print("z (np): %s" % (z2))
            if debug: print("\n\n")

            assert np.allclose(z, z2, atol=0.1)

        # TN variant
        with tf.Session(''):
            t_init.run()
            if debug: print("Wave Kernel (TN):\n-------------------------------------------------")
            a_t = tf.transpose(a)
            if debug: print("a: %s" % (a_t.eval()))
            if debug: print("b: %s" % (b.eval()))
            # (3, 2).T * (3, 4) = (2, 4)
            z = waveflow.wavecomp_ops_module.wave_mat_mul(a_t, b, transpose_a=True).eval()
            if debug: print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(a_t.eval())
            b_np = np.array(b.eval())
            assert np.allclose(a.eval(), a_np.T)

            z2 = np.matmul(a_np.T, b_np)
            if debug: print("\nNumpy:\n-------------------------------------------------")
            if debug: print("a (np): %s" % (a_np))
            if debug: print("b (np): %s" % (b_np))
            if debug: print("z (np): %s" % (z2))
            if debug: print("\n\n")

            assert np.allclose(z, z2, atol=0.1)


        # NT variant
        with tf.Session(''):
            t_init.run()
            if debug: print("Wave Kernel (NT):\n-------------------------------------------------")
            b_t = tf.transpose(b)
            if debug: print("a: %s" % (a.eval()))
            if debug: print("b: %s" % (b_t.eval()))
            z = waveflow.wavecomp_ops_module.wave_mat_mul(a, b_t, transpose_b=True).eval()
            if debug: print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(a.eval())
            b_np = np.array(b_t.eval())
            z2 = np.matmul(a_np, b_np.T)
            if debug: print("\nNumpy:\n-------------------------------------------------")
            if debug: print("a (np): %s" % (a_np))
            if debug: print("b (np): %s" % (b_np))
            if debug: print("z (np): %s" % (z2))
            if debug: print("\n\n")

            assert np.allclose(z, z2, atol=0.1)

        # TT variant
        with tf.Session(''):
            t_init.run()
            if debug: print("Wave Kernel (TT):\n-------------------------------------------------")
            a_t = tf.transpose(a)
            b_t = tf.transpose(b)
            if debug: print("a: %s" % (a_t.eval()))
            if debug: print("b: %s" % (b_t.eval()))
            # (3, 2).T * (4, 3).T = (2, 4)
            z = waveflow.wavecomp_ops_module.wave_mat_mul(a_t, b_t, transpose_a=True, transpose_b=True).eval()
            if debug: print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(a_t.eval())
            b_np = np.array(b_t.eval())
            z2 = np.matmul(a_np.T, b_np.T)
            if debug: print("\nNumpy:\n-------------------------------------------------")
            if debug: print("a (np): %s" % (a_np))
            if debug: print("b (np): %s" % (b_np))
            if debug: print("z (np): %s" % (z2))
            if debug: print("\n\n")

            assert np.allclose(z, z2, atol=0.1)

    pbar.finish()
    return True


if __name__ == "__main__":
    test_matmul()

