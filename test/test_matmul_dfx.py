#!/usr/bin/env python3

# test_matmul_dfx.py 
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
#      Basic matmul DFX test.
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
import pytest

import waveflow


def test_matmul_dfx():
    ''' Run tests on the Wave custom matmul operator. 
    '''
    tf.reset_default_graph()

    if True:
        a_v = tf.get_variable("a", [2, 3], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        b_v = tf.get_variable("b", [3, 4], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        iters = 100
    else:
        a_v = tf.get_variable("a", dtype=tf.float32, initializer=[[-0.04236992, -0.2878763,   0.1678541 ],
            [-0.03303943,  0.44365183,  0.06329777]])
        b_v = tf.get_variable("b", dtype=tf.float32, initializer=[[ 0.5406903,   0.1869686,  -0.13071704, -0.1649987 ],
             [-0.4734501,  -0.05453536,  0.38498643, -0.11109588],
             [-0.31146765, -0.33795518,  0.3565544,  -0.38937578]])
        iters = 1

    t_init = tf.global_variables_initializer()
    debug = False

    widgets = ["matmul BP test: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=iters)
    pbar.start()

    for i in range(iters):
        pbar.update(i)

        # NN variant
        with tf.Session('') as sess:
            t_init.run()
            # (2, 3) * (3, 4) = (2, 4)
            z_op = waveflow.wavecomp_ops_module.wave_mat_mul_dfx(a_v, b_v)
            z, a, b = sess.run([z_op, a_v, b_v])
            if debug: print("Wave Kernel (NN):\n-------------------------------------------------")
            if debug: print("a: %s" % (a))
            if debug: print("b: %s" % (b))
            if debug: print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(a)
            b_np = np.array(b)
            z2 = np.matmul(a_np, b_np)
            if debug: print("\nNumpy:\n-------------------------------------------------")
            if debug: print("a (np): %s" % (a_np))
            if debug: print("b (np): %s" % (b_np))
            if debug: print("z (np): %s" % (z2))
            if debug: print("\n\n")

            assert np.allclose(z, z2, atol=0.1)

        # TN variant
        with tf.Session('') as sess:
            t_init.run()
            a_t = tf.transpose(a)
            # (3, 2).T * (3, 4) = (2, 4)
            z_op = waveflow.wavecomp_ops_module.wave_mat_mul(a_t, b_v, transpose_a=True)
            z, at, b = sess.run([z_op, a_t, b_v])
            if debug: print("Wave Kernel (TN):\n-------------------------------------------------")
            if debug: print("a: %s" % (at))
            if debug: print("b: %s" % (b))
            if debug: print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(at)
            b_np = np.array(b)
            assert np.allclose(a, a_np.T)

            z2 = np.matmul(a_np.T, b_np)
            if debug: print("\nNumpy:\n-------------------------------------------------")
            if debug: print("a (np): %s" % (a_np))
            if debug: print("b (np): %s" % (b_np))
            if debug: print("z (np): %s" % (z2))
            if debug: print("\n\n")

            assert np.allclose(z, z2, atol=0.1)


        # NT variant
        with tf.Session('') as sess:
            t_init.run()
            b_t = tf.transpose(b)
            z_op = waveflow.wavecomp_ops_module.wave_mat_mul(a_v, b_t, transpose_b=True)
            z, a, bt = sess.run([z_op, a_v, b_t])
            if debug: print("Wave Kernel (NT):\n-------------------------------------------------")
            if debug: print("a: %s" % (a))
            if debug: print("b: %s" % (bt))
            if debug: print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(a)
            b_np = np.array(bt)
            z2 = np.matmul(a_np, b_np.T)
            if debug: print("\nNumpy:\n-------------------------------------------------")
            if debug: print("a (np): %s" % (a_np))
            if debug: print("b (np): %s" % (b_np))
            if debug: print("z (np): %s" % (z2))
            if debug: print("\n\n")

            assert np.allclose(z, z2, atol=0.1)

        # TT variant
        with tf.Session('') as sess:
            t_init.run()
            a_t = tf.transpose(a)
            b_t = tf.transpose(b)
            # (3, 2).T * (4, 3).T = (2, 4)
            z_op = waveflow.wavecomp_ops_module.wave_mat_mul(a_t, b_t, transpose_a=True, transpose_b=True)
            z, at, bt = sess.run([z_op, a_t, b_t])
            if debug: print("Wave Kernel (TT):\n-------------------------------------------------")
            if debug: print("a: %s" % (at))
            if debug: print("b: %s" % (bt))
            if debug: print("z: %s" % (z))

            # Convert to numpy
            a_np = np.array(at)
            b_np = np.array(bt)
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
    test_matmul_dfx()

