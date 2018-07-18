#!/usr/bin/env python3

# test_mul_int.py
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
#      Test integer kernel of multiply operation.
#
# Author          : Djordje Kovacevic
# Created On      : 05/31/2018
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


import sys
import os

import pytest

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import waveflow


def compare_tensor(z1, z2, msg):
    ''' Run a compare on 2 tensors for equality. Report failure details.
    '''
    # assert z1.shape == z2.shape, msg
    if z1.shape != z2.shape:
        print(msg)
        print("z1 shape: %s, z2 shape: %s" % (str(z1.shape), str(z2.shape)))
        return False

    rtol = 1e-2
    if not np.allclose(z1, z2, atol=rtol):
        print("\n\n")
        d = ~np.isclose(z1, z2, atol=rtol)
        print("z1 mismatch: %s" % (z1[d]))
        print("z2 mismatch: %s" % (z2[d]))
        print("at: %s" % (str(np.where(d))))
        print("Failure: %s" % (msg))
        return False

    return True

def test_multiply_int():

    tf.reset_default_graph()
    debug = False

    shapes = [
        ([2, 5], [1]),
        ([1], [3, 6]),
        ([5, 3], [5, 1]),
        ([1, 5], [5, 1]),
        ([7, 1], [1, 7]),
        ([2, 8], [1, 8]),
        ([3, 1], [3, 9]),
        ([1, 4], [6, 4]),
        ([4, 6], [4, 6]),
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
        ([4, 3, 1, 4], [4, 3, 4, 1]),
        ([4, 3, 1, 4], [1, 3, 4, 4]),
        ([2, 5, 3, 6], [2, 5, 3, 6]),
        ([1, 1, 3, 6], [2, 5, 3, 6]),
        ([2, 0, 5, 2], [2, 0, 1, 2]),
        ([5, 2, 3, 0], [5, 2, 3, 1]),
        ([20, 10, 1, 1], [20, 10, 30, 30]),
        ([5, 2, 3, 1, 4], [1, 2, 3, 4, 1])
    ]

    t_init = tf.global_variables_initializer()

    for (xs, ys) in shapes:
        #x = np.random.randint(-10, 10, np.prod(xs)).astype(np.float32).reshape(xs)
        #y = np.random.randint(-10, 10, np.prod(ys)).astype(np.float32).reshape(ys)
        x = np.random.random(xs).astype(np.float32)
        y = np.random.random(ys).astype(np.float32)
        inx = ops.convert_to_tensor(x)
        iny = ops.convert_to_tensor(y)

        for i in range(10):

            with tf.Session('') as sess:

                t_init.run()

                z1_op = tf.multiply(inx, iny)
                z2_op = waveflow.wavecomp_ops_module.wave_mul_int(inx, iny)
                z1, z2 = sess.run([z1_op, z2_op])
                if not compare_tensor(z1, z2, "Failure "):
                    print("a: %s" % (inx.eval()))
                    print("b: %s" % (iny.eval()))
                    print("z1: %s" % (z1))
                    print("z2: %s" % (z2))
                    print("\n\n")
                    assert False
    return True



if __name__ == "__main__":
    test_multiply_int()

