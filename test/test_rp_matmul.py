#!/usr/bin/env python3

# test_rp_matmul.py 
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
#      Top level run script for regression testing.
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


import tensorflow as tf

import waveflow


def test_rp_matmul():
    ''' Run tests on the Reduced Precision Matmul operator. 
    '''
    tf.reset_default_graph()

    with tf.Session(''):
        # a = tf.Variable(tf.zeros([32], dtype=tf.int16))
        # b = tf.Variable(tf.zeros([32], dtype=tf.int16))

        # z = wavecomp_ops_module.wc_reducedp_mat_mul(a, b, 8, 8, 8).eval()

        z = waveflow.wavecomp_ops_module.wc_reducedp_mat_mul([[1, 2, 3]], [[1], [2], [3]], 8, 8, 8).eval()
    return True


if __name__ == "__main__":
    test_rp_matmul()

