#!/usr/bin/env python3

# test_quantize.py 
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
#      Unit test for tensor quantize.
# 
# Author          : Ken Shiring
# Created On      : 05/24/2018
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

import progressbar as pb

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




def test_quantize():
    ''' Run tests on the Wave custom conv2d operator. 
    '''
    tf.reset_default_graph()

    iterations = 1

    # widgets = ["BP tests: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    # pbar = pb.ProgressBar(widgets=widgets, maxval=iterations)
    # pbar.start()

    tensor_const1 = tf.get_variable("const_a", initializer=np.full(shape=(8, 4, 16), fill_value=1., dtype=np.float32), dtype=tf.float32)
    tensor_const2 = tf.get_variable("const_b", initializer=np.full(shape=(32, 2, 4), fill_value=0.99, dtype=np.float32), dtype=tf.float32)
    tensor_const3 = tf.get_variable("const_c", initializer=np.full(shape=(4, 2, 4), fill_value=0.49, dtype=np.float32), dtype=tf.float32)

    tensor_c1v    = tf.get_variable("const_c1v", initializer=np.full(shape=(16), fill_value=1., dtype=np.float32), dtype=tf.float32)
    tensor_c2v    = tf.get_variable("const_c2v", initializer=np.full(shape=(16), fill_value=.99, dtype=np.float32), dtype=tf.float32)

    tensor_var1 = tf.get_variable("var_a", [16, 4, 32], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    tensor_var2 = tf.get_variable("var_b", [8, 2, 4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    tensor_var3 = tf.get_variable("var_c", [32, 4, 8], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))

    t_init = tf.global_variables_initializer()

    debug = True

    # Static data tests. These only need to run once.
    with tf.Session('') as sess:
        t_init.run()
        
        # Quantization should prune integer bits if set too small. Ensure the same variable loses
        # accuracy if required to quantize to an insufficient BP.
        const1_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const1, bp_o0='(16,15)')
        const1_q = waveflow.wavecomp_ops_module.wave_quantize_dfx(tensor_const1, const1_bp)

        val1_const, val1_bp, val1_q = sess.run([tensor_const1, const1_bp, const1_q])

        if debug: print("bp1: %s" % (val1_bp))
        assert np.array_equal(val1_bp, [16, 15])
        assert np.less(val1_q, val1_const).all()

        const2_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const2, bp_o0='(16,14)')
        const2_q = waveflow.wavecomp_ops_module.wave_quantize_dfx(tensor_const2, const2_bp)

        val2_const, val2_bp, val2_q = sess.run([tensor_const2, const2_bp, const2_q])

        if debug: print("bp2: %s" % (val2_bp))
        assert np.array_equal(val2_bp, [16, 14])
        assert np.less(val2_q, val2_const).all()

        const3_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const3, bp_o0='(16,13)')
        const3_q = waveflow.wavecomp_ops_module.wave_quantize_dfx(tensor_const3, const3_bp)

        val3_const, val3_bp, val3_q = sess.run([tensor_const3, const3_bp, const3_q])

        if debug: print("bp3: %s" % (val3_bp))
        assert np.array_equal(val3_bp, [16, 13])
        assert np.less(val3_q, val3_const).all()

    '''
    # Dynamic data tests.
    for i in range(iterations):
        # pbar.update(i)
        # tf.reset_default_graph()

        with tf.Session('') as sess:
            with sess.graph.device(my_custom_dev):                
                t_init.run()                
    '''

    # pbar.finish()
    return True


if __name__ == "__main__":
    test_quantize()

