#!/usr/bin/env python3

# test_bp.py 
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
#      Unit test for FP Conv2D
# 
# Author          : Ken Shiring
# Created On      : 05/16/2018
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


# Used for debugging and prototyping.
def my_custom_dev(n):
    # This should include more complex decision logic (including specific
    # checks below). If a supported op is seen, the device type should be 'DPU'.
    # All else should fall back to CPU.
    # n.attr['custom'] = 'bar'
    debug = False
    if debug:
        print("Got dev arg: %s\n%s" % (type(n), str(n)))
        # print("Got opdef: %s" % (str(n.op_def)))
        print("Got nodedef: %s" % (str(n.node_def)))
        print("nodedef members: %s" % (dir(n.node_def)))
    cin = n.control_inputs
    if cin:
        c_op = cin[0]
        if debug: print("control: %s" % (c_op))
        # ndef = n.node_def
        # ndef.__setattr__('afoo', 'abar')

    # return "/device:DPU:0"
    return "/device:CPU:0"





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




def test_bp():
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

    debug = False

    # Static data tests. These only need to run once.
    with tf.Session('') as sess:
        with sess.graph.device(my_custom_dev):                
            t_init.run()
            
            # The op output is the binary point.
            const1_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const1)
            const2_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const2)
            const3_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const3)

            bp_val1, bp_val2, bp_val3 = sess.run([const1_bp, const2_bp, const3_bp])

            if debug: print("bp1: %s" % (bp_val1))
            assert np.array_equal(bp_val1, [16, 14])
            if debug: print("bp2: %s" % (bp_val2))
            assert np.array_equal(bp_val2, [16, 15])
            if debug: print("bp3: %s" % (bp_val3))
            assert np.array_equal(bp_val3, [16, 16])

            const1_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const1, bp_o0='(32,-1)')
            const2_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const2, bp_o0='(32,-1)')
            const3_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const3, bp_o0='(32,-1)')

            bp_val1, bp_val2, bp_val3 = sess.run([const1_bp, const2_bp, const3_bp])

            if debug: print("bp1: %s" % (bp_val1))
            assert np.array_equal(bp_val1, [32, 30])
            if debug: print("bp2: %s" % (bp_val2))
            assert np.array_equal(bp_val2, [32, 31])
            if debug: print("bp3: %s" % (bp_val3))
            assert np.array_equal(bp_val3, [32, 32])

            # Static BP
            vec1_static = waveflow.wavecomp_ops_module.wave_vec_add_dfx(tensor_const1, tensor_c1v, bp_i0="(16,14)", bp_i1="(16,14)")
            vec1_z_bp = waveflow.wavecomp_ops_module.wave_bp_gen(vec1_static)

            # Output BP is set to an artificially low value, which saturates the result. Subsequent 
            # characterization of the output tensor should mirror this saturation.
            vec2_static = waveflow.wavecomp_ops_module.wave_vec_add_dfx(tensor_const1, tensor_c1v, bp_i0="(16,14)", bp_i1="(16,14)", bp_o0="(16,14)")
            vec2_z_bp = waveflow.wavecomp_ops_module.wave_bp_gen(vec2_static)

            # Set one input with incorrect high BP. Should propagate to the output.
            vec3_static = waveflow.wavecomp_ops_module.wave_vec_add_dfx(tensor_const1, tensor_c2v, bp_i0="(16,16)")
            vec3_z_bp = waveflow.wavecomp_ops_module.wave_bp_gen(vec3_static)

            _, vz_val1, _, vz_val2, _, vz_val3 = sess.run([vec1_static, vec1_z_bp, vec2_static, vec2_z_bp, vec3_static, vec3_z_bp])

            if debug: print("vz_val1: %s" % (vz_val1))
            assert np.array_equal(vz_val1, [16, 13])
            if debug: print("vz_val2: %s" % (vz_val2))
            assert np.array_equal(vz_val2, [16, 14])
            if debug: print("vz_val3: %s" % (vz_val3))
            assert np.array_equal(vz_val3, [16, 14])

    # Dynamic data tests.
    for i in range(iterations):
        # pbar.update(i)
        # tf.reset_default_graph()

        with tf.Session('') as sess:
            with sess.graph.device(my_custom_dev):                
                t_init.run()                


    # pbar.finish()
    return True


if __name__ == "__main__":
    test_bp()

