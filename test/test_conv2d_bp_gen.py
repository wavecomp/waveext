#!/usr/bin/env python3

# test_conv2d_vp2.py 
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




def test_conv2d_vp2():
    ''' Run tests on the Wave custom conv2d operator. 
    '''
    tf.reset_default_graph()

    iterations = 10

    widgets = ["conv2d tests: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=iterations)
    pbar.start()


    # Interesting kernel variants to cycle through
    kernel_params = [
        {'t_n':100, 't_ci':1,  't_co':32,  't_h':28,  't_w':28,  'w_k':5},
        {'t_n':4,   't_ci':32, 't_co':32,  't_h':15,  't_w':15,  'w_k':3},
        {'t_n':1,   't_ci':4,  't_co':64,  't_h':16,  't_w':16,  'w_k':3},
        {'t_n':4,   't_ci':64, 't_co':128, 't_h':7,   't_w':7,   'w_k':5},
        {'t_n':4,   't_ci':8,  't_co':4,   't_h':224, 't_w':224, 'w_k':7},
        {'t_n':100, 't_ci':1,  't_co':32,  't_h':28,  't_w':28,  'w_k':1},
        {'t_n':1,   't_ci':1,  't_co':2,   't_h':4,   't_w':4,   'w_k':1}
    ]

    for i in range(iterations):
        pbar.update(i)
        tf.reset_default_graph()

        # NCHW
        p = kernel_params[i % len(kernel_params)]
        t_n = p['t_n']
        t_ci = p['t_ci']
        t_co = p['t_co']
        t_h = p['t_h']
        t_w = p['t_w']
        w_k = p['w_k']

        # N H W C
        activations = tf.get_variable("a", [t_n, t_h, t_w, t_ci], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        # K K Ci Co
        c2d_wts =     tf.get_variable("b", [w_k, w_k, t_ci, t_co], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))

        t_init = tf.global_variables_initializer()


        # SAME variant
        with tf.Session('') as sess:
            with sess.graph.device(my_custom_dev):                
                t_init.run()
                
                # The op output is the Dynamic Fixed Point. These binary point values will be automatically bound to 
                # the name of the op itself.
                wts_dfx = waveflow.wavecomp_ops_module.wave_bp_gen(c2d_wts)
                act_dfx = waveflow.wavecomp_ops_module.wave_bp_gen(activations)

                # On the consumer side, we bind the BP of each input to the binary point generating
                # ops. We also use a control dependency to enforce proper ordering of execution.
                with tf.control_dependencies([act_dfx, wts_dfx]):
                    z_op = waveflow.wavecomp_ops_module.wave_conv2d_dfx(activations, c2d_wts, strides=[1, 1, 1, 1], padding='SAME',
                    bp_i0=act_dfx.op.name, bp_i1=wts_dfx.op.name)
                
                # Base tensorflow. Only supports NHWC.
                z2_op = tf.nn.conv2d(activations, c2d_wts,
                    strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', use_cudnn_on_gpu=False)

                # Tensorflow has a bug which causes control ops to be pruned if they have no other data dependencies.
                # This means they have to be explicitly consumed in the run() command.
                z, z2, act_val, wts_val, dfx_val, _ = sess.run([z_op, z2_op, activations, c2d_wts, wts_dfx, act_dfx])

                # print("Got BP: %s" % (dfx_val))
                assert_str = "Failure on i: %d, mode: SAME, params: %s" % (i, str(p))
                if not compare_tensor(z, z2, assert_str):
                    print("activations: %s" % (act_val))
                    print("c2d_wts: %s" % (wts_val))
                    print("\n\n")
                    assert False


    pbar.finish()
    return True


if __name__ == "__main__":
    test_conv2d_vp2()

