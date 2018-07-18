#!/usr/bin/env python3

# test_tanh.py
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
#      Test hyperbolic tangent activation
#      function and its gradient
#
# Author          : Stanislav Ocovaj
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


import progressbar as pb
import numpy as np
import tensorflow as tf
import waveflow

def compare_tensor(z1, z2, msg, tol):
    ''' Run a compare on 2 tensors for equality. Report failure details.
    '''
    assert z1.shape == z2.shape, msg
    if not np.allclose(z1, z2, atol=tol):
        print("\n\n")
        d = ~np.isclose(z1, z2, atol=tol)
        print("z1 mismatch: %s" % (z1[d]))
        print("z2 mismatch: %s" % (z2[d]))
        print("at: %s" % (str(np.where(d))))
        print("Failure: %s" % (msg))
        return False

    return True

def test_tanh():

    tf.reset_default_graph()

    v_a = tf.Variable(tf.truncated_normal([10, 100], mean=0, stddev=5), name="a")
    v_da = tf.Variable(tf.truncated_normal([10, 100], mean=0, stddev=1), name="da")

    t_init = tf.global_variables_initializer()
    t_debug = False

    iters = 100

    widgets = ["Tanh BP test: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=iters)
    pbar.start()

    for i in range(iters):
        pbar.update(i)

        with tf.Session('') as sess:
            t_init.run()
    
            z_op = waveflow.wavecomp_ops_module.wave_tanh_int(v_a)
            z_op2 = tf.nn.tanh(v_a)
            z_grad = tf.gradients(z_op, v_a, v_da)[0]
            z_grad2 = tf.gradients(z_op2, v_a, v_da)[0]
            z, z2, zg, zg2, a, da = sess.run([z_op, z_op2, z_grad, z_grad2, v_a, v_da])
            if t_debug:
                print("a: %s" % (a))
                print("da: %s" % (da))
                print("z: %s" % (z))
                print("z2: %s" % (z2))
                print("zg: %s" % (zg))
                print("zg2: %s" % (zg2))

            assert compare_tensor(z, z2, "Output mismatch", 1e-3)
            assert compare_tensor(zg, zg2, "Gradient mismatch", 1e-3)

    pbar.finish()
    return True

if __name__ == "__main__":
    test_tanh()
