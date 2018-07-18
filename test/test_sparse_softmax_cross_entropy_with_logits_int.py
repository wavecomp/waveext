#!/usr/bin/env python3

# test_softmax.py
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
#      Test sparse softmax cross entropy loss
#      function and its gradient
#
# Author          : Stanislav Ocovaj
# Created On      : 06/27/2018
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
from tensorflow.python.ops import gen_nn_ops
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

def test_sparse_softmax_cross_entropy():

    tf.reset_default_graph()

    num_batches = 100
    num_classes = 10
    v_logits = tf.Variable(tf.truncated_normal([num_batches, num_classes], mean=0, stddev=5), name="logits")
    v_labels = tf.Variable(tf.random_uniform([num_batches], minval=0, maxval=num_classes, dtype=tf.int64), name="labels")
    v_grad = tf.Variable(tf.truncated_normal([num_batches], mean=0, stddev=1), name="grad")

    t_init = tf.global_variables_initializer()
    t_debug = False

    iters = 100

    widgets = ["Sparse softmax cross entropy int test: ", pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=iters)
    pbar.start()

    for i in range(iters):
        pbar.update(i)

        with tf.Session('') as sess:
            t_init.run()

            z_op, z_back = waveflow.wavecomp_ops_module.wave_sparse_softmax_cross_entropy_with_logits_int(features=v_logits, labels=v_labels)
            z_op2, z_back2 = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(features=v_logits, labels=v_labels)
            z_grad = tf.gradients(z_op, v_logits, v_grad)[0]
            z_grad2 = tf.gradients(z_op2, v_logits, v_grad)[0]
            z, zb, zg, z2, zb2, zg2, labels, logits, grad = sess.run([z_op, z_back, z_grad, z_op2, z_back2, z_grad2, v_labels, v_logits, v_grad])
            if t_debug:
                print("labels: %s" % (labels))
                print("logits: %s" % (logits))
                print("grad: %s" % (grad))
                print("z: %s" % (z))
                print("z2: %s" % (z2))
                print("zb: %s" % (zb))
                print("zb2: %s" % (zb2))
                print("zg: %s" % (zg))
                print("zg2: %s" % (zg2))

            assert compare_tensor(z, z2, "Output mismatch", 1e-3)
            assert compare_tensor(zb, zb2, "Backprop mismatch", 1e-3)
            assert compare_tensor(zg, zg2, "Gradient mismatch", 1e-3)

    pbar.finish()
    return True

if __name__ == "__main__":
    test_sparse_softmax_cross_entropy()
