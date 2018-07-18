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
#      Unit test for quantization
#      with stochastic rounding
# 
# Author          : Stanislav Ocovaj
# Created On      : 06/12/2018
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
from scipy.stats import binom_test as binom_test

import waveflow

debug = True

def binomial_test(x, val, orig_bp, bp, n):
    val_int = round(val * 2 ** orig_bp)
    bp_diff = orig_bp - bp
    val_reduced = val_int >> bp_diff
    v_0 = val_reduced / (1 << bp)     # low value
    v_1 = (val_reduced+1) / (1 << bp) # high value
    p_1 = (val_int & ((1 << bp_diff)-1)) / (1 << bp_diff)  # probability of rounding to high value
    exp_n1 = round(p_1 * n)           # expected number of occurrences of high value
    exp_n0 = n - exp_n1               # expected number of occurrences of low value
    act_n1 = 0                        # actual number of occurrences of high value
    act_n0 = 0                        # actual number of occurrences of low value
    for v in x:
        if v == v_0: act_n0 += 1
        if v == v_1: act_n1 += 1
    p = binom_test(act_n1, n, p_1)

    print("value = %s, bp = %s" % (val, bp))
    print("binomial test p-value = %s" % (p))

    if debug:
        #print("x: %s" % (x))
        print("exp_n0 = %s, act_n0 = %s" % (exp_n0, act_n0))
        print("exp_n1 = %s, act_n1 = %s" % (exp_n1, act_n1))

    assert act_n0+act_n1 == n         # assert that there are no values other than high and low
    #assert p > 0.01                   # assert that we can reject the null hypothesis
    #                                  # at 1% significance level
    assert abs(act_n0 - exp_n0)/n < 0.01  # assert that the actual probabilities are
    assert abs(act_n1 - exp_n1)/n < 0.01  # close enough to the expected

def test_quantize_stochastic_dfx():
    ''' Run tests on the Wave custom conv2d operator. 
    '''
    tf.reset_default_graph()

    val1 = 0.6
    n = 100000
    tensor_const1 = tf.get_variable("const_a", initializer=np.full(shape=(n), fill_value=val1, dtype=np.float32), dtype=tf.float32)

    t_init = tf.global_variables_initializer()

    # Static data tests. These only need to run once.
    with tf.Session('') as sess:
        t_init.run()
        
        const1_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const1, bp_o0='(16,14)')
        const1_q = waveflow.wavecomp_ops_module.wave_quantize_dfx(tensor_const1, const1_bp, use_stochastic_round=True)

        val1_const, val1_bp, val1_q = sess.run([tensor_const1, const1_bp, const1_q])

        binomial_test(val1_q, val1, 15, 14, n)

        const1_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const1, bp_o0='(16,12)')
        const1_q = waveflow.wavecomp_ops_module.wave_quantize_dfx(tensor_const1, const1_bp, use_stochastic_round=True)

        val1_const, val1_bp, val1_q = sess.run([tensor_const1, const1_bp, const1_q])

        binomial_test(val1_q, val1, 15, 12, n)

        const1_bp = waveflow.wavecomp_ops_module.wave_bp_gen(tensor_const1, bp_o0='(16,10)')
        const1_q = waveflow.wavecomp_ops_module.wave_quantize_dfx(tensor_const1, const1_bp, use_stochastic_round=True)

        val1_const, val1_bp, val1_q = sess.run([tensor_const1, const1_bp, const1_q])

        binomial_test(val1_q, val1, 15, 10, n)

    return True


if __name__ == "__main__":
    test_quantize_stochastic_dfx()
