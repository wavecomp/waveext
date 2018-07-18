#!/usr/bin/env python3

# wf_kernel_lib.py 
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
#      Common import module for all Wave custom ops.
# 
# Author          : Ken Shiring
# Created On      : 02/28/2018
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

import os
import json


import tensorflow as tf

from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops

from google.protobuf import json_format

from keras.callbacks import TensorBoard


#  Get list of operators used in graph
#  Call Examples:
#    Keras: op_list(K.get_session().graph)
#       TF: op_list(tf.get_default_graph())
def op_list(graph):
    result = set()
    for o in graph.get_operations():
        if type(o.op_def) is not type(None):
            #print('node_def name: ', o.op_def.name)
            result.add(o.op_def.name)
    return result


repo_base = os.environ['WF_REPO_DIR']
wavecomp_ops_module = tf.load_op_library(repo_base+'/build/waveflow/kernel_lib/libwf_kernel_lib.so')
print("Waveflow module loaded")

# Global flag to indicate what kind of arithmetic to use. Valid values are:
# "tf" : vanilla Tensorflow operators
# "wf" : FP32 Wave operators
# "dfx" : Dynamic Fixed Point Wave operators
# "int" : Vectorized DFX operators
waveflow_arithmetic = "tf"


# Device functor
# ----------------------------------------------------------------------------
def wf_custom_dev(n):
    # This should include more complex decision logic (including specific
    # checks below). If a supported op is seen, the device type should be 'DPU'.
    # All else should fall back to CPU.
    # print("Got dev arg: %s\n%s" % (type(n), str(n)))
    # return "/device:CPU:0"
    return "/device:DPU:0"


# These functions help automate the process of monkey-patching the provided
# Tensorflow functions. This cuts down on the boilerplate code.
class wf_shim_obj(object):
    def __init__(self, func_entries, run_enable=None):
        self._fe = func_entries
        self._re = run_enable

    def _parse_target(self, arith):
        # Check for the target-augmented entry first. If no match,
        # fall back to the simple name. e.g.:
        # 1. 'int:gpu' , then
        # 2. 'int'
        g = test_util.IsGoogleCudaEnabled()
        if g:
            fulltype = arith + ":gpu"
        else:
            fulltype = arith + ":cpu"

        # print("fulltype: %s" % (fulltype))
        if fulltype in self._fe:
            # print("Found support")
            return fulltype

        assert arith in self._fe, "Fatal: can't find support for arithmetic %s" % (arith)
        return arith

    def run(self, *args, **kwargs):
        arith = waveflow_arithmetic
        # print("%s: Run()" % (self._fe['tf']))
        if self._re:
            # print("%s: Run enable defined" % (self._fe['tf']))
            # Apply the run filter first to see if we are allowed to run
            # the specialized implementation of this op. Some arithmetic
            # types have limits, and we have to drop back to the default
            # if we can't implement the given parameters.
            if not self._re(*args, **kwargs):
                arith = 'tf'
                #print("%s: Not enabled" % (self._fe['tf']))
            # else:
            #     print("%s: Enabled" % (self._fe['tf']))
        # See if this op has a specific target specified.
        arith = self._parse_target(arith)
        # print("Using arithmetic type: %s for %s" % (waveflow_arithmetic, str(self._fe[arith])))
        return self._fe[arith](*args, **kwargs)

def wf_resolve_func(func_name, optional=False):
    fields = func_name.split('.')
    assert len(fields) > 1, "Original op must specify a module"
    try:
        mod = globals()[fields[0]]
        for m in fields[1:]:
            mod = getattr(mod, m)
    except:
        assert optional, "Can't find object in hierarchy %s" % (func_name)
    return mod

def wf_subst(orig_op, name_entries, run_enable=None):
    original_func = wf_resolve_func(orig_op)
    func_entries = {'tf':original_func}
    for key, fname in name_entries.items():
        opt = ':' in key
        func_entries[key] = wf_resolve_func(fname, opt)
    shim_op = wf_shim_obj(func_entries, run_enable)
    # Replace the original
    # print("Replacing: %s with %s" % (original_func, shim_op.run))
    # original_func = shim_op.run
    return shim_op.run


# matmul()
# ----------------------------------------------------------------------------
tf.matmul = wf_subst("tf.matmul",
    {'wf': "wavecomp_ops_module.wave_mat_mul",
     'dfx': "wavecomp_ops_module.wave_mat_mul_dfx",
     'int:cpu': "wavecomp_ops_module.wave_mat_mul_int",
     'int:gpu': "wavecomp_ops_module.wave_mat_mul_cuda_int"})
math_ops.matmul = wf_subst("math_ops.matmul",
    {'wf': "wavecomp_ops_module.wave_mat_mul",
     'dfx': "wavecomp_ops_module.wave_mat_mul_dfx",
     'int:cpu': "wavecomp_ops_module.wave_mat_mul_int",
     'int:gpu': "wavecomp_ops_module.wave_mat_mul_cuda_int"})

# bias_add()
# ----------------------------------------------------------------------------
tf.nn.bias_add = wf_subst("tf.nn.bias_add",
    {'wf': "wavecomp_ops_module.wave_vec_add",
     'dfx': "wavecomp_ops_module.wave_vec_add_dfx",
     'int': "wavecomp_ops_module.wave_vec_add_int"})

nn_ops.bias_add = wf_subst("nn_ops.bias_add",
    {'wf': "wavecomp_ops_module.wave_vec_add",
     'dfx': "wavecomp_ops_module.wave_vec_add_dfx",
     'int': "wavecomp_ops_module.wave_vec_add_int"})


# add()
# ----------------------------------------------------------------------------
tf.add = wf_subst("tf.add",
    {'wf': "tf.add",
     'dfx': "wavecomp_ops_module.wave_add_dfx",
     'int': "wavecomp_ops_module.wave_add_dfx"})
gen_math_ops.add = wf_subst("gen_math_ops.add",
    {'wf': "gen_math_ops.add",
     'dfx': "wavecomp_ops_module.wave_add_dfx",
     'int': "wavecomp_ops_module.wave_add_dfx"})


# sub()
# ----------------------------------------------------------------------------
tf.subtract = wf_subst("tf.subtract",
    {'wf': "tf.subtract",
     'dfx': "wavecomp_ops_module.wave_sub_dfx",
     'int': "wavecomp_ops_module.wave_sub_dfx"})
# Needs support for pure INT operands to enable this.
# gen_math_ops.sub = wf_subst("gen_math_ops.sub",
#     {'wf': "gen_math_ops.sub",
#      'dfx': "wavecomp_ops_module.wave_sub_dfx",
#      'int': "wavecomp_ops_module.wave_sub_dfx"})


# mul()
# ----------------------------------------------------------------------------
tf.multiply = wf_subst("tf.multiply",
    {'wf': "tf.multiply",
     'dfx': "wavecomp_ops_module.wave_mul_dfx",
     'int': "wavecomp_ops_module.wave_mul_int"})
# Bugged, needs fixing
# math_ops.multiply = wf_subst("math_ops.multiply",
#     {'wf': "math_ops.multiply",
#      'dfx': "wavecomp_ops_module.wave_mul_dfx",
#      'int': "wavecomp_ops_module.wave_mul_int"})

# conv2d()
# ----------------------------------------------------------------------------
def conv2d_run_enable(*args, **kwargs):
    if kwargs['padding'] == 'SAME' and kwargs['strides'][1] != 1:
        return False
    return True

tf.nn.conv2d = wf_subst("tf.nn.conv2d",
    {'wf': "wavecomp_ops_module.wave_conv2d",
     'dfx': "wavecomp_ops_module.wave_conv2d_dfx",
     'int': "wavecomp_ops_module.wave_conv2d_dfx"}, conv2d_run_enable)
gen_nn_ops.conv2d = wf_subst("gen_nn_ops.conv2d",
    {'wf': "wavecomp_ops_module.wave_conv2d",
     'dfx': "wavecomp_ops_module.wave_conv2d_dfx",
     'int': "wavecomp_ops_module.wave_conv2d_dfx"}, conv2d_run_enable)


# maxpool()
# ----------------------------------------------------------------------------
tf.nn.max_pool = wf_subst("tf.nn.max_pool",
    {'wf': "tf.nn.max_pool",
     'dfx': "wavecomp_ops_module.wave_max_pool_dfx",
     'int': "wavecomp_ops_module.wave_max_pool_dfx"})


# avgpool()
# ----------------------------------------------------------------------------
tf.nn.avg_pool = wf_subst("tf.nn.avg_pool",
    {'wf': "tf.nn.avg_pool",
     'dfx': "wavecomp_ops_module.wave_avg_pool_dfx",
     'int': "wavecomp_ops_module.wave_avg_pool_dfx"})


# softmax()
# ----------------------------------------------------------------------------
tf.nn.softmax = wf_subst("tf.nn.softmax",
    {'wf': "tf.nn.softmax",
     'dfx': "wavecomp_ops_module.wave_softmax_dfx",
     'int': "wavecomp_ops_module.wave_softmax_dfx"})
# Note: The implemented op matches the interface of the low-level op
# gen_nn_ops.sparse_softmax_cross_entropy_with_logits
# tf.nn.sparse_softmax_cross_entropy_with_logits is a high-level wrapper that
# does some additional work and calls the low-level op.
# It has a different interface and should not be substituted
gen_nn_ops.sparse_softmax_cross_entropy_with_logits = wf_subst("gen_nn_ops.sparse_softmax_cross_entropy_with_logits",
    {'wf': "gen_nn_ops.sparse_softmax_cross_entropy_with_logits",
     'dfx': "wavecomp_ops_module.wave_sparse_softmax_cross_entropy_with_logits_int",
     'int': "wavecomp_ops_module.wave_sparse_softmax_cross_entropy_with_logits_int"})


# tanh()
# ----------------------------------------------------------------------------
tf.tanh = wf_subst("tf.tanh",
    {'wf': "tf.tanh",
     'dfx': "wavecomp_ops_module.wave_tanh_dfx",
     'int': "wavecomp_ops_module.wave_tanh_int"})
math_ops.tanh = wf_subst("math_ops.tanh",
    {'wf': "math_ops.tanh",
     'dfx': "wavecomp_ops_module.wave_tanh_dfx",
     'int': "wavecomp_ops_module.wave_tanh_int"})
tf.nn.tanh = wf_subst("tf.nn.tanh",
    {'wf': "tf.nn.tanh",
     'dfx': "wavecomp_ops_module.wave_tanh_dfx",
     'int': "wavecomp_ops_module.wave_tanh_int"})


# sigmoid()
# ----------------------------------------------------------------------------
tf.sigmoid = wf_subst("tf.sigmoid",
    {'wf': "tf.sigmoid",
     'dfx': "wavecomp_ops_module.wave_sigmoid_dfx",
     'int': "wavecomp_ops_module.wave_sigmoid_int"})
math_ops.sigmoid = wf_subst("math_ops.sigmoid",
    {'wf': "math_ops.sigmoid",
     'dfx': "wavecomp_ops_module.wave_sigmoid_dfx",
     'int': "wavecomp_ops_module.wave_sigmoid_int"})



# Gradients
# ----------------------------------------------------------------------------

@ops.RegisterGradient("WaveConv2D")
def _wave_conv2d_grad_cc(op, grad):
    dilations = op.get_attr("dilations")
    strides = op.get_attr("strides")
    padding = op.get_attr("padding")
    use_cudnn_on_gpu = op.get_attr("use_cudnn_on_gpu")
    data_format = op.get_attr("data_format")
    shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
    if padding == 'SAME' and strides[1] != 1:
        g_input = nn_ops.conv2d_backprop_input
        g_weight = nn_ops.conv2d_backprop_filter
    else:
        g_input = wavecomp_ops_module.wave_conv2d_gradient_input
        g_weight = wavecomp_ops_module.wave_conv2d_gradient_weight

    return [
      g_input(
          shape_0,
          op.inputs[1],
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          data_format=data_format),
      g_weight(
          op.inputs[0],
          shape_1,
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          data_format=data_format)
    ]


@ops.RegisterGradient("WaveConv2DDfx")
def _wave_conv2d_dfx_grad_cc(op, grad):
    dilations = op.get_attr("dilations")
    strides = op.get_attr("strides")
    padding = op.get_attr("padding")
    use_cudnn_on_gpu = op.get_attr("use_cudnn_on_gpu")
    data_format = op.get_attr("data_format")
    shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
    if padding == 'SAME' and strides[1] != 1:
        g_input = nn_ops.conv2d_backprop_input
        g_weight = nn_ops.conv2d_backprop_filter
    else:
        g_input = wavecomp_ops_module.wave_conv2d_dfx_gradient_input
        g_weight = wavecomp_ops_module.wave_conv2d_dfx_gradient_weight

    return [
      g_input(
          shape_0,
          op.inputs[1],
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          data_format=data_format),
      g_weight(
          op.inputs[0],
          shape_1,
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          data_format=data_format)
    ]


@ops.RegisterGradient("WaveMatMul")
def _wave_mat_mul_grad_cc(op, grad):
    grad_inputs = wavecomp_ops_module.wave_mat_mul(grad, op.inputs[1], transpose_b=True)
    grad_weights = wavecomp_ops_module.wave_mat_mul(op.inputs[0], grad, transpose_a=True)
    # grad_inputs = tf.matmul(grad, op.inputs[1], transpose_b=True)
    # grad_weights = tf.matmul(op.inputs[0], grad, transpose_a=True)
    return grad_inputs, grad_weights

@ops.RegisterGradient("WaveMatMulDfx")
def _wave_mat_mul_dfx_grad_cc(op, grad):
    grad_inputs = wavecomp_ops_module.wave_mat_mul_dfx(grad, op.inputs[1], transpose_b=True)
    grad_weights = wavecomp_ops_module.wave_mat_mul_dfx(op.inputs[0], grad, transpose_a=True)
    return grad_inputs, grad_weights

@ops.RegisterGradient("WaveMatMulInt")
def _wave_mat_mul_int_grad_cc(op, grad):
    grad_inputs = wavecomp_ops_module.wave_mat_mul_int(grad, op.inputs[1], transpose_b=True)
    grad_weights = wavecomp_ops_module.wave_mat_mul_int(op.inputs[0], grad, transpose_a=True)
    return grad_inputs, grad_weights

@ops.RegisterGradient("WaveMatMulCudaInt")
def _wave_mat_mul_cuda_int_grad_cc(op, grad):
    grad_inputs = wavecomp_ops_module.wave_mat_mul_cuda_int(grad, op.inputs[1], transpose_b=True)
    grad_weights = wavecomp_ops_module.wave_mat_mul_cuda_int(op.inputs[0], grad, transpose_a=True)
    return grad_inputs, grad_weights


@ops.RegisterGradient("WaveVecAdd")
def _wave_vec_add_grad_cc(op, grad):
    grad_inputs = grad
    # (batch, activations) -> (activations)
    # grad_bias = tf.reduce_sum(grad, axis=0)
    grad_bias = wavecomp_ops_module.wave_vec_add_grad(grad)
    return grad_inputs, grad_bias

@ops.RegisterGradient("WaveVecAddDfx")
def _wave_vec_add_grad_dfx_cc(op, grad):
    grad_inputs = grad
    # (batch, activations) -> (activations)
    grad_bias = wavecomp_ops_module.wave_vec_add_grad_dfx(grad)
    return grad_inputs, grad_bias

@ops.RegisterGradient("WaveVecAddInt")
def _wave_vec_add_grad_int_cc(op, grad):
    grad_inputs = grad
    # (batch, activations) -> (activations)
    grad_bias = wavecomp_ops_module.wave_vec_add_grad_int(grad)
    return grad_inputs, grad_bias


@ops.RegisterGradient("WaveAddDfx")
def _wave_add_grad_dfx_cc(op, grad):
  """Gradient for Add."""
  return grad, grad

@ops.RegisterGradient("WaveSubDfx")
def _wave_sub_grad_dfx_cc(op, grad):
  """Gradient for Sub."""
  return grad, -grad

@ops.RegisterGradient("WaveMulDfx")
def _wave_mul_grad_dfx_cc(op, grad):
  """The gradient of scalar multiplication."""
  x = op.inputs[0]
  y = op.inputs[1]
  return (wavecomp_ops_module.wave_mul_dfx(grad, y),
          wavecomp_ops_module.wave_mul_dfx(grad, x))

@ops.RegisterGradient("WaveMulInt")
def _wave_mul_grad_int_cc(op, grad):
  """The gradient of scalar multiplication."""
  x = op.inputs[0]
  y = op.inputs[1]
  return (wavecomp_ops_module.wave_mul_int(grad, y),
          wavecomp_ops_module.wave_mul_int(grad, x))

@ops.RegisterGradient("WaveMaxPoolDfx")
def _wave_max_pool_grad_dfx_cc(op, grad):
  return wavecomp_ops_module.wave_max_pool_grad_dfx(
      op.inputs[0],
      op.outputs[0],
      grad,
      op.get_attr("ksize"),
      op.get_attr("strides"),
      padding=op.get_attr("padding"),
      data_format=op.get_attr("data_format"))

@ops.RegisterGradient("WaveAvgPoolDfx")
def _wave_avg_pool_grad_dfx_cc(op, grad):
  return wavecomp_ops_module.wave_avg_pool_grad_dfx(
      array_ops.shape(op.inputs[0]),
      grad,
      op.get_attr("ksize"),
      op.get_attr("strides"),
      op.get_attr("padding"),
      data_format=op.get_attr("data_format"))

@ops.RegisterGradient("WaveSigmoidDfx")
def _wave_sigmoid_grad_dfx_cc(op, grad):
  y = op.outputs[0]  # y = sigmoid(x)
  # calculates grad * y * (1 - y)
  return wavecomp_ops_module.wave_sigmoid_grad_dfx(y, grad)

@ops.RegisterGradient("WaveSigmoidInt")
def _wave_sigmoid_grad_int_cc(op, grad):
  y = op.outputs[0]  # y = sigmoid(x)
  # calculates grad * y * (1 - y)
  return wavecomp_ops_module.wave_sigmoid_grad_int(y, grad)

@ops.RegisterGradient("WaveTanhDfx")
def _wave_tanh_grad_dfx_cc(op, grad):
  y = op.outputs[0]  # y = tanh(x)
  # calculates grad * (1 - y * y)
  return wavecomp_ops_module.wave_tanh_grad_dfx(y, grad)

@ops.RegisterGradient("WaveTanhInt")
def _wave_tanh_grad_int_cc(op, grad):
  y = op.outputs[0]  # y = tanh(x)
  # calculates grad * (1 - y * y)
  return wavecomp_ops_module.wave_tanh_grad_int(y, grad)

@ops.RegisterGradient("WaveSoftmaxDfx")
def _wave_softmax_grad_dfx_cc(op, grad):
  y = op.outputs[0]  # y = softmax(x)
  # calculates grad * y - sum(grad * y) * y
  return wavecomp_ops_module.wave_softmax_grad_dfx(y, grad)

@ops.RegisterGradient("WaveSparseSoftmaxCrossEntropyWithLogitsInt")
def _wave_sparse_softmax_cross_entropy_with_logits_grad_int_cc(op, grad_0, _):
  sparse_softmax_grad_without_gradient = array_ops.prevent_gradient(
      op.outputs[1],
      message="Currently there is no way to take the second "
      "derivative of sparse_softmax_cross_entropy_with_logits due to the fused "
      "implementation's interaction with tf.gradients()")
  grad_0 = array_ops.expand_dims(grad_0, -1)
  return wavecomp_ops_module.wave_mul_int(grad_0, sparse_softmax_grad_without_gradient), None


@ops.RegisterGradient("WaveQuantizeDfx")
def _wave_quantize_grad_dfx(op, grad):
  return grad, [0, 0]



class TF_TBLogger(object):

    def __init__(self, log_dir, enable_tb=False, enable_trace=False, unified_trace=False, arith_type='tf'):
        ''' Keeps Tensorboard and trace settings for each run. Helps to abstract away some of the
            hard to manage details from the TF main API.
        '''
        self._enable_tb = enable_tb
        self._enable_trace = enable_trace
        self._unified_trace = unified_trace
        self._log_dir = log_dir.rstrip('/')
        self._atype = arith_type
        if self._enable_tb:
            self._tb_log_dir = self._log_dir + '/' + arith_type
            print("Logging data to tensorboard, directory: %s" % (self._tb_log_dir))

        self._trace_interval = 10
        self._timeline_dict = None

    def init_session(self, sess):
        if self._enable_tb:
            self._tb_writer = tf.summary.FileWriter(self._tb_log_dir, sess.graph)
            self._tb_merged = tf.summary.merge_all()

        if self._enable_trace:
            ttype = {True:'unified', False:'individual'}
            print("Generating %s trace data" % (ttype[self._unified_trace]))
            self._options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self._run_metadata = tf.RunMetadata()
            self.write_graph(sess)
        else:
            self._options = None
            self._run_metadata = None
        self._sess = sess

    def write_graph(self, sess):
        jfile = '%s/graph_%s.json' % (self._log_dir, self._atype)
        print("Writing graph to JSON file %s" % (jfile))
        graph_def = sess.graph_def
        json_string = json_format.MessageToJson(graph_def)
        with open(jfile, 'w') as f:
            f.write(json_string)
            f.close()

    def gen_trace(self, iter):
        if iter % self._trace_interval == 0:
            return self._run_metadata
        return None

    def run_session(self, iter, ops, feed):
        if self._enable_tb: 
            ops_for_train = [ops, self._tb_merged]
        else:
            ops_for_train = ops

        train_results = self._sess.run(ops_for_train, feed_dict=feed, 
                options=self._options, run_metadata=self._run_metadata)
            
        if self._enable_tb:
            tb_summary = train_results[-1]
            self._tb_writer.add_summary(tb_summary, iter)
            md = self.gen_trace(iter)
            if md:
                try:
                    self._tb_writer.add_run_metadata(md, 'step_%d' % iter)
                    fetched_timeline = timeline.Timeline(self._run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    self.write_timeline(chrome_trace, iter)
                except Exception as e:
                    print(e)
            self._tb_writer.flush()
            return train_results[:-1]
        return train_results

    def write_timeline(self, trace, iter):
        if self._unified_trace:
            self.update_timeline(trace, iter)
        else:
            # Create the Timeline object, and write it to a json file
            tfile = '%s/timeline_%s_%d.json' % (self._log_dir, self._atype, iter)
            with open(tfile, 'w') as f:
                # print("Writing timeline %s" % (tfile))
                f.write(trace)

    def update_timeline(self, trace, iter):
        ''' Helper function to merge JSON traces from multiple iterations into a single
            plot.
            Copied from  Illarion Khlestov's blog entry here: 
            https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
        '''
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def close(self):
        if self._enable_trace and self._unified_trace:
            # Write out the final accumulated trace data.
            with open('%s/timeline_%s_full.json' % (self._log_dir, self._atype), 'w') as f:
                json.dump(self._timeline_dict, f)

        if self._enable_tb: 
            self._tb_writer.close()


class K_TBLogger(object):

    def __init__(self, log_dir, enable_tb=False, enable_trace=False, unified_trace=False, arith_type='tf'):
        ''' Keeps Tensorboard and trace settings for each run. Helps to abstract away some of the
            hard to manage details from the Keras API.
        '''
        self._enable_tb = enable_tb
        self._enable_trace = enable_trace
        self._unified_trace = unified_trace
        self._log_dir = log_dir.rstrip('/')
        self._atype = arith_type
        if self._enable_tb:
            self._tb_log_dir = self._log_dir + '/' + arith_type
            print("Logging data to tensorboard, directory: %s" % (self._tb_log_dir))
            self._tb = TensorBoard(log_dir=self._tb_log_dir)

        self._trace_interval = 10
        self._timeline_dict = None

        if self._enable_trace:
            ttype = {True:'unified', False:'individual'}
            print("Generating %s trace data" % (ttype[self._unified_trace]))
            self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self._run_metadata = tf.RunMetadata()

    def compile(self, model, *args, **kwargs):
        if self._enable_trace:
            assert 'run_metadata' not in kwargs
            kwargs['run_metadata'] = self._run_metadata
            # We can't support user options just yet.
            assert 'options' not in kwargs
            kwargs['options'] = self._run_options

        return model.compile(*args, **kwargs)


    def fit(self, model, *args, **kwargs):
        if self._enable_tb:
            if 'callbacks' in kwargs:
                kwargs['callbacks'].append(self._tb)
            else:
                kwargs['callbacks'] = [self._tb]

        r = model.fit(*args, **kwargs)

        if self._enable_trace:
            trace = timeline.Timeline(step_stats=self._run_metadata.step_stats)
            chrome_trace = trace.generate_chrome_trace_format()
            self.write_timeline(chrome_trace, 0)
        return r


    def write_timeline(self, trace, iter):
        if self._unified_trace:
            self.update_timeline(trace, iter)
        else:
            # Create the Timeline object, and write it to a json file
            tfile = '%s/timeline_%s_%d.json' % (self._log_dir, self._atype, iter)
            with open(tfile, 'w') as f:
                # print("Writing timeline %s" % (tfile))
                f.write(trace)


    def update_timeline(self, trace, iter):
        ''' Helper function to merge JSON traces from multiple iterations into a single
            plot.
            Copied from  Illarion Khlestov's blog entry here: 
            https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
        '''
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def close(self):
        if self._enable_trace and self._unified_trace:
            # Write out the final accumulated trace data.
            with open('%s/timeline_%s_full.json' % (self._log_dir, self._atype), 'w') as f:
                json.dump(self._timeline_dict, f)


class BP_Manager(object):
    ''' This class takes care of instantiating Tensorflow ops which need
        binary point management on inputs and outputs.
    '''
    def __init__(self, bp_func=wavecomp_ops_module.wave_bp_gen):
        self.debug = False
        self._tensor_map = {}
        self._bp_ops = []
        self._bp_func = bp_func

    def _find_or_create(self, t):
        ''' Find the BP function in the map, or create one.
        '''
        tname = t.op.name
        if tname in self._tensor_map:
            if self.debug: print("Found tensor %s in map" % (tname))
            return self._tensor_map[tname]
        o = self._bp_func(t)
        self._tensor_map[tname] = o
        if self.debug: print("Creating BP for tensor %s" % (tname))
        return o


    def wrap(self, tf_call, tensors):
        bp_ops = []
        bp_attrs = {}
        for i, t in enumerate(tensors):
            o = self._find_or_create(t)
            bp_ops.append(o)
            k = 'dfx_i%d' % (i)
            bp_attrs[k] = o.op.name

        self._bp_ops.extend(bp_ops)
        bp_attrs['bp_o'] = "(32,-1)"
        with tf.control_dependencies(bp_ops):
            op = tf_call(**bp_attrs)

        # Output will have a separate BP as well. However, this is not an a-priori
        # calculation, and requires a full-precision input to quantize. This means
        # we have to separate the quantization from the generation op.
        z_q = self._bp_func(op)
        z_qq = tf.stop_gradient(z_q)
        z = wavecomp_ops_module.wave_quantize_dfx(op, z_qq)
        # Register the output of the quantize func as the associated tensor
        self._tensor_map[z.op.name] = z_q
        return z

    def get_bp_ops(self):
        return self._bp_ops
