/*
 * Copyright (c) 2010-2018 Wave Computing, Inc. and its applicable licensors.   
 * All rights reserved; provided, that any files identified as open source shall
 * be governed by the specific open source license(s) applicable to such files. 
 *
 * For any files associated with distributions under the Apache 2.0 license, 
 * full attribution to The Apache Software Foundation is given via the license 
 * below.
 */
/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include <stdlib.h>
#include <string.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/bcast.h"
#include "dyn_fx_pt.h"
#include "dfx_op_base.h"

#define USE_VECTOR_INSTRUCTIONS

#ifdef USE_VECTOR_INSTRUCTIONS
#include <emmintrin.h>
#include <smmintrin.h>
#endif

using namespace tensorflow;

REGISTER_OP("WaveMulInt")
    .Input("a: float")
    .Input("b: float")
    .Output("z: float")
    .Attr("data_format: string = 'NHWC'")
    // Dynamic Fixed Point linkage
    .Attr("bp_i0: string = ''")
    .Attr("bp_i1: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);


class WaveMulIntOp : public WaveDynFxPointOp {
public:

  explicit WaveMulIntOp(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"bp_i0", "bp_i1", "bp_o0"}) {
      string data_format_str;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
      OP_REQUIRES(ctx, (data_format_str == "NHWC"),
          errors::InvalidArgument(
          "wave_mul requires the data_format attribute to be NHWC"));
      if (m_show_banner) {
          printf("WaveMulIntOp() init\n");
          m_show_banner = false;
      }
    }

  void Compute(OpKernelContext* context) override {

      DCHECK_EQ(2, context->num_inputs());
      const Tensor& tensor_a = context->input(0);
      const Tensor& tensor_b = context->input(1);

      // check shapes of input and weights
      const TensorShape& a_shape = tensor_a.shape();
      const TensorShape& b_shape = tensor_b.shape();
      // Broadcast preparation
      BCast bcast(BCast::FromShape(a_shape), BCast::FromShape(b_shape));
      if (!bcast.IsValid()) {
        context->SetStatus(errors::InvalidArgument("Incompatible shapes: ",
            a_shape.DebugString(), " vs. ", b_shape.DebugString()));
        return;
      }

      const TensorShape output_shape = BCast::ToShape(bcast.output_shape());

      // create output tensor
      Tensor* output = NULL;
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                        {0, 1}, 0, output_shape, &output));
      ndims = static_cast<int>(bcast.x_reshape().size());

      auto a_flat = tensor_a.flat<float>();
      auto b_flat = tensor_b.flat<float>();
      auto z_flat = output->flat<float>();
      auto a_size = a_flat.size();
      auto b_size = b_flat.size();
      auto z_size = z_flat.size();

      m_a_shape = BCast::ToShape(bcast.x_reshape());
      m_b_shape = BCast::ToShape(bcast.y_reshape());
      m_z_shape = BCast::ToShape(bcast.result_shape());

      if (ndims <= 1) {
          // Inputs transformed to 1D tensor
          int32_t* m_a = (int32_t*)aligned_alloc(16, (a_size+4) * sizeof(int32_t));
          int32_t* m_b = (int32_t*)aligned_alloc(16, (b_size+4) * sizeof(int32_t));
          int32_t* m_z = (int32_t*)aligned_alloc(16, (z_size+4) * sizeof(int32_t));

          fxbp a_bp = get_fxbp(true, 0);
          fxbp b_bp = get_fxbp(true, 1);

          partial_in(a_bp, m_a, a_flat.data(), a_size);
          partial_in(b_bp, m_b, b_flat.data(), b_size);

          int num_a = tensor_a.NumElements();
          int num_b = tensor_b.NumElements();

          if (num_a == 1 && num_b == 1) {
              int32_t res = m_a[0] * m_b[0];
#ifdef USE_VECTOR_INSTRUCTIONS
              __m128i r = _mm_set_epi32(res, res, res, res);
              for (int i = 0; i < z_size; i+=4) {
                _mm_store_si128((__m128i *) &m_z[i], r);
              }
#else
              for (int i = 0; i < z_size; i++) {
                m_z[i] = res;
              }
#endif
          } else if (num_a == 1) {
#ifdef USE_VECTOR_INSTRUCTIONS
              __m128i xa = _mm_set_epi32(m_a[0], m_a[0], m_a[0], m_a[0]);
              for (int i = 0; i < z_size; i+=4) {
                  __m128i xb = _mm_load_si128((__m128i const*)&m_b[i]);
                  __m128i xz = _mm_mullo_epi32(xa, xb);
                  _mm_store_si128((__m128i *) &m_z[i], xz);
              }
#else
              for (int i = 0; i < z_size; i++) {
                m_z[i] = m_a[0] * m_b[i];
              }
#endif
          } else if (num_b == 1) {
#ifdef USE_VECTOR_INSTRUCTIONS
              __m128i xb = _mm_set_epi32(m_b[0], m_b[0], m_b[0], m_b[0]);
              for (int i = 0; i < z_size; i+=4) {
                  __m128i xa = _mm_load_si128((__m128i const*)&m_a[i]);
                  __m128i xz = _mm_mullo_epi32(xa, xb);
                  _mm_store_si128((__m128i *) &m_z[i], xz);
              }
#else
              for (int i = 0; i < z_size; i++) {
                m_z[i] = m_a[i] * m_b[0];
              }
#endif
          } else {
#ifdef USE_VECTOR_INSTRUCTIONS
              for (int i = 0; i < z_size; i+=4) {
                  __m128i xa = _mm_load_si128((__m128i const*)&m_a[i]);
                  __m128i xb = _mm_load_si128((__m128i const*)&m_b[i]);
                  __m128i xz = _mm_mullo_epi32(xa, xb);
                  _mm_store_si128((__m128i *) &m_z[i], xz);
              }
#else
              for (int i = 0; i < z_size; i++) {
                m_z[i] = m_a[i] * m_b[i];
              }
#endif
          }

          fxbp z_bp = get_fxbp(false, 0);
          partial_out(z_bp, z_flat.data(), m_z, a_bp.m_bp + b_bp.m_bp, z_size);

          free(m_a);
          free(m_b);
          free(m_z);
      } else if (ndims == 2) {
          // inputs are 2D tensors
          op_2d(z_flat.data(), a_flat.data(), b_flat.data(), z_size,
                bcast.x_reshape()[0], bcast.x_reshape()[1],
                bcast.y_reshape()[0], bcast.y_reshape()[1]);

      } else if (ndims < 6) {
          op_nd(z_flat.data(), a_flat.data(), b_flat.data(), ndims);
      } else {
          context->SetStatus(errors::Unimplemented("Broadcast between ",
              context->input(0).shape().DebugString(), " and ",
              context->input(1).shape().DebugString(), " is not supported yet."));
      }
  }

private:
  TensorShape m_a_shape;
  TensorShape m_b_shape;
  TensorShape m_z_shape;
  int ndims;

  static bool m_show_banner;

  void op_2d(float* z_data, const float* a_data, const float* b_data,
             int z_size, int a_rows, int a_cols, int b_rows, int b_cols) {

      int32_t* m_a = (int32_t*)aligned_alloc(16, (a_rows*a_cols+4) * sizeof(int32_t));
      int32_t* m_b = (int32_t*)aligned_alloc(16, (b_rows*b_cols+4) * sizeof(int32_t));
      int32_t* m_z = (int32_t*)aligned_alloc(16, (z_size+4) * sizeof(int32_t));

      fxbp a_bp = get_fxbp(true, 0);
      fxbp b_bp = get_fxbp(true, 1);

      partial_in(a_bp, m_a, a_data, a_rows*a_cols);
      partial_in(b_bp, m_b, b_data, b_rows*b_cols);

      int j, k;

      if (a_rows == 1) {
          if (b_cols == 1) {
#ifdef USE_VECTOR_INSTRUCTIONS
              // a_rows=1, b_cols=1
              for (int i = 0; i < b_rows; i++) {
                  __m128i xb = _mm_set_epi32(m_b[i], m_b[i], m_b[i], m_b[i]);
                  for (int j = 0; j < a_cols; j+=4) {
                      __m128i xa = _mm_loadu_si128((__m128i const*)&m_a[j]);
                      __m128i xz = _mm_mullo_epi32(xa, xb);
                      _mm_storeu_si128((__m128i *) &m_z[i*a_cols+j], xz);
                  }
              }
#else
              for (int i = 0; i < z_size; i++) {
                  j = i % a_cols;
                  k = i / a_cols;
                  m_z[i] = m_a[j] * m_b[k];
              }
#endif
          } else {
#ifdef USE_VECTOR_INSTRUCTIONS
              // a_rows=1, a_cols==b_cols
              for (int i = 0; i < b_rows; i++) {
                  for (int j = 0; j < b_cols; j+=4) {
                      __m128i xa = _mm_loadu_si128((__m128i const*)&m_a[j]);
                      __m128i xb = _mm_loadu_si128((__m128i const*)&m_b[i*b_cols+j]);
                      __m128i xz = _mm_mullo_epi32(xa, xb);
                      _mm_storeu_si128((__m128i *) &m_z[i*b_cols+j], xz);
                  }
              }
#else
              for (int i = 0; i < z_size; i++) {
                  j = i % a_cols;
                  m_z[i] = m_a[j] * m_b[i];
              }
#endif
          }
      } else if(b_rows == 1) {
          if (a_cols == 1) {
#ifdef USE_VECTOR_INSTRUCTIONS
              // a_cols=1, b_rows=1
              for (int i = 0; i < a_rows; i++) {
                  __m128i xa = _mm_set_epi32(m_a[i], m_a[i], m_a[i], m_a[i]);
                  for (int j = 0; j < b_cols; j+=4) {
                      __m128i xb = _mm_loadu_si128((__m128i const*)&m_b[j]);
                      __m128i xz = _mm_mullo_epi32(xa, xb);
                      _mm_storeu_si128((__m128i *) &m_z[i*b_cols+j], xz);
                  }
              }
#else
              for (int i = 0; i < z_size; i++) {
                  j = i / b_cols;
                  k = i % b_cols;
                  m_z[i] = m_a[j] * m_b[k];
              }
#endif
          } else {
#ifdef USE_VECTOR_INSTRUCTIONS
              //  b_rows=1, a_cols==b_cols
              for (int i = 0; i < a_rows; i++) {
                  for (int j = 0; j < a_cols; j+=4) {
                      __m128i xa = _mm_loadu_si128((__m128i const*)&m_a[i*a_cols+j]);
                      __m128i xb = _mm_loadu_si128((__m128i const*)&m_b[j]);
                      __m128i xz = _mm_mullo_epi32(xa, xb);
                      _mm_storeu_si128((__m128i *) &m_z[i*a_cols+j], xz);
                  }
              }
#else
              for (int i = 0; i < z_size; i++) {
                  k = i % b_cols;
                  m_z[i] = m_a[i] * m_b[k];
              }
#endif
          }
      } else if (a_cols == 1) {
#ifdef USE_VECTOR_INSTRUCTIONS
              // a_cols=1, a_rows==b_rows
              for (int i = 0; i < a_rows; i++) {
                  __m128i xa = _mm_set_epi32(m_a[i], m_a[i], m_a[i], m_a[i]);
                  for (int j = 0; j < b_cols; j+=4) {
                      __m128i xb = _mm_loadu_si128((__m128i const*)&m_b[i*b_cols+j]);
                      __m128i xz = _mm_mullo_epi32(xa, xb);
                      _mm_storeu_si128((__m128i *) &m_z[i*b_cols+j], xz);
                  }
              }
#else
              for (int i = 0; i < z_size; i++) {
                  j = i / b_cols;
                  m_z[i] = m_a[j] * m_b[i];
              }
#endif
      } else if (b_cols == 1) {
#ifdef USE_VECTOR_INSTRUCTIONS
              // b_cols=1, a_rows==b_rows
              for (int i = 0; i < b_rows; i++) {
                  __m128i xb = _mm_set_epi32(m_b[i], m_b[i], m_b[i], m_b[i]);
                  for (int j = 0; j < a_cols; j+=4) {
                      __m128i xa = _mm_loadu_si128((__m128i const*)&m_a[i*a_cols+j]);
                      __m128i xz = _mm_mullo_epi32(xa, xb);
                      _mm_storeu_si128((__m128i *) &m_z[i*a_cols+j], xz);
                  }
              }
#else
              for (int i = 0; i < z_size; i++) {
                  k = i / a_cols;
                  m_z[i] = m_a[i] * m_b[k];
              }
#endif
      }

      fxbp bp_out = get_fxbp(false, 0);
      partial_out(bp_out, z_data, m_z, a_bp.m_bp + b_bp.m_bp, z_size);
      free(m_a);
      free(m_b);
      free(m_z);
  }

  void op_nd(float* z_data, const float* a_data, const float* b_data,
             int curr_num_dims) {
      int a_offset = 1;
      int b_offset = 1;
      int z_offset = 1;
      int top_dim = ndims-curr_num_dims;

      for (int i=ndims-1; i>top_dim; i--) {
          a_offset *= m_a_shape.dim_size(i);
          b_offset *= m_b_shape.dim_size(i);
          z_offset *= m_z_shape.dim_size(i);
      }

      for (int i = 0; i < m_z_shape.dim_size(top_dim); i++) {
          int j = (m_a_shape.dim_size(top_dim) == 1) ? 0 : i;
          int k = (m_b_shape.dim_size(top_dim) == 1) ? 0 : i;

          const float* a_in = a_data + j*a_offset;
          const float* b_in = b_data + k*b_offset;
          float* z_out = z_data + i*z_offset;

          if (curr_num_dims > 3) {
              op_nd(z_out, a_in, b_in, curr_num_dims-1);
          } else {
              op_2d(z_out, a_in, b_in,
                  m_z_shape.dim_size(ndims-1)*m_z_shape.dim_size(ndims-2),
                  m_a_shape.dim_size(ndims-2), m_a_shape.dim_size(ndims-1),
                  m_b_shape.dim_size(ndims-2), m_b_shape.dim_size(ndims-1));
          }
      }
  }
};

bool WaveMulIntOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveMulInt").Device(DEVICE_CPU), WaveMulIntOp);
