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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/bcast.h"

#include <stdlib.h>

#include "dyn_fx_pt.h"

using namespace tensorflow;


class WaveBinDfxOp : public OpKernel {
public:
  typedef std::vector<DynFxPoint>    DFXVector;

  explicit WaveBinDfxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  
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
#if 0
        printf("Got shapes: a: [%d](%d, %d), b: [%d](%d)\n",
               a_shape.dims(),
               a_shape.dim_size(0), a_shape.dim_size(1),
               b_shape.dims(), b_shape.dim_size(0));
        printf("shape z: [%d](%d, %d)\n", out_shape.dims(),
               out_shape.dim_size(0), out_shape.dim_size(1));
#endif

      // create output tensor
      Tensor* output = NULL;
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                        {0, 1}, 0, output_shape, &output));
      ndims = static_cast<int>(bcast.x_reshape().size());

      auto a_flat = tensor_a.flat<float>();
      auto b_flat = tensor_b.flat<float>();
      auto z_flat = output->flat<float>();

      m_a_shape = BCast::ToShape(bcast.x_reshape());
      m_b_shape = BCast::ToShape(bcast.y_reshape());
      m_z_shape = BCast::ToShape(bcast.result_shape());

      if (ndims <= 1) {
          DFXVector a_vp_vec, b_vp_vec, z_vp_vec;
          a_vp_vec.resize(a_flat.size());
          b_vp_vec.resize(b_flat.size());
          z_vp_vec.resize(z_flat.size());

          partial_in(a_vp_vec, a_flat.data());
          partial_in(b_vp_vec, b_flat.data());

          for (int i = 0; i < z_flat.size(); i++) {
              int j = (tensor_a.NumElements() == 1) ? 0 : i;
              int k = (tensor_b.NumElements() == 1) ? 0 : i;
              binop(z_vp_vec[i], a_vp_vec[j], b_vp_vec[k]);
          }

          partial_out(z_flat.data(), z_vp_vec);
      } else if (ndims == 2) {
          op_2d(z_flat.data(), a_flat.data(), b_flat.data(), z_flat.size(),
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
   
  void partial_in(DFXVector& m_out, const float* flat_arr) {
      fxbp dest_bp(0, 16);
      dest_bp.set_range_fp(flat_arr, flat_arr + m_out.size());

      for (int i = 0; i < m_out.size(); i++) {
          m_out[i].set_fxbp(dest_bp);
          m_out[i] = flat_arr[i];
      }
  }

  void partial_out(float* conv_out, const DFXVector& m_out) {
      fxbp out_bp(0, 16);

      out_bp.set_range_dfx(m_out.data(), m_out.data()+m_out.size());
      for (int i = 0; i < m_out.size(); i++) {
          DynFxPoint v(out_bp);
          v = m_out[i];
          conv_out[i] = v.to_fp();
      }
  }

  void op_2d(float* z_data, const float* a_data, const float* b_data, 
             int z_offset, int a_rows, int a_cols, int b_rows, int b_cols) {
      DFXVector a_vp_vec, b_vp_vec, z_vp_vec;
      a_vp_vec.resize(a_rows*a_cols);
      b_vp_vec.resize(b_rows*b_cols);
      z_vp_vec.resize(z_offset);

      partial_in(a_vp_vec, a_data);
      partial_in(b_vp_vec, b_data);

      for (int i = 0; i < z_offset; i++) {
          int j, k;
          if (a_rows == 1) {
              j = i % (a_rows * a_cols);
              if (b_cols == 1) {
                  k = i / (a_rows * a_cols);
              } else {
                  k = i;
              }
          } else if(b_rows == 1) {
              if (a_cols == 1) {
                  j = i / (b_rows * b_cols);
              } else {
                  j = i;
              }
              k = i % (b_rows * b_cols);
          } else if (a_cols == 1) {
              j = i / b_cols;
              k = i;
          } else if (b_cols == 1) {
              j = i;
              k = i / a_cols;
          }

          binop(z_vp_vec[i], a_vp_vec[j], b_vp_vec[k]);
      }

      partial_out(z_data, z_vp_vec);
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
  // Overrride this function
  virtual void binop(DynFxPoint& c, const DynFxPoint& a, const DynFxPoint& b) = 0;
};

