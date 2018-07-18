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

#include "math_ops.h"

namespace tensorflow {

    REGISTER_OP("WcReducedpMatMul")
        .Input("a: int16")
        .Input("b: int16")
        .Input("a_point: int32")
        .Input("b_point: int32")
        .Input("out_point: int32")
        .Output("out: int16")
        .Attr("transpose_a: bool = false")
        .Attr("transpose_b: bool = false")
        .Attr(
            "round_mode: {'STOCHASTIC', 'HALF_AWAY_FROM_ZERO', 'HALF_TO_EVEN'} = "
            "'HALF_AWAY_FROM_ZERO'")
        .SetShapeFn([](shape_inference::InferenceContext* c) {
        //TF_RETURN_IF_ERROR(shape_inference::MatMulShape(c));
          shape_inference::ShapeHandle unused;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));

          return Status::OK();
        })
        .Doc(R"doc(
    Perform a quantized matrix multiplication of  `a` by the matrix `b`.

    The inputs must be two-dimensional matrices and the inner dimension of
    `a` (after being transposed if `transpose_a` is non-zero) must match the
    outer dimension of `b` (after being transposed if `transposed_b` is
    non-zero).

    a: Must be a two-dimensional tensor.
    b: Must be a two-dimensional tensor.
    transpose_a: If true, `a` is transposed before multiplication.
    transpose_b: If true, `b` is transposed before multiplication.

    )doc");


    
    class WcReducedpMatMulOp : public OpKernel {
    public:
      explicit WcReducedpMatMulOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
        printf("matmul constructor\n");
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
        // right now we are not supporting transpost
        OP_REQUIRES(ctx, !(transpose_a_ | transpose_b_),
            errors::InvalidArgument("In[0] is not a matrix"));    
    }

    void Compute(OpKernelContext* ctx) override {
        printf("compute matmul\n");
        const Tensor& a = ctx->input(0);
        const Tensor& b = ctx->input(1);

        // Check that the dimensions of the two matrices are valid.
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
            errors::InvalidArgument("In[0] is not a matrix"));
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
            errors::InvalidArgument("In[1] is not a matrix"));
        Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
        dim_pair[0].first = transpose_a_ ? 0 : 1;
        dim_pair[0].second = transpose_b_ ? 1 : 0;

        OP_REQUIRES(
            ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
            errors::InvalidArgument(
                "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
                ", In[1]: ", b.shape().DebugString()));
        int a_dim_remaining = 1 - dim_pair[0].first;
        int b_dim_remaining = 1 - dim_pair[0].second;
        TensorShape out_shape(
            {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

        if (out->NumElements() == 0) {
          // If a has shape [0, x] or b has shape [x, 0], the output shape
          // is a 0-element matrix, so there is nothing to do.
          printf("no out elements\n");
          return;
      }

      if (a.NumElements() == 0 || b.NumElements() == 0) {
          // If a has shape [x, 0] and b has shape [0, y], the
          // output shape is [x, y] where x and y are non-zero, so we fill
          // the output with zeros.
          //functor::SetZeroFunctor<Device, T> f;
          //f(ctx->eigen_device<Device>(), out->flat<T>());
          printf("no a or b elements\n");
          return;
      }

      auto a_m = a.matrix<int16>();
      auto b_m = b.matrix<int16>();
      auto output = out->matrix<int16>();
      output(0,0) = 100;
      printf("end %i\n", output(0,0));
      for (int i=0; i< a.dim_size(0); i++)
          for (int j =0; j < b.dim_size(1); j++) {
              printf("generating out(%i, %i)\n", i, j);
              int64_t val = 0;
              for (int k = 0; k < a.dim_size(1); k++) {
                  val += a_m(i, k)*b_m(k, j);
              }
              output(i, j) = val >> 8;
          }

      }

  private:

      bool transpose_a_;
      bool transpose_b_;
};

  REGISTER_KERNEL_BUILDER(Name("WcReducedpMatMul").Device(DEVICE_CPU), \
    WcReducedpMatMulOp);

}
