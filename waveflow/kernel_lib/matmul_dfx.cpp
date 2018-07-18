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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"

#include "Eigen/Core"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "dyn_fx_pt.h"
#include "gemm_dfx.h"
#include "dfx_op_base.h"



using namespace tensorflow;

static bool first_time = true;

REGISTER_OP("WaveMatMulDfx")
    .Input("a: float")
    .Input("b: float")
    .Output("z: float")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    // .Attr("T: float")
    // Dynamic Fixed Point linkage
    .Attr("bp_i0: string = ''")
    .Attr("bp_i1: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::MatMulShape);

class WaveMatMulDfxOp : public WaveDynFxPointOp {
public:
    explicit WaveMatMulDfxOp(OpKernelConstruction* ctx) 
    : WaveDynFxPointOp(ctx, {"bp_i0", "bp_i1", "bp_o0"}) {
        // printf("Calling Wave matmul() ...\n");
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));        
        if (m_show_banner) {
            printf("WaveMatMulDfxOp() init\n");
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

        // check dimensions
        DCHECK_EQ(a_shape.dims(), 2);
        DCHECK_EQ(b_shape.dims(), 2);

        // create output shape
        TensorShape out_shape(get_output_shape(a_shape, b_shape));
                
        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        // Create temp buffers. KS: there needs to be a better way to do this
        // in TF, but I can't find any docs on it. Apparently tensor dims aren't
        // known at construction time, which makes this hard.
        DFXMatrix2d m_a_dfx(a_shape.dim_size(0), a_shape.dim_size(1));
        DFXMatrix2d m_b_dfx(b_shape.dim_size(0), b_shape.dim_size(1));
        DFXMatrix2d m_z_dfx(out_shape.dim_size(0), out_shape.dim_size(1));

        fxbp bp_a = get_fxbp(true, 0);
        fxbp bp_b = get_fxbp(true, 1);
        fp2dfx(bp_a, m_a_dfx, tensor_a.matrix<float>().data());
        fp2dfx(bp_b, m_b_dfx, tensor_b.matrix<float>().data());

        int n;
        if (transpose_a_)
            n = a_shape.dim_size(0);
        else
            n = a_shape.dim_size(1);
        int shift = 31 - __builtin_clz(n);
        if (n == (1 << shift))
            shift--;
        fxbp sum_bp(bp_a.m_bp + bp_b.m_bp - shift, 32);

        dfx_clear(sum_bp, m_z_dfx, m_z_dfx.rows(), m_z_dfx.cols());

        if (!transpose_a_ && !transpose_b_) {
            mm_nn(false, sum_bp, m_a_dfx, m_a_dfx.rows(), m_a_dfx.cols(),
                  m_b_dfx, m_b_dfx.rows(), m_b_dfx.cols(), m_z_dfx);
        } else if (transpose_a_ && !transpose_b_) {
            mm_tn(false, sum_bp, m_a_dfx, m_a_dfx.rows(), m_a_dfx.cols(),
                  m_b_dfx, m_b_dfx.rows(), m_b_dfx.cols(), m_z_dfx);
        } else if (!transpose_a_ && transpose_b_) {
            mm_nt(false, sum_bp, m_a_dfx, m_a_dfx.rows(), m_a_dfx.cols(),
                  m_b_dfx, m_b_dfx.rows(), m_b_dfx.cols(), m_z_dfx);
        } else {
            mm_tt(false, sum_bp, m_a_dfx, m_a_dfx.rows(), m_a_dfx.cols(),
                  m_b_dfx, m_b_dfx.rows(), m_b_dfx.cols(), m_z_dfx);
        }

        // Transfer the BP result back into an FP tensor.
        fxbp bp_z = get_fxbp(false, 0);
        dfx2fp(bp_z, output->matrix<float>().data(), m_z_dfx);
    }
private:

    bool transpose_a_;
    bool transpose_b_;

    TensorShape get_output_shape(const TensorShape& a_shape, const TensorShape& b_shape)
    {
        if (!transpose_a_ && !transpose_b_) {
            DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(0));
            return TensorShape({a_shape.dim_size(0), b_shape.dim_size(1)});
        } else if (transpose_a_ && !transpose_b_) {
            DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(0));
            return TensorShape({a_shape.dim_size(1), b_shape.dim_size(1)});
        } else if (!transpose_a_ && transpose_b_) {
            DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(1));
            return TensorShape({a_shape.dim_size(0), b_shape.dim_size(0)});
        } else {
            DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(1));
            return TensorShape({a_shape.dim_size(1), b_shape.dim_size(0)});
        }
    }

private:
    static bool m_show_banner;
};

bool WaveMatMulDfxOp::m_show_banner = true;
 
 
REGISTER_KERNEL_BUILDER(Name("WaveMatMulDfx").Device(DEVICE_CPU), WaveMatMulDfxOp);
