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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "gemm_int.h"
#include "dfx_op_base.h"

using namespace tensorflow;

static bool first_time = true;

REGISTER_OP("WaveMatMulInt")
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

class WaveMatMulIntOp : public WaveDynFxPointOp {
public:
    explicit WaveMatMulIntOp(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"bp_i0", "bp_i1", "bp_o0"}) {
        // printf("Calling Wave matmul() ...\n");
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
        if (m_show_banner) {
            printf("WaveMatMulIntOp() init\n");
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

        auto a_fp = tensor_a.flat<float>();
        auto b_fp = tensor_b.flat<float>();
        auto z_fp = output->flat<float>();
        auto a_x = a_shape.dim_size(0);
        auto a_y = a_shape.dim_size(1);
        auto b_x = b_shape.dim_size(0);
        auto b_y = b_shape.dim_size(1);

        if (transpose_a_)
            std::swap(a_x, a_y);
        if (!transpose_b_)
            std::swap(b_x, b_y);

        auto n = (a_y + 3) & ~3;

        int32_t* m_a = (int32_t*)aligned_alloc(16, a_x * n * sizeof(int32_t));
        int32_t* m_b = (int32_t*)aligned_alloc(16, b_x * n * sizeof(int32_t));
        int32_t* m_z = (int32_t*)aligned_alloc(16, a_x * b_x * sizeof(int32_t));

        fxbp a_bp = get_fxbp(true, 0);
        matrix_in(a_bp, m_a, a_fp.data(), a_x, a_y, n, transpose_a_);
        fxbp b_bp = get_fxbp(true, 1);
        matrix_in(b_bp, m_b, b_fp.data(), b_x, b_y, n, !transpose_b_);

        int shift = 31 - __builtin_clz(n);
        if (n == (1 << shift))
            shift--;
        mm_nt(m_a, a_x, n, m_b, b_x, n, m_z, shift, m_use_stochastic_round);

        fxbp z_bp = get_fxbp(false, 0);
        auto bp = a_bp.m_bp + b_bp.m_bp;
        partial_out(z_bp, z_fp.data(), m_z, bp - shift, a_x * b_x);

        free(m_a);
        free(m_b);
        free(m_z);
    }
private:

    bool transpose_a_;
    bool transpose_b_;

    void matrix_in(fxbp& dest_bp, int32_t* m_out, const float* flat_arr, int x, int y,
                   int stride, bool transpose)
    {
        int padding = stride - y;

        if (dest_bp.m_bp == -1 || !dest_bp.m_initialized) {
            dest_bp.set_range_fp(flat_arr, flat_arr + x * y);
        }

        float k = (float)(1 << dest_bp.m_bp);
        if (transpose) {
            int32_t* m_ptr;
            for (int j = 0; j < y; j++) {
                m_ptr = m_out++;
                for (int i = 0; i < x; i++) {
                    *m_ptr = (int32_t)roundf(*flat_arr++ * k);
                    m_ptr += stride;
                }
            }
            for (int j = 0; j < padding; j++) {
                m_ptr = m_out++;
                for (int i = 0; i < x; i++) {
                    *m_ptr = 0;
                    m_ptr += stride;
                }
            }
        } else {
            for (int i = 0; i < x; i++) {
                for (int j = 0; j < y; j++)
                    *m_out++ = (int32_t)roundf(*flat_arr++ * k);
                for (int j = 0; j < padding; j++)
                    *m_out++ = 0;
            }
        }
    }

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

bool WaveMatMulIntOp::m_show_banner = true;


REGISTER_KERNEL_BUILDER(Name("WaveMatMulInt").Device(DEVICE_CPU), WaveMatMulIntOp);
