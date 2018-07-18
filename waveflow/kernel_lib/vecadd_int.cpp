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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "dyn_fx_pt.h"
#include "dfx_op_base.h"

#define USE_VECTOR_INSTRUCTIONS

#ifdef USE_VECTOR_INSTRUCTIONS
    #include <emmintrin.h>
#endif

using namespace tensorflow;

REGISTER_OP("WaveVecAddInt")
    .Input("a: float")
    .Input("b: float")
    .Output("z: float")
    .Attr("data_format: string = 'NHWC'")
    // Dynamic Fixed Point linkage
    .Attr("bp_i0: string = ''")
    .Attr("bp_i1: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

class WaveVecAddIntOp : public WaveDynFxPointOp {
public:
    explicit WaveVecAddIntOp(OpKernelConstruction* ctx)
        : WaveDynFxPointOp(ctx, {"bp_i0", "bp_i1", "bp_o0"}),
        m_mm_a_dfx(), m_mm_b_dfx(), m_mm_z_dfx() {
        string data_format_str;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
        OP_REQUIRES(ctx, (data_format_str == "NHWC"),
            errors::InvalidArgument(
            "wave_vec_add requires the data_format attribute to be NHWC"));
        if (m_show_banner) {
            printf("WaveVecAddIntOp() init\n");
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

        const int last_dim_a = a_shape.dims() - 1;

        DCHECK_EQ(b_shape.dims(), 1);
        DCHECK_EQ(a_shape.dim_size(last_dim_a), b_shape.dim_size(0));

        TensorShape out_shape(a_shape);
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
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        auto a_fp = tensor_a.flat<float>();
        auto b_fp = tensor_b.flat<float>();
        auto z_fp = output->flat<float>();
        auto a_size = a_fp.size();
        auto b_size = b_fp.size();
        auto z_size = z_fp.size();

        int32_t* m_a = (int32_t*)aligned_alloc(16, (a_size+4) * sizeof(int32_t));
        int32_t* m_b = (int32_t*)aligned_alloc(16, (b_size+4) * sizeof(int32_t));
        int32_t* m_z = (int32_t*)aligned_alloc(16, (z_size+4) * sizeof(int32_t));

        fxbp a_bp = get_fxbp(true, 0);

        partial_in(a_bp, m_a, a_fp.data(), a_size);
        partial_in(a_bp, m_b, b_fp.data(), b_size);

        memset(m_z, 0, z_size);

        int n = a_size / b_size;

#ifdef USE_VECTOR_INSTRUCTIONS
        for (int i=0; i<n; i++) {
            for (int j = 0; j<b_size; j+=4) {
                __m128i xa = _mm_loadu_si128((__m128i const*)&m_a[i*b_size+j]);
                __m128i xb = _mm_loadu_si128((__m128i const*)&m_b[j]);
                __m128i xz = _mm_add_epi32(xa, xb);

                _mm_storeu_si128((__m128i *) &m_z[i*b_size+j], xz);
            }
        }
#else
        for (int i=0; i<n; i++) {
            for (int j = 0; j<b_size; j++) {
                m_z[i*b_size+j] = m_a[i*b_size+j] + m_b[j];
            }
        }
#endif

        fxbp bp_out = get_fxbp(false, 0);
        partial_out(bp_out, z_fp.data(), m_z, a_bp.m_bp, z_size);

        free(m_a);
        free(m_b);
        free(m_z);
    }

private:
    DFXVector m_mm_a_dfx;
    DFXVector m_mm_b_dfx;
    DFXVector m_mm_z_dfx;

    static bool m_show_banner;
};

bool WaveVecAddIntOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveVecAddInt").Device(DEVICE_CPU), WaveVecAddIntOp);
