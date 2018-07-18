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

#define USE_VECTOR_INSTRUCTIONS

#if defined(USE_VECTOR_INSTRUCTIONS)
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#endif

#include "dyn_fx_pt.h"
#include "dfx_op_base.h"

using namespace tensorflow;

REGISTER_OP("WaveTanhGradInt")
    .Input("a: float")
    .Input("g: float")
    .Output("z: float")
    .Attr("bp_i0: string = ''")
    .Attr("bp_i1: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

class WaveTanhGradIntOp : public WaveDynFxPointOp {
public:
    typedef std::vector<DynFxPoint> DFXVector;

    explicit WaveTanhGradIntOp(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"bp_i0", "bp_i1", "bp_o0"})
    {
        if (m_show_banner) {
            printf("WaveTanhGradIntOp() init\n");
            m_show_banner = false;
        }
    }

    void Compute(OpKernelContext* context) override
    {
        DCHECK_EQ(2, context->num_inputs());

        const Tensor& tensor_a = context->input(0);
        const Tensor& tensor_g = context->input(1);

        TensorShape out_shape(tensor_a.shape());
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        auto a_fp = tensor_a.flat<float>();
        auto g_fp = tensor_g.flat<float>();
        auto z_fp = output->flat<float>();
        auto n = a_fp.size();
        auto n_ext = (n + 3) & ~3;

        int32_t* m_a = (int32_t*)aligned_alloc(16, n_ext * sizeof(int32_t));
        int32_t* m_g = (int32_t*)aligned_alloc(16, n_ext * sizeof(int32_t));
        int32_t* m_z = (int32_t*)aligned_alloc(16, n_ext * sizeof(int32_t));

        fxbp a_bp(14, 16);
        partial_in(a_bp, m_a, a_fp.data(), n);
        fxbp g_bp = get_fxbp(true, 1);
        partial_in(g_bp, m_g, g_fp.data(), n);

#if defined(USE_VECTOR_INSTRUCTIONS)
        __m128i xmask = _mm_set1_epi32(0x3fff);
        __m128i xround = _mm_set1_epi32(0x2000);
        __m128i xbias = _mm_set1_epi32(0x4000);
        __m128i xone = _mm_set1_epi32(1);
        __m128i *a, *g, *z;

        a = (__m128i *)m_a;
        g = (__m128i *)m_g;
        z = (__m128i *)m_z;
        for (int i = 0; i < n_ext>>2; i++) {
            __m128i xt, xv;

            xv = _mm_load_si128((__m128i*)a);
            xt = _mm_mullo_epi32(xv, xv);
            xv = _mm_and_si128(xt, xmask);
            xt = _mm_add_epi32(xt, xround);
            xt = _mm_srai_epi32(xt, 14);
            xv = _mm_cmpeq_epi32(xv, xround);
            xv = _mm_sub_epi32(xv, xone);
            xt = _mm_and_si128(xt, xv);

            xv = _mm_load_si128((__m128i*)g);
            xt = _mm_sub_epi32(xbias, xt);
            xt = _mm_mullo_epi32(xt, xv);
            xv = _mm_and_si128(xt, xmask);
            xt = _mm_add_epi32(xt, xround);
            xt = _mm_srai_epi32(xt, 14);
            xv = _mm_cmpeq_epi32(xv, xround);
            xv = _mm_sub_epi32(xv, xone);
            xt = _mm_and_si128(xt, xv);

            _mm_store_si128((__m128i*)z, xt);
            ++a;
            ++g;
            ++z;
        }
#else
        for (int i = 0; i < n; i++) {
            
            int32_t tmp, r;

            tmp = m_a[i] * m_a[i];
            r = tmp & 0x3fff;
            tmp = (tmp + 0x2000) >> 14;
            if (r == 0x2000)
                tmp &= ~1;

            tmp = 0x4000 - tmp;
            tmp = tmp * m_g[i];
            r = tmp & 0x3fff;
            tmp = (tmp + 0x2000) >> 14;
            if (r == 0x2000)
                tmp &= ~1;

            m_z[i] = tmp;
        }
#endif

        fxbp z_bp = get_fxbp(false, 0);
        if (z_bp.m_bp == -1 || !z_bp.m_initialized)
            partial_out(z_fp.data(), m_z, g_bp.m_bp, n);
        else
            partial_out(z_bp, z_fp.data(), m_z, g_bp.m_bp, n);

        free(m_a);
        free(m_g);
        free(m_z);
    }

private:

    static bool m_show_banner;
};

bool WaveTanhGradIntOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveTanhGradInt").Device(DEVICE_CPU), WaveTanhGradIntOp);
