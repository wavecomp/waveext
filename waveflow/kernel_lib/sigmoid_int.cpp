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
#endif

#include "dfx_op_base.h"
#include "sigmoid_lut_dfx.h"

#include <sys/time.h>

using namespace tensorflow;

REGISTER_OP("WaveSigmoidInt")
    .Input("a: float")
    .Output("z: float")
    .Attr("bp_i0: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

class WaveSigmoidIntOp : public WaveDynFxPointOp {
public:
    explicit WaveSigmoidIntOp(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"bp_i0", "bp_o0"})
    {
        if (m_show_banner) {
            printf("WaveSigmoidIntOp() init\n");
            m_show_banner = false;
        }
    }

    void Compute(OpKernelContext* context) override
    {
        DCHECK_EQ(1, context->num_inputs());

        const Tensor& tensor_a = context->input(0);

        TensorShape out_shape(tensor_a.shape());
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        auto a_fp = tensor_a.flat<float>();
        auto z_fp = output->flat<float>();
        auto n = a_fp.size();
        auto n_ext = (n + 3) & ~3;

        int32_t* m_a = (int32_t*)aligned_alloc(16, n_ext * sizeof(int32_t));
        int32_t* m_z = (int32_t*)aligned_alloc(16, n_ext * sizeof(int32_t));

        fxbp a_bp = get_fxbp(true, 0);
        partial_in(a_bp, m_a, a_fp.data(), n);

#if defined(USE_VECTOR_INSTRUCTIONS)
        __m128i xmask = _mm_set1_epi32(1023);
        __m128i xround = _mm_set1_epi32(512);
        __m128i xone = _mm_set1_epi32(1);
        __m128i xbias = _mm_set1_epi32(16384);
        __m128i xv1, xv2, xk, xt, xa;
        int32_t idx;
        int32_t *a, *z;

        if (a_bp.m_bp > 14) {
            a = m_a;
            z = m_z;
            for (int i = 0; i < n_ext; i++) {
                idx = *a++;
                idx = idx < 0 ? -idx : idx;
                idx = idx >> (a_bp.m_bp - 4);
                if (idx > SIGMOID_LUT_SIZE - 1)
                    *z++ = 0x4000c000;
                else
                    *z++ = sigmoid_lut[idx];
            }
            
            a = m_a;
            z = m_z;
            for (int i = 0; i < n_ext>>2; i++) {
                xa  = _mm_load_si128((__m128i*)a);
                xv1 = _mm_load_si128((__m128i*)z);
                xk  = _mm_abs_epi32(xa);
                xk  = _mm_srai_epi32(xk, a_bp.m_bp - 14);
                xk  = _mm_and_si128(xk, xmask);
                xt  = _mm_slli_epi32(xk, 16);
                xk  = _mm_or_si128(xk, xt);
                xv2 = _mm_madd_epi16(xv1, xk);
                xv1 = _mm_srai_epi32(xv1, 16);
                xk  = _mm_and_si128(xv2, xmask);
                xv2 = _mm_add_epi32(xv2, xround);
                xv2 = _mm_srai_epi32(xv2, 10);
                xk  = _mm_cmpeq_epi32(xk, xround);
                xk  = _mm_sub_epi32(xk, xone);
                xv2 = _mm_and_si128(xv2, xk);
                xa  = _mm_srai_epi32(xa, 31);
                xk  = _mm_and_si128(xbias, xa);
                xv2 = _mm_sub_epi32(xv1, xv2);
                xv2 = _mm_xor_si128(xv2, xa);
                xv2 = _mm_sub_epi32(xv2, xa);
                xv2 = _mm_add_epi32(xv2, xk);
                _mm_store_si128((__m128i*)z, xv2);
                a += 4;
                z += 4;
            }
        } else if (a_bp.m_bp > 4) {
          a = m_a;
          z = m_z;
          for (int i = 0; i < n_ext; i++) {
            idx = *a++;
            idx = idx < 0 ? -idx : idx;
            idx = idx >> (a_bp.m_bp - 4);
            if (idx > SIGMOID_LUT_SIZE - 1)
              *z++ = 0x4000c000;
            else
              *z++ = sigmoid_lut[idx];
          }
        
          a = m_a;
          z = m_z;
          for (int i = 0; i < n_ext>>2; i++) {
            xa  = _mm_load_si128((__m128i*)a);
            xv1 = _mm_load_si128((__m128i*)z);
            xk  = _mm_abs_epi32(xa);
            xk  = _mm_slli_epi32(xk, 14 - a_bp.m_bp);
            xk  = _mm_and_si128(xk, xmask);
            xt  = _mm_slli_epi32(xk, 16);
            xk  = _mm_or_si128(xk, xt);
            xv2 = _mm_madd_epi16(xv1, xk);
            xv1 = _mm_srai_epi32(xv1, 16);
            xk  = _mm_and_si128(xv2, xmask);
            xv2 = _mm_add_epi32(xv2, xround);
            xv2 = _mm_srai_epi32(xv2, 10);
            xk  = _mm_cmpeq_epi32(xk, xround);
            xk  = _mm_sub_epi32(xk, xone);
            xv2 = _mm_and_si128(xv2, xk);
            xa  = _mm_srai_epi32(xa, 31);
            xk  = _mm_and_si128(xbias, xa);
            xv2 = _mm_sub_epi32(xv1, xv2);
            xv2 = _mm_xor_si128(xv2, xa);
            xv2 = _mm_sub_epi32(xv2, xa);
            xv2 = _mm_add_epi32(xv2, xk);
            _mm_store_si128((__m128i*)z, xv2);
            a += 4;
            z += 4;
          }
        } else {  // a_bp.m_bp <= 4
          a = m_a;
          z = m_z;
          for (int i = 0; i < n_ext; i++) {
            idx = *a++;
            idx = idx < 0 ? -idx : idx;
            idx = idx << (4 - a_bp.m_bp);
            if (idx > SIGMOID_LUT_SIZE - 1)
              *z++ = 0x4000c000;
            else
              *z++ = sigmoid_lut[idx];
          }
        
          a = m_a;
          z = m_z;
          for (int i = 0; i < n_ext>>2; i++) {
            xa  = _mm_load_si128((__m128i*)a);
            xv1 = _mm_load_si128((__m128i*)z);
            xk  = _mm_abs_epi32(xa);
            xk  = _mm_slli_epi32(xk, 14 - a_bp.m_bp);
            xk  = _mm_and_si128(xk, xmask);
            xt  = _mm_slli_epi32(xk, 16);
            xk  = _mm_or_si128(xk, xt);
            xv2 = _mm_madd_epi16(xv1, xk);
            xv1 = _mm_srai_epi32(xv1, 16);
            xk  = _mm_and_si128(xv2, xmask);
            xv2 = _mm_add_epi32(xv2, xround);
            xv2 = _mm_srai_epi32(xv2, 10);
            xk  = _mm_cmpeq_epi32(xk, xround);
            xk  = _mm_sub_epi32(xk, xone);
            xv2 = _mm_and_si128(xv2, xk);
            xa  = _mm_srai_epi32(xa, 31);
            xk  = _mm_and_si128(xbias, xa);
            xv2 = _mm_sub_epi32(xv1, xv2);
            xv2 = _mm_xor_si128(xv2, xa);
            xv2 = _mm_sub_epi32(xv2, xa);
            xv2 = _mm_add_epi32(xv2, xk);
            _mm_store_si128((__m128i*)z, xv2);
            a += 4;
            z += 4;
          }
        }
#else
        if (a_bp.m_bp > 14) {
            for (int i = 0; i < n; i++) {
                int32_t idx, r;

                idx = m_a[i];
                idx = idx < 0 ? -idx : idx;

                r = (idx >> (a_bp.m_bp - 14)) & 0x3ff;
                idx = idx >> (a_bp.m_bp - 4);

                m_z[i] = sigmoid(idx, r);
                if (m_a[i] < 0)
                    m_z[i] = 16384 - m_z[i];
            }
        } else if (a_bp.m_bp > 4) {
            for (int i = 0; i < n; i++) {
                int32_t idx, r;

                idx = m_a[i];
                idx = idx < 0 ? -idx : idx;

                r = (idx << (14 - a_bp.m_bp)) & 0x3ff;
                idx = idx >> (a_bp.m_bp - 4);

                m_z[i] = sigmoid(idx, r);
                if (m_a[i] < 0)
                    m_z[i] = 16384 - m_z[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                int32_t idx, r;

                idx = m_a[i];
                idx = idx < 0 ? -idx : idx;

                r = (idx << (14 - a_bp.m_bp)) & 0x3ff;
                idx = idx << (4 - a_bp.m_bp);

                m_z[i] = sigmoid(idx, r);
                if (m_a[i] < 0)
                    m_z[i] = 16384 - m_z[i];
            }
        }
#endif

        fxbp z_bp = get_fxbp(false, 0);
        if (z_bp.m_bp == -1 || !z_bp.m_initialized)
            partial_out(z_fp.data(), m_z, 14, n);
        else
            partial_out(z_bp, z_fp.data(), m_z, 14, n);

        free(m_a);
        free(m_z);
    }

private:

    static bool m_show_banner;

    int32_t sigmoid(int32_t idx, int32_t r)
    {
        int32_t v1, v2;

        if (idx > SIGMOID_LUT_SIZE - 2)
            return 16384;
        else {
            v1 = sigmoid_lut[idx];
            v2 = sigmoid_lut[idx+1];
            v2 = v1 - v2;
            v2 = r * v2;
            r = v2 & 1023;
            v2 = (v2 + 512) >> 10;
            if (r == 512)
              v2 &= ~1;
            v2 = v1 - v2;
            return v2;
        }
    }
};

bool WaveSigmoidIntOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveSigmoidInt").Device(DEVICE_CPU), WaveSigmoidIntOp);
