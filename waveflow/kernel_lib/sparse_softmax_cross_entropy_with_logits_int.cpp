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

#include "dyn_fx_pt.h"
#include "dfx_op_base.h"
#include "softmax_lut_dfx.h"

using namespace tensorflow;

REGISTER_OP("WaveSparseSoftmaxCrossEntropyWithLogitsInt")
    .Input("features: float")
    .Input("labels: Tlabels")
    .Output("loss: float")
    .Output("backprop: float")
    .Attr("vp_i0: string = ''")
    .Attr("vp_o0: string = ''")
    .Attr("Tlabels: {int32, int64} = DT_INT64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle features;
        shape_inference::ShapeHandle labels;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &features));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &labels));

        shape_inference::DimensionHandle batch_size;
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(features, 0), c->Dim(labels, 0), &batch_size));
        TF_RETURN_IF_ERROR(c->ReplaceDim(features, 0, batch_size, &features));

        c->set_output(0, c->Vector(batch_size));
        c->set_output(1, features);
        return Status::OK();
    });

template <typename Tlabels>
class WaveSparseSoftmaxCrossEntropyWithLogitsIntOp : public WaveDynFxPointOp {
public:
    explicit WaveSparseSoftmaxCrossEntropyWithLogitsIntOp(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"vp_i0", "vp_o0"})
    {
        if (m_show_banner) {
            printf("WaveSparseSoftmaxCrossEntropyWithLogitsIntOp() init\n");
            m_show_banner = false;
        }
    }

    void Compute(OpKernelContext* context) override
    {
        DCHECK_EQ(1, context->num_inputs());

        const Tensor& tensor_logits = context->input(0);
        const TensorShape& logits_shape = tensor_logits.shape();
        const int logits_last_dim = logits_shape.dims() - 1;
        auto n = logits_shape.dim_size(logits_last_dim);
        auto n_ext = (n + 3) & ~3;

        const Tensor& tensor_labels = context->input(1);
        const TensorShape& labels_shape = tensor_labels.shape();

        TensorShape loss_shape(labels_shape);
        Tensor* tensor_loss = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, loss_shape, &tensor_loss));

        TensorShape grad_shape(logits_shape);
        Tensor* tensor_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, grad_shape, &tensor_grad));

        auto labels = tensor_labels.flat<Tlabels>();
        auto logits_fp = tensor_logits.flat<float>();
        auto loss_fp = tensor_loss->flat<float>();
        auto grad_fp = tensor_grad->flat<float>();
        auto m = logits_fp.size() / n;

        int32_t* m_logits = (int32_t*)aligned_alloc(16, n_ext * sizeof(int32_t));
        int32_t* m_loss = (int32_t*)aligned_alloc(16, m * sizeof(int32_t));
        int32_t* m_grad = (int32_t*)aligned_alloc(16, n_ext * sizeof(int32_t));

        for (int i = n; i < n_ext; i++)
            m_logits[i] = -1000000000;  // padding

        fxbp logits_bp = get_fxbp(true, 0);
        if (logits_bp.m_bp == -1 || !logits_bp.m_initialized)
            logits_bp.set_range_fp(logits_fp.data(), logits_fp.data()+logits_fp.size());

        fxbp grad_bp = get_fxbp(false, 1);

        for (int i = 0; i < m; i++) {
            partial_in(logits_bp, m_logits, logits_fp.data() + i * n, n);

            m_loss[i] = sparse_softmax_cross_entropy(m_logits, m_grad, logits_bp.m_bp, labels(i), n_ext);

            if (grad_bp.m_bp == -1 || !grad_bp.m_initialized)
                partial_out(grad_fp.data() + i * n, m_grad, 14, n);
            else
                partial_out(grad_bp, grad_fp.data() + i * n, m_grad, 14, n);
        }

        fxbp loss_bp = get_fxbp(false, 0);
        if (loss_bp.m_bp == -1 || !loss_bp.m_initialized)
            partial_out(loss_fp.data(), m_loss, 15, m);
        else
            partial_out(loss_bp, loss_fp.data(), m_loss, 15, m);

        free(m_logits);
        free(m_loss);
        free(m_grad);
    }

private:

    static bool m_show_banner;

    int32_t sparse_softmax_cross_entropy(int32_t *logits, int32_t *grad, int32_t src_bp, int64_t label, int n)
    {
        int i;
        int32_t r, s, max;

#if defined(USE_VECTOR_INSTRUCTIONS)
        __m128i xv1, xv2, xk;
        __m128i xmax = _mm_set1_epi32(-32768);
        int32_t *a, *z;

        a = logits;
        for (int i = 0; i < n>>2; i++) {
            xv1 = _mm_load_si128((__m128i*)a);
            xmax = _mm_max_epi32(xmax, xv1);
            a += 4;
        }
        int32_t tmp[4];
        _mm_store_si128((__m128i*)tmp, xmax);
        max = tmp[0];
        if (tmp[1] > max)
            max = tmp[1];
        if (tmp[2] > max)
            max = tmp[2];
        if (tmp[3] > max)
            max = tmp[3];

        __m128i xmask = _mm_set1_epi32(1023);
        __m128i xround = _mm_set1_epi32(512);
        __m128i xone = _mm_set1_epi32(1);
        __m128i xs = _mm_xor_si128(xmask, xmask);
        xmax = _mm_set1_epi32(max);
        int32_t idx;

        if (src_bp > 15) {
            a = logits;
            z = grad;
            for (int i = 0; i < n; i++) {
                idx = max - *a++;
                idx = idx >> (src_bp - 5);
                if (idx > EXP_LUT_SIZE - 1)
                    *z++ = 0;
                else
                    *z++ = exp_lut[idx];
            }

            a = logits;
            z = grad;
            for (int i = 0; i < n>>2; i++) {
                xk  = _mm_load_si128((__m128i*)a);
                xv1 = _mm_load_si128((__m128i*)z);
                xk  = _mm_sub_epi32(xmax, xk);
                xk  = _mm_srai_epi32(xk, src_bp - 15);
                xk  = _mm_and_si128(xk, xmask);
                xv2 = _mm_slli_epi32(xk, 16);
                xk  = _mm_or_si128(xk, xv2);
                xv2 = _mm_madd_epi16(xv1, xk);
                xv1 = _mm_srai_epi32(xv1, 16);
                xk  = _mm_and_si128(xv2, xmask);
                xv2 = _mm_add_epi32(xv2, xround);
                xv2 = _mm_srai_epi32(xv2, 10);
                xk  = _mm_cmpeq_epi32(xk, xround);
                xk  = _mm_sub_epi32(xk, xone);
                xv2 = _mm_and_si128(xv2, xk);
                xv2 = _mm_sub_epi32(xv1, xv2);
                xs  = _mm_add_epi32(xs, xv2);
                _mm_store_si128((__m128i*)z, xv2);
                a += 4;
                z += 4;
            }
        } else if (src_bp > 5) {
            a = logits;
            z = grad;
            for (int i = 0; i < n; i++) {
                idx = max - *a++;
                idx = idx >> (src_bp - 5);
                if (idx > EXP_LUT_SIZE - 1)
                    *z++ = 0;
                else
                    *z++ = exp_lut[idx];
            }
            
            a = logits;
            z = grad;
            for (int i = 0; i < n>>2; i++) {
                xk  = _mm_load_si128((__m128i*)a);
                xv1 = _mm_load_si128((__m128i*)z);
                xk  = _mm_sub_epi32(xmax, xk);
                xk  = _mm_slli_epi32(xk, 15 - src_bp);
                xk  = _mm_and_si128(xk, xmask);
                xv2 = _mm_slli_epi32(xk, 16);
                xk  = _mm_or_si128(xk, xv2);
                xv2 = _mm_madd_epi16(xv1, xk);
                xv1 = _mm_srai_epi32(xv1, 16);
                xk  = _mm_and_si128(xv2, xmask);
                xv2 = _mm_add_epi32(xv2, xround);
                xv2 = _mm_srai_epi32(xv2, 10);
                xk  = _mm_cmpeq_epi32(xk, xround);
                xk  = _mm_sub_epi32(xk, xone);
                xv2 = _mm_and_si128(xv2, xk);
                xv2 = _mm_sub_epi32(xv1, xv2);
                xs  = _mm_add_epi32(xs, xv2);
                _mm_store_si128((__m128i*)z, xv2);
                a += 4;
                z += 4;
            }
        } else {  // src_bp <= 5
            a = logits;
            z = grad;
            for (int i = 0; i < n; i++) {
                idx = max - *a++;
                idx = idx << (5 - src_bp);
                if (idx > EXP_LUT_SIZE - 1)
                    *z++ = 0;
                else
                    *z++ = exp_lut[idx];
            }
            
            a = logits;
            z = grad;
            for (int i = 0; i < n>>2; i++) {
                xk  = _mm_load_si128((__m128i*)a);
                xv1 = _mm_load_si128((__m128i*)z);
                xk  = _mm_sub_epi32(xmax, xk);
                xk  = _mm_slli_epi32(xk, 15 - src_bp);
                xk  = _mm_and_si128(xk, xmask);
                xv2 = _mm_slli_epi32(xk, 16);
                xk  = _mm_or_si128(xk, xv2);
                xv2 = _mm_madd_epi16(xv1, xk);
                xv1 = _mm_srai_epi32(xv1, 16);
                xk  = _mm_and_si128(xv2, xmask);
                xv2 = _mm_add_epi32(xv2, xround);
                xv2 = _mm_srai_epi32(xv2, 10);
                xk  = _mm_cmpeq_epi32(xk, xround);
                xk  = _mm_sub_epi32(xk, xone);
                xv2 = _mm_and_si128(xv2, xk);
                xv2 = _mm_sub_epi32(xv1, xv2);
                xs  = _mm_add_epi32(xs, xv2);
                _mm_store_si128((__m128i*)z, xv2);
                a += 4;
                z += 4;
            }
        }
        _mm_store_si128((__m128i*)tmp, xs);
        s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#else
        max = -32768;
        for (i = 0; i < n; i++)
            if (logits[i] > max)
                max = logits[i];

        s = 0;
        if (src_bp > 15) {
            for (int i = 0; i < n; i++) {
                int32_t idx, r;

                idx = max - logits[i];
                r = (idx >> (src_bp - 15)) & 0x3ff;
                idx = idx >> (src_bp - 5);
                grad[i] = exp(idx, r);
                s = s + grad[i];
            }
        } else if (src_bp > 5) {
            for (int i = 0; i < n; i++) {
                int32_t idx, r;

                idx = max - logits[i];
                r = (idx << (15 - src_bp)) & 0x3ff;
                idx = idx >> (src_bp - 5);
                grad[i] = exp(idx, r);
                s = s + grad[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                int32_t idx, r;

                idx = max - logits[i];
                r = (idx << (15 - src_bp)) & 0x3ff;
                idx = idx << (5 - src_bp);
                grad[i] = exp(idx, r);
                s = s + grad[i];
            }
        }
#endif

        int32_t shift;
        int32_t round;
        int32_t mask;

        int32_t b = __builtin_clz(s) - 1;
        int32_t l = 17 - b;
        l *= 22713;  // ln(2)
        l = l - log(s << b);

        if (src_bp <= 15)
            l += ((max - logits[label]) << (15 - src_bp));
        else {
            int32_t tmp, r;

            shift = src_bp - 15;
            round = 1 << (shift - 1);
            mask = (round << 1) - 1;
            tmp = max - logits[label];
            r = tmp & mask;
            tmp = (tmp + round) >> shift;
            if (r == round)
                tmp &= ~1;
            l += tmp;
        }

        shift = 16 - b;
        round = 1 << (shift - 1);
        mask = (round << 1) - 1;
        if (shift != 0) {
            r = s & mask;
            s = (s + round) >> shift;
            if (r == round)
                s &= ~1;
        }
        s = recip(s);

        shift += 14;
        round = 1 << (shift - 1);
        mask = (round << 1) - 1;
#if defined(USE_VECTOR_INSTRUCTIONS)
        xround = _mm_slli_epi32(xone, shift - 1);
        xmask = _mm_slli_epi32(xround, 1);
        xmask = _mm_sub_epi32(xmask, xone);
        xs = _mm_set1_epi32(s);
        z = grad;
        for (i = 0; i < n>>2; i++) {
            xv1 = _mm_load_si128((__m128i*)z);
            xv1 = _mm_mullo_epi32(xv1, xs);
            xk  = _mm_and_si128(xv1, xmask);
            xv1 = _mm_add_epi32(xv1, xround);
            xv1 = _mm_srai_epi32(xv1, shift);
            xk  = _mm_cmpeq_epi32(xk, xround);
            xk  = _mm_sub_epi32(xk, xone);
            xv1 = _mm_and_si128(xv1, xk);
            _mm_store_si128((__m128i*)z, xv1);
            z += 4;
        }
#else
        for (i = 0; i < n; i++) {
            int32_t tmp;
            tmp = grad[i] * s;
            r = tmp & mask;
            tmp = (tmp + round) >> shift;
            if (r == round)
                tmp &= ~1;
            grad[i] = tmp;
        }
#endif
        grad[label] -= 0x4000;
        return l;
    }

#if !defined(USE_VECTOR_INSTRUCTIONS)
    int32_t exp(int32_t idx, int32_t r)
    {
        int32_t v1, v2;

        if (idx > EXP_LUT_SIZE - 2)
            return 0;
        else {
            v1 = exp_lut[idx];
            v2 = exp_lut[idx+1];
            v2 = v1 - v2;
            v2 = r * v2;
            r = v2 & 0x3ff;
            v2 = (v2 + 0x200) >> 10;
            if (r == 0x200)
                v2 &= ~1;
            v2 = v1 - v2;
            return v2;
        }
    }
#endif

    // calculate fixed-point reciprocal
    // this function assumes that x is normalized
    // i.e. ideal bp is already set by the caller
    int32_t recip(int32_t x)
    {
        int32_t i;
        int32_t res, t, r;

        i = x < 0 ? -x : x;
        i = (i >> 7) - 128;

        res = recip_lut[i];

        t = res * x;
        r = t & 0x7fff;
        t = (t + 0x4000) >> 15;
        if (r == 0x4000)
            t &= ~1;
        t = 0x4000 - t;
        res = res * t;
        r = res & 0x1fff;
        res = (res + 0x1000) >> 13;
        if (r == 0x1000)
            res &= ~1;

        t = res * x;
        r = t & 0x7fff;
        t = (t + 0x4000) >> 15;
        if (r == 0x4000)
            t &= ~1;
        t = 0x4000 - t;
        res = res * t;
        r = res & 0x1fff;
        res = (res + 0x1000) >> 13;
        if (r == 0x1000)
            res &= ~1;

        if (x < 0)
            res = -res;

        return res;
    }

    // calculate fixed-point negative log
    // this function assumes that x is normalized
    // i.e. ideal bp is already set by the caller
    int32_t log(int32_t x)
    {
        int32_t idx, r;
        int32_t v1, v2;

        x >>= 13;
        idx = (x >> 10) - 128;
        r = x & 0x3ff;

        v1 = log_lut[idx];
        v2 = log_lut[idx+1];
        v2 = v1 - v2;
        v2 = r * v2;
        r = v2 & 0x3ff;
        v2 = (v2 + 0x200) >> 10;
        if (r == 0x200)
            v2 &= ~1;
        v2 = v1 - v2;

        return v2;
    }
};

template <typename Tlabels>
bool WaveSparseSoftmaxCrossEntropyWithLogitsIntOp<Tlabels>::m_show_banner = true;

REGISTER_KERNEL_BUILDER(
    Name("WaveSparseSoftmaxCrossEntropyWithLogitsInt")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("Tlabels"),
    WaveSparseSoftmaxCrossEntropyWithLogitsIntOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("WaveSparseSoftmaxCrossEntropyWithLogitsInt")
    .Device(DEVICE_CPU).TypeConstraint<int64>("Tlabels"),
    WaveSparseSoftmaxCrossEntropyWithLogitsIntOp<int64>);
