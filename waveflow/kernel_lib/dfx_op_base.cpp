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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/node_def.pb.h"

#include <stdlib.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "dfx_op_base.h"
#include "dfx_registry.h"

#define USE_VECTOR_INSTRUCTIONS

#if defined(USE_VECTOR_INSTRUCTIONS)
#include <emmintrin.h>
#include <tmmintrin.h>
#endif

using namespace tensorflow;

class RandomGenerator
{
public:
    static RandomGenerator& Instance() {
        static RandomGenerator s;
        return s;
    }

    std::mt19937 & get() {
        return mt;
    }

private:
    RandomGenerator() {
        std::random_device rd;
        mt.seed(rd());
    }
    ~RandomGenerator() {}

    std::mt19937 mt;
};


// Constructor will read all provided attributes and bind them to the I/O
// indexes. Note that these attributes must exist in the OpDef or else this 
// code asserts (intentional).
WaveDynFxPointOp::WaveDynFxPointOp(OpKernelConstruction* ctx, 
                                   const StringVec& attrs) 
: OpKernel(ctx), m_iLookup(ctx->num_inputs()), m_oLookup(ctx->num_outputs()), m_use_stochastic_round(false)
{
    for (auto a : attrs) {
        bool isInput;
        int index;
        if (!get_iospec(a, isInput, index)) {
            continue;
        }
        string attrVal;
        OP_REQUIRES_OK(ctx, ctx->GetAttr(a, &attrVal));
        if (isInput) {
            assert(index < ctx->num_inputs());
            m_iLookup[index] = attrVal;
        } else {
            assert(index < ctx->num_outputs());
            m_oLookup[index] = attrVal;
        }
    }
}


string WaveDynFxPointOp::get_vp_key(bool isInput, int n) const
{
    std::string lName;
    if (isInput) {
        assert(n < m_iLookup.size());
        lName = m_iLookup[n];
    } else {
        assert(n < m_oLookup.size());
        lName = m_oLookup[n];
    }
    return lName;
}


// Gets the Dynamic Fixed Point for a given I/O. This function finds the key associated
// with this I/O, then looks up the BP associated with the key. If no generator
// is found, it will return an uninitialized BP, which the caller should handle
// appropriately. This ususally means the caller is free to assign an ideal BP.
fxbp WaveDynFxPointOp::get_fxbp(bool isInput, int n) const
{
    std::string lName = get_vp_key(isInput, n);
    // If the string is empty, nobody has registered this BP. If so, return
    // an uninitialized fxbp.
    if (lName.size() == 0) {
        return fxbp();
    }
    // Test to see if the BP is static.
    fxbp bp;
    if (get_static_bp(lName, bp)) {
        assert(bp.m_initialized);
        return bp;
    }
    // See if there is a registered BP assignment.
    if (get_dynamic_bp(lName, bp)) {
        assert(bp.m_initialized);
        return bp;
    }
    // If nothing matches, return the uninitialized BP.
    assert(!bp.m_initialized);
    return bp;
}

// Helper function to map attribute names to I/O.
bool WaveDynFxPointOp::get_iospec(const std::string& attr, bool& isInput, int& n)
{
    assert(attr.size() > 0);
    // Parse the attr
    auto a0 = boost::algorithm::find_first(attr, "bp_");
    if (a0.begin() == attr.end()) {
        return false;
    }
    const char ioType = *(a0.end());
    isInput = (ioType == 'i');
    std::string::const_iterator a1 = a0.end() + 1;
    
    // We will use the IO index without checking the actual index limit.
    n = boost::lexical_cast<int>(*a1);
    return true;
}

// This helper function parses an attribute. If the attribute encodes a
// static binary point, it will return it in an fxbp object.
bool WaveDynFxPointOp::get_static_bp(const string& attr, fxbp& bp) const
{
    // This code assumes encoded attributes are well-formed. 
    // TODO: add warning/error.
    if (attr.front() != '(' || attr.back() != ')') {
        return false;
    }
    std::vector<std::string> params;
    boost::split(params, attr, boost::is_any_of("(,)"));
    if (params.size() != 4) {
        return false;
    }
    bp = fxbp(boost::lexical_cast<int>(params[2]), 
              boost::lexical_cast<int>(params[1]));
    return true;
}

// This helper function parses an attribute. If the attribute encodes a
// reference to a dynamic binary point, it will return it in an fxbp object.
bool WaveDynFxPointOp::get_dynamic_bp(const string& attr, fxbp& bp) const
{
    // Consult the registry to get Tensor ref from name.
    const DynFxPointRegistry* reg = DynFxPointRegistry::get_registry();
    const Tensor* t = reg->find_dfx(attr);
    if (!t) {
        return false;
    }
    assert(t);
    assert(t->dims() == 1);
    assert(t->dim_size(0) == 2);
    assert(DataTypeIsInteger(t->dtype()));
    // Copy the tensor data into an fxbp object.
    const auto a = t->flat<int32>();
    // Tensor: [WL, BP]  - Constructor: (BP, WL)
    bp = fxbp(a(1), a(0));
    return true;
}

void WaveDynFxPointOp::partial_in(fxbp& dest_bp, WaveDynFxPointOp::DFXVector& m_out, 
                                  const float* flat_arr)
{
    if (dest_bp.m_bp == -1 || !dest_bp.m_initialized) {
        dest_bp.set_range_fp(flat_arr, flat_arr + m_out.size());
    }
    for (int i = 0; i < m_out.size(); i++) {
        m_out[i].set_fxbp(dest_bp);
        m_out[i] = flat_arr[i];
    }
}

void WaveDynFxPointOp::partial_out(fxbp& out_bp, float* conv_out, 
                                   const WaveDynFxPointOp::DFXVector& m_out)
{
    if (out_bp.m_bp == -1 || !out_bp.m_initialized) {
        out_bp.set_range_dfx(m_out.data(), m_out.data()+m_out.size());
    }
    DynFxPoint v(out_bp);
    if (m_use_stochastic_round) { // stochastic rounding
        for (int i = 0; i < m_out.size(); i++) {
            v.set_bits(stochastic_rounding(m_out[i].get_value(), out_bp.m_bp - m_out[i].get_fxbp().m_bp));
            conv_out[i] = v.to_fp();
        }
    } else {  // convergent rounding
        for (int i = 0; i < m_out.size(); i++) {
            v = m_out[i];
            conv_out[i] = v.to_fp();
        }
    }
}

void WaveDynFxPointOp::partial_in(fxbp& dest_bp, int32_t* m_out, const float* flat_arr, int n)
{
    if (dest_bp.m_bp == -1 || !dest_bp.m_initialized) {
        dest_bp.set_range_fp(flat_arr, flat_arr + n);
    }
    float k = (float)(1 << dest_bp.m_bp);
#if defined(USE_VECTOR_INSTRUCTIONS)
    __m128 xk = _mm_set1_ps(k);

    for (int i = 0; i < n>>2; i++) {
        __m128i xv;
        __m128 xf;

        xf = _mm_loadu_ps(flat_arr);
        xf = _mm_mul_ps(xf, xk);
        xf = _mm_round_ps(xf, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        xv = _mm_cvtps_epi32(xf);
        _mm_store_si128((__m128i *)m_out, xv);
        flat_arr += 4;
        m_out += 4;
    }

    for (int i = 0; i < (n&3); i++)
        *m_out++ = (int32_t)roundf(*flat_arr++ * k);
#else
    for (int i = 0; i < n; i++)
        *m_out++ = (int32_t)roundf(*flat_arr++ * k);
#endif
}

void WaveDynFxPointOp::partial_out(fxbp& out_bp, float* conv_out, const int32_t* m_out, int32_t src_bp, int n)
{
    if (out_bp.m_bp == -1 || !out_bp.m_initialized) {
        out_bp.set_range_int32(m_out, m_out + n, src_bp);
    }
    float k = 1.0f / (1 << out_bp.m_bp);
    if (out_bp.m_bp >= src_bp) {
        int32_t shift = out_bp.m_bp - src_bp;
#if defined(USE_VECTOR_INSTRUCTIONS)
        __m128 xk = _mm_set1_ps(k);

        for (int i = 0; i < n>>2; i++) {
            __m128i xv;
            __m128 xf;

            xv = _mm_load_si128((__m128i*)m_out);
            xv = _mm_slli_epi32(xv, shift);
            xf = _mm_cvtepi32_ps(xv);
            xf = _mm_mul_ps(xf, xk);
            _mm_storeu_ps(conv_out, xf);
            m_out += 4;
            conv_out += 4;
        }

        for (int i = 0; i < (n&3); i++) {
            int32_t v;
            v = *m_out++;
            // convert bp
            v <<= shift;
            *conv_out++ = v * k;
        }
#else
        for (int i = 0; i < n; i++) {
            int32_t v;
            v = *m_out++;
            // convert bp
            v <<= shift;
            *conv_out++ = v * k;
        }
#endif
    } else {
        int32_t r;
        int32_t shift = src_bp - out_bp.m_bp;

        if (m_use_stochastic_round) { // stochastic rounding
            int32_t round;
#if defined(USE_VECTOR_INSTRUCTIONS)
            std::mt19937 &mt = RandomGenerator::Instance().get();
            std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
            uint32_t rnd[4];
            rnd[0] = dist(mt);  // random seed
            rnd[1] = dist(mt);
            rnd[2] = dist(mt);
            rnd[3] = dist(mt);
            __m128i xrnd = _mm_load_si128((__m128i*)rnd);
            __m128i xc1 = _mm_set1_epi32(1664525);
            __m128i xc2 = _mm_set1_epi32(1013904223);
            __m128 xk = _mm_set1_ps(k);

            for (int i = 0; i < n>>2; i++) {
                __m128i xv, xr;
                __m128 xf;

                xv = _mm_load_si128((__m128i*)m_out);
                xr = _mm_srli_epi32(xrnd, 32 - shift);
                xv = _mm_add_epi32(xv, xr);
                xv = _mm_srai_epi32(xv, shift);
                xf = _mm_cvtepi32_ps(xv);
                xrnd = _mm_mullo_epi32(xrnd, xc1);
                xrnd = _mm_add_epi32(xrnd, xc2);
                xf = _mm_mul_ps(xf, xk);
                _mm_storeu_ps(conv_out, xf);
                m_out += 4;
                conv_out += 4;
            }

            _mm_store_si128((__m128i*)rnd, xrnd);
            for (int i = 0; i < (n&3); i++) {
                int32_t v;
                v = *m_out++;
                // convert bp
                round = rnd[0] >> (32 - shift);
                v = (v + round) >> shift;
                // linear congruential RNG from
                // numerical recipes
                rnd[0] = rnd[0] * 1664525 + 1013904223;
                *conv_out++ = v * k;
            }
#else
            std::mt19937 &mt = RandomGenerator::Instance().get();
            std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
            uint32_t rnd = dist(mt);  // random seed

            for (int i = 0; i < n; i++) {
                int32_t v;
                v = m_out[i];
                // convert bp
                round = rnd >> (32 - shift);
                v = (v + round) >> shift;
                rnd = rnd * 1664525 + 1013904223;
                conv_out[i] = v * k;
            }
#endif
        } else {  // convergent rounding
            int32_t round = 1 << (shift - 1);
            int32_t mask = (round << 1) - 1;
#if defined(USE_VECTOR_INSTRUCTIONS)
            __m128i xmask = _mm_set1_epi32(mask);
            __m128i xround = _mm_set1_epi32(round);
            __m128i xone = _mm_set1_epi32(1);
            __m128 xk = _mm_set1_ps(k);

            for (int i = 0; i < n>>2; i++) {
                __m128i xv, xr;
                __m128 xf;

                xv = _mm_load_si128((__m128i*)m_out);
                xr = _mm_and_si128(xv, xmask);
                xv = _mm_add_epi32(xv, xround);
                xv = _mm_srai_epi32(xv, shift);
                xr = _mm_cmpeq_epi32(xr, xround);
                xr = _mm_sub_epi32(xr, xone);
                xv = _mm_and_si128(xv, xr);
                xf = _mm_cvtepi32_ps(xv);
                xf = _mm_mul_ps(xf, xk);
                _mm_storeu_ps(conv_out, xf);
                m_out += 4;
                conv_out += 4;
            }

            for (int i = 0; i < (n&3); i++) {
                int32_t v;
                v = *m_out++;
                // convert bp
                r = v & mask;
                v = (v + round) >> shift;
                if (r == round)
                    v &= ~1;
                *conv_out++ = v * k;
            }
#else
            for (int i = 0; i < n; i++) {
                int32_t v;
                v = *m_out++;
                // convert bp
                r = v & mask;
                v = (v + round) >> shift;
                if (r == round)
                    v &= ~1;
                *conv_out++ = v * k;
            }
#endif
        }
    }
}

void WaveDynFxPointOp::partial_out(float* conv_out, const int32_t* m_out, int src_bp, int n)
{
    float k = 1.0f / (1 << src_bp);
#if defined(USE_VECTOR_INSTRUCTIONS)
    __m128 xk = _mm_set1_ps(k);

    for (int i = 0; i < n>>2; i++) {
        __m128i xv;
        __m128 xf;

        xv = _mm_load_si128((__m128i*)m_out);
        xf = _mm_cvtepi32_ps(xv);
        xf = _mm_mul_ps(xf, xk);
        _mm_storeu_ps(conv_out, xf);
        m_out += 4;
        conv_out += 4;
    }

    for (int i = 0; i < (n&3); i++)
        *conv_out++ = *m_out++ * k;
#else
    for (int i = 0; i < n; i++)
        *conv_out++ = *m_out++ * k;
#endif
}

// Computes the assigned rounding for value v with rounded bits r.
int32_t WaveDynFxPointOp::stochastic_rounding(int32_t v, int32_t r)
{
    int64_t nv = v;
    // Positive diffs shift left.
    if (r >= 0) {
        nv = nv << r;
    } else {
        // This will be compiled as an arithmetic shift (due to sign).
        nv = nv >> (-r);        

        int32_t round_mask = (1 << (-r)) - 1;
        int32_t rv = v & round_mask;

        std::mt19937 &mt = RandomGenerator::Instance().get();
        std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
        uint32_t rand_num = (dist(mt) >> (32+r)) & round_mask;

        nv += rand_num < rv ? 1 : 0;
    }

    // Saturation check
 //   return sat_eval(nv);

    return nv;
}
