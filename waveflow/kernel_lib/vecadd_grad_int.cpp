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

#include "dyn_fx_pt.h"
#include "dfx_op_base.h"

#define USE_VECTOR_INSTRUCTIONS

#ifdef USE_VECTOR_INSTRUCTIONS
    #include <emmintrin.h>
#endif
#define VECTOR_SIZE 4   //SSE2 vect size

using namespace tensorflow;

REGISTER_OP("WaveVecAddGradInt")
    .Input("a: float")
    .Output("z: float")
    // Dynamic Fixed Point linkage
    .Attr("bp_i0: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::BiasAddGradShape);

class WaveVecAddGradIntOp : public WaveDynFxPointOp {
public:

    explicit WaveVecAddGradIntOp(OpKernelConstruction* context)
        : WaveDynFxPointOp(context, {"bp_i0", "bp_o0"}) {
        if (m_show_banner) {
            printf("WaveVecAddGradIntOp() init\n");
            m_show_banner = false;
        }
    }

    void Compute(OpKernelContext* context) override
    {
        DCHECK_EQ(1, context->num_inputs());

        const Tensor& tensor_a = context->input(0);

        const TensorShape& a_shape = tensor_a.shape();
        const int last_dim_a = a_shape.dims() - 1;

        // Output gradient is 1-dimensional. For bias gradients, we use the
        // last Tensor dimension as the vector size. Tensors are always assumed
        // NHWC.
        TensorShape out_shape({a_shape.dim_size(last_dim_a)});

        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        auto a_fp = tensor_a.flat<float>();
        auto z_fp = output->flat<float>();
        auto a_size = a_fp.size();
        auto z_size = z_fp.size();

        int32_t* m_a = (int32_t*)aligned_alloc(16, a_size * sizeof(int32_t) + (VECTOR_SIZE-1) * 4);
        int32_t* m_z = (int32_t*)aligned_alloc(16, z_size * sizeof(int32_t) + (VECTOR_SIZE-1) * 4);

        fxbp a_bp = get_fxbp(true, 0);
        partial_in(a_bp, m_a, a_fp.data(), a_size);

        memset(m_z, 0, z_size);

        // divide and round up
        int vec_num = (z_size + (VECTOR_SIZE - 1)) / VECTOR_SIZE;
        int iters = (a_size + (z_size - 1)) / z_size;

#ifdef USE_VECTOR_INSTRUCTIONS
         __m128i vsum;

        for (int i = 0; i < vec_num; i++) {
            __m128i vsum = _mm_set1_epi32(0);
            for (int j = 0; j < iters; j++) {
                __m128i v = _mm_loadu_si128((__m128i const*) &m_a[i*VECTOR_SIZE + j*z_size]);  // load vector of 4 x 32 bit values
                vsum = _mm_add_epi32(vsum, v);      // accumulate to 32 bit partial sum vector
            }
           _mm_store_si128((__m128i *) &m_z[i * VECTOR_SIZE], vsum);
        }
#else
        int vsum[VECTOR_SIZE];

        for (int i = 0; i < vec_num; i++) {
            memset(vsum, 0, sizeof(vsum));
            for (int j = 0; j < iters; j++) {
                for (int k = 0; k < VECTOR_SIZE; k++) {
                    vsum[k] += m_a[i*VECTOR_SIZE + j*z_size + k];
                }
            }
            memcpy(m_z + i * VECTOR_SIZE, vsum, sizeof(vsum));
        }
#endif

        fxbp bp_out = get_fxbp(false, 0);
        partial_out(bp_out, z_fp.data(), m_z, a_bp.m_bp, z_size);

        free(m_a);
        free(m_z);
    }
private:

    static bool m_show_banner;
};

bool WaveVecAddGradIntOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveVecAddGradInt").Device(DEVICE_CPU), WaveVecAddGradIntOp);
