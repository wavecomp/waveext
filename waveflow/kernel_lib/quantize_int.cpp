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
#include "tensorflow/core/framework/shape_inference.h"

#include <stdlib.h>

#include "dyn_fx_pt.h"
#include "dfx_op_base.h"
#include "dfx_registry.h"


// #define DFX_GEN_DEBUG 1

using namespace tensorflow;

REGISTER_OP("WaveQuantizeInt")
    .Input("a: float")
    .Input("bp: int32")
    .Output("z: float")
    .Attr("use_stochastic_round: bool = false")
    // Dynamic Fixed Point linkage
    .Attr("bp_i0: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

/* This class represents a simple Dynamic Fixed Point generator op. This base
 * implementation simply sweeps through the input tensor and computes the ideal
 * BP. Other varieties can be implemented, including simple constant values.
 *
 * In all cases, the user needs to bind a unique key to the Output-1 tensor
 * via an attribute. This key will be used by consumers to lookup the BP.
 * */
class WaveQuantizeInt : public WaveDynFxPointOp {
public:
    explicit WaveQuantizeInt(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"bp_i0", "bp_o0"})
    {
        if (m_show_banner) {
            printf("WaveQuantizeInt() init\n");
            m_show_banner = false;
        }
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_stochastic_round", &m_use_stochastic_round));
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& a_input = context->input(0);
        const TensorShape& a_shape = a_input.shape();

        TensorShape out_shape(a_shape);
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        // Ensure the B tensor input is a DFX value.
        const Tensor& bp_input = context->input(1);
        const TensorShape& bp_shape = bp_input.shape();
        DCHECK_EQ(bp_shape.dims(), 1);
        DCHECK_EQ(bp_shape.dim_size(0), 2);
        DCHECK(DataTypeIsInteger(bp_input.dtype()));

        auto a_fp = a_input.flat<float>();
        auto n = a_fp.size();

        int32_t* m_a = (int32_t*)aligned_alloc(16, n * sizeof(int32_t));

        fxbp a_bp = get_fxbp(true, 0);
        partial_in(a_bp, m_a, a_fp.data(), n);

        // Pull the BP data from the B tensor.
        // Tensor: [WL, BP]  - Constructor: (BP, WL)
        const auto bp_m = bp_input.flat<int32>();
        fxbp out_bp(bp_m(1), bp_m(0));
        partial_out(out_bp, output->flat<float>().data(), m_a, a_bp.m_bp, n);

        free(m_a);
#ifdef DFX_GEN_DEBUG
        const std::string dfx_key = name();
        printf("Gen: key: %s, dfx: (%d,%d)\n", dfx_key.c_str(), m_bp.m_wl, m_bp.m_bp);
#endif
    }

private:

    static bool m_show_banner;
};


bool WaveQuantizeInt::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveQuantizeInt").Device(DEVICE_CPU), WaveQuantizeInt);
