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

#include <stdlib.h>

#include "dyn_fx_pt.h"
#include "dfx_op_base.h"


using namespace tensorflow;

REGISTER_OP("WaveVecAddDfx")
    .Input("a: float")
    .Input("b: float")
    .Output("z: float")
    .Attr("data_format: string = 'NHWC'")
    // Dynamic Fixed Point linkage
    .Attr("bp_i0: string = ''")
    .Attr("bp_i1: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);
 
class WaveVecAddDfxOp : public WaveDynFxPointOp {
public:
    explicit WaveVecAddDfxOp(OpKernelConstruction* ctx) 
        : WaveDynFxPointOp(ctx, {"bp_i0", "bp_i1", "bp_o0"}),
        m_mm_a_dfx(), m_mm_b_dfx(), m_mm_z_dfx() {
        // KS: for right now, we will handle only NHWC. TF seems only to have
        // this implemented internally, so it should be the only thing we see.
        string data_format_str;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
        OP_REQUIRES(ctx, (data_format_str == "NHWC"),
            errors::InvalidArgument(
            "wave_vec_add requires the data_format attribute to be NHWC"));        
        if (m_show_banner) {
            printf("WaveVecAddDfxOp() init\n");
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
        
        // KS: This version only handles broadcasting on the last dimension
        // of Tensor A.
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
        
        auto a_m = tensor_a.flat<float>();
        auto b_m = tensor_b.flat<float>();
        auto z_m = output->flat<float>();
        
        m_mm_a_dfx.resize(a_m.size());
        m_mm_b_dfx.resize(b_m.size());
        m_mm_z_dfx.resize(z_m.size());
        
        fxbp a_bp = get_fxbp(true, 0);
        fxbp b_bp = get_fxbp(true, 1);
        partial_in(a_bp, m_mm_a_dfx, a_m.data());
        partial_in(b_bp, m_mm_b_dfx, b_m.data());
        
        for (int i = 0; i < a_m.size(); i++) {
            int j = i % b_shape.dim_size(0);
            m_mm_z_dfx[i] = m_mm_a_dfx[i] + m_mm_b_dfx[j];
        }

        fxbp z_bp = get_fxbp(false, 0);
        partial_out(z_bp, z_m.data(), m_mm_z_dfx);
    }
    
private:
    DFXVector m_mm_a_dfx;
    DFXVector m_mm_b_dfx;
    DFXVector m_mm_z_dfx;
    
    static bool m_show_banner;
};

bool WaveVecAddDfxOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveVecAddDfx").Device(DEVICE_CPU), WaveVecAddDfxOp);