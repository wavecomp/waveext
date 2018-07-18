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


using namespace tensorflow;

REGISTER_OP("WaveVecAddGradDfx")
    .Input("a: float")
    .Output("z: float")
    // Dynamic Fixed Point linkage
    .Attr("bp_i0: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::BiasAddGradShape);
 
class WaveVecAddGradDfxOp : public WaveDynFxPointOp {
public:
    
    explicit WaveVecAddGradDfxOp(OpKernelConstruction* context) 
        : WaveDynFxPointOp(context, {"bp_i0", "bp_o0"}), m_mm_a_dfx(), m_mm_z_dfx() {
        if (m_show_banner) {
            printf("WaveVecAddGradDfxOp() init\n");
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
        
        auto a_m = tensor_a.flat<float>();
        auto z_m = output->flat<float>();

        m_mm_a_dfx.resize(a_m.size());
        m_mm_z_dfx.resize(z_m.size());
        
        fxbp bp_in = get_fxbp(true, 0);
        partial_in(bp_in, m_mm_a_dfx, a_m.data());
        
        fxbp    z_fxbp(16, 32);
        for (int j = 0; j < z_m.size(); j++) {
            m_mm_z_dfx[j].set_fxbp(z_fxbp);
            m_mm_z_dfx[j] = 0;
        }
        for (int i = 0; i < a_m.size(); i++) {
            int j = i % z_m.size();
            m_mm_z_dfx[j] += m_mm_a_dfx[i];
        }
        
        fxbp bp_out = get_fxbp(false, 0);
        partial_out(bp_out, z_m.data(), m_mm_z_dfx);
    }
private:
    DFXVector m_mm_a_dfx;
    DFXVector m_mm_z_dfx;
    
    static bool m_show_banner;

};

bool WaveVecAddGradDfxOp::m_show_banner = true;
 
REGISTER_KERNEL_BUILDER(Name("WaveVecAddGradDfx").Device(DEVICE_CPU), WaveVecAddGradDfxOp);