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

REGISTER_OP("WaveBpGen")
    .Input("tensor_in: float")
    .Output("bp_out: int32")
    // Dynamic Fixed Point linkage
    .Attr("bp_o0: string = ''")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // c->set_output(0, c->input(0));
      return Status::OK();
    });

/* This class represents a simple Dynamic Fixed Point generator op. This base 
 * implementation simply sweeps through the input tensor and computes the ideal 
 * BP. Other varieties can be implemented, including simple constant values.
 * 
 * In all cases, the user needs to bind a unique key to the Output-1 tensor
 * via an attribute. This key will be used by consumers to lookup the BP.
 * */
class WaveBpGenOp : public WaveDynFxPointOp {
public:
    
    explicit WaveBpGenOp(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"bp_o0"}), m_bp(get_fxbp(false, 0))
    {
        // If the BP isn't specified, default to 16b auto.
        if (!m_bp.m_initialized) {
            m_bp.initialize(-1, 16);
        }
        if (m_show_banner) {
            printf("WaveBpGenOp() init\n");
            m_show_banner = false;
        }
        auto reg = DynFxPointRegistry::get_registry();
        // BP output setup
        TensorShape dfx_shape({2});
        OP_REQUIRES_OK(ctx, 
                       ctx->allocate_persistent(DT_INT32, dfx_shape, &m_vp_tensor, NULL));
        std::string dfx_key = name();
        assert(dfx_key.size() > 0);
        Tensor* t_vp_internal = m_vp_tensor.AccessTensor(ctx);
        assert(t_vp_internal);
        reg->register_dfx(dfx_key, t_vp_internal);
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& t_input = context->input(0);    

        // BP output. We still do a 2nd op output here which will be a copy of
        // the internal persistent tensor, because Python will want to access 
        // the BP for instrumentation.
        TensorShape out_shape({2});
        Tensor* t_dfx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &t_dfx));
        Tensor* t_vp_internal = m_vp_tensor.AccessTensor(context);
        
        // Compute BP. Here we make a copy of the initial one. This is
        // important, because the ideal BP will be computed here, but should
        // only exist for this tensor. It should be recomputed each time.
        fxbp current_bp(m_bp);
        if (current_bp.m_bp == -1 || !current_bp.m_initialized) {
            const float* d = t_input.flat<float>().data();
            current_bp.set_range_fp(d, d+t_input.flat<float>().size());
        }

        // Assign BP to t_dfx.
        t_dfx->flat<int32_t>()(0) = current_bp.m_wl;
        t_dfx->flat<int32_t>()(1) = current_bp.m_bp;
        t_vp_internal->flat<int32_t>()(0) = current_bp.m_wl;
        t_vp_internal->flat<int32_t>()(1) = current_bp.m_bp;
                
#ifdef DFX_GEN_DEBUG
        const std::string dfx_key = name();
        printf("Gen: key: %s, dfx: (%d,%d)\n", dfx_key.c_str(), current_bp.m_wl, current_bp.m_bp);
#endif
    }
    
private:
    
    fxbp                m_bp;
    PersistentTensor    m_vp_tensor;
    
    static bool m_show_banner;
    
};


bool WaveBpGenOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveBpGen").Device(DEVICE_CPU), WaveBpGenOp);
