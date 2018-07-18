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
#include <math.h>
#include <assert.h>
#include <algorithm>

#include "dyn_fx_pt.h"
#include "dfx_op_base.h"
#include "dfx_registry.h"


// #define DFX_GEN_DEBUG 1

using namespace tensorflow;

REGISTER_OP("Cor15BpGen")
    .Input("tensor_in: float")
    .Output("bp_out: int32")
    .Attr("recompute_interval: int = 100")
    .Attr("ovf_rate: float = 0.01")
    // Dynamic Fixed Point linkage
    .Attr("bp_o0: string = ''")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // c->set_output(0, c->input(0));
      return Status::OK();
    });

/* This class implements the BP selection method described in M. Courbariaux's
 * 2015 paper TRAINING DEEP NEURAL NETWORKS WITH LOW PRECISION MULTIPLICATIONS.
 * The intent is to follow the original description faithfully with no
 * optimizations.
 * */
class Cor15BPGenOp : public WaveDynFxPointOp {
public:
    
    explicit Cor15BPGenOp(OpKernelConstruction* ctx) 
    : WaveDynFxPointOp(ctx, {"bp_o0"}), m_bp(get_fxbp(false, 0)), m_interval_count(0)
    {
        // If the BP isn't specified, default to 16b auto.
        if (!m_bp.m_initialized) {
            m_bp.initialize(-1, 16);
        }
        if (m_show_banner) {
            printf("Cor15BPGenOp() init\n");
            m_show_banner = false;
        }
        OP_REQUIRES_OK(ctx, ctx->GetAttr("recompute_interval", &m_rec_interval));
        OP_REQUIRES(ctx, (m_rec_interval > 0),
            errors::InvalidArgument(
            "Cor15BPGen requires recompute_interval > 0"));

        OP_REQUIRES_OK(ctx, ctx->GetAttr("ovf_rate", &m_ovf_rate));
        OP_REQUIRES(ctx, (m_ovf_rate >= 0.f && m_ovf_rate <= 1.f),
            errors::InvalidArgument(
            "Cor15BPGen requires (1.0 >= ovf_rate >= 0)"));

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
        
#ifdef DFX_GEN_DEBUG
        printf("Op: %s, bp: (%d,%d)\n", name().c_str(), m_bp.m_wl, m_bp.m_bp);
#endif
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& t_input = context->input(0);

        // BP output.
        TensorShape out_shape({2});
        Tensor* t_dfx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &t_dfx));
        Tensor* t_vp_internal = m_vp_tensor.AccessTensor(context);
        
        // Compute BP.
        if (m_interval_count == 0 || m_bp.m_bp == -1 || !m_bp.m_initialized) {
            const float* d = t_input.flat<float>().data();
            m_bp.set_range_fp(d, d+t_input.flat<float>().size());
#ifdef DFX_GEN_DEBUG
            printf("Op: %s, recompute bp: (%d,%d)\n", name().c_str(), m_bp.m_wl, m_bp.m_bp);
#endif
        } else {
            // If we are inside the recomputation interval, we need to look at
            // overflow/underflow. 
            bp_adjust(t_input);
        }
        assert(m_bp.m_bp < 32);
        assert(m_bp.m_bp >= 0);
        
        // Assign BP to t_dfx.
        t_dfx->flat<int32_t>()(0) = m_bp.m_wl;
        t_dfx->flat<int32_t>()(1) = m_bp.m_bp;
        t_vp_internal->flat<int32_t>()(0) = m_bp.m_wl;
        t_vp_internal->flat<int32_t>()(1) = m_bp.m_bp;

        // Interval count must be per-batch, because some tensors don't have
        // the batch size dimension. Without a local copy of the batch size,
        // we don't know how many samples we have actually processed. So the
        // caller needs to handle this abstraction for us.
        m_interval_count = (m_interval_count+1) % m_rec_interval;

#ifdef DFX_GEN_DEBUG
        printf("Op: %s, count: %d, dfx: (%d,%d)\n", name().c_str(), 
               m_interval_count, m_bp.m_wl, m_bp.m_bp);
#endif
        
    }
    
private:
    
    fxbp                m_bp;
    int32_t             m_interval_count;
    int32_t             m_rec_interval;
    float               m_ovf_rate;
    PersistentTensor    m_vp_tensor;
    
    static bool m_show_banner;
    
    // BP adjustment happens here. We will scan the tensor in FP32 format, since
    // that will prevent the need to recompute any excessive saturation/
    // underflow conditions on a 2nd pass. The final effect will be the same, 
    // since we adjust the BP +/-1 based on the findings.
    void bp_adjust(const Tensor& t)
    {
        // Scan the tensor and count the 2 overflow values.
        const float ovf = (((uint32_t)1 << (m_bp.m_wl-1)) - 1) / m_bp.epsilon();
        const float ovf2 = ovf * 0.5f;
        
        int ov2_count = 0;
        int ovf_count = 0;
        
        auto m = t.flat<float>();
        for (int i = 0; i < m.size(); i++) {
            const float v = fabs(m(i));
            ov2_count += (v > ovf2) ? 1 : 0;
            ovf_count += (v > ovf) ? 1 : 0;
        }
        
        // If the counts exceed the limit, then we increment/decrement the BP.
        int maxval = m_ovf_rate * (float)m.size();
        maxval = (maxval < 1) ? 1 : maxval;
        // Greater BP means more fraction bits. Full overflow is checked first.
        // If we overflow the simple bound, we always add 1 integer bit. After,
        // if we are safely below the half-max bound threshold, we can safely
        // give an extra fraction bit.
        if (ovf_count >= maxval) {
            // Min BP is 0
            m_bp.m_bp -= (m_bp.m_bp > 0) ? 1 : 0;
        } else if (ov2_count < maxval) {
            // Max BP is 31
            m_bp.m_bp += (m_bp.m_bp < 31) ? 1 : 0;
        }
#ifdef DFX_GEN_DEBUG
        printf("Op: %s, tsz: %d, thresh u/u2: %.6f/%.6f, maxcnt: %d, ovf/ov2: %d/%d, bp: (%d,%d)\n",
            name().c_str(), (int)m.size(), ovf, ovf2, maxval, ovf_count, ov2_count,
            m_bp.m_wl, m_bp.m_bp
        );
#endif
        assert(m_bp.m_bp < 32);
        assert(m_bp.m_bp >= 0);
    }
    
};


bool Cor15BPGenOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("Cor15BpGen").Device(DEVICE_CPU), Cor15BPGenOp);
