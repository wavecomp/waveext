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
#include "tanh_lut_dfx.h"

using namespace tensorflow;

REGISTER_OP("WaveTanhDfx")
    .Input("a: float")
    .Output("z: float")
    .Attr("bp_i0: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

class WaveTanhDfxOp : public WaveDynFxPointOp {
public:
    typedef std::vector<DynFxPoint> DFXVector;

    explicit WaveTanhDfxOp(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"bp_i0", "bp_o0"}), m_mm_a_dfx(), m_mm_z_dfx()
    {
        if (m_show_banner) {
            printf("WaveTanhDfxOp() init\n");
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

        auto a_m = tensor_a.flat<float>();
        auto z_m = output->flat<float>();

        m_mm_a_dfx.resize(a_m.size());
        m_mm_z_dfx.resize(z_m.size());

        fxbp a_bp = get_fxbp(true, 0);
        partial_in(a_bp, m_mm_a_dfx, a_m.data());

        for (int i = 0; i < a_m.size(); i++) {
            m_mm_z_dfx[i] = tanh(m_mm_a_dfx[i]);
        }

        fxbp z_bp = get_fxbp(false, 0);
        if (z_bp.m_bp == -1 || !z_bp.m_initialized)
            convert_output(z_m.data(), m_mm_z_dfx);
        else
            partial_out(z_bp, z_m.data(), m_mm_z_dfx);
    }

private:

    DFXVector m_mm_a_dfx;
    DFXVector m_mm_z_dfx;

    static bool m_show_banner;

    void convert_output(float* conv_out, const DFXVector& m_out)
    {
        for (int i = 0; i < m_out.size(); i++) {
            conv_out[i] = m_out[i].to_fp();
        }
    }

    DynFxPoint tanh(DynFxPoint x)
    {
        int vx, i, r;
        DynFxPoint res, v1, v2, vr;
        fxbp fx;

        res.set_fxbp(14, 16);
        fx = x.get_fxbp();

        vx = x.get_value();
        i = vx < 0 ? -vx : vx;

        if (fx.m_bp > 15)
            r = (i >> (fx.m_bp - 15)) & 0x3ff;
        else
            r = (i << (15 - fx.m_bp)) & 0x3ff;
        if (fx.m_bp > 5)
            i = i >> (fx.m_bp - 5);
        else
            i = i << (5 - fx.m_bp);

        if (i > TANH_LUT_SIZE - 2)
            res.set_bits(16384);
        else {
            v1.set_fxbp(14, 16);
            v1.set_bits(tanh_lut[i]);
            v2.set_fxbp(14, 16);
            v2.set_bits(tanh_lut[i+1]);
            vr.set_fxbp(10, 16);
            vr.set_bits(r);
            v2 = v1 - v2;
            v2 = vr * v2;
            res = v1 - v2;
        }

        if (vx < 0)
            res = -res;

        return res;
    }
};

bool WaveTanhDfxOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveTanhDfx").Device(DEVICE_CPU), WaveTanhDfxOp);
