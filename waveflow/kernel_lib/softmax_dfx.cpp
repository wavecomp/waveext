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
#include "softmax_lut_dfx.h"

using namespace tensorflow;

REGISTER_OP("WaveSoftmaxDfx")
    .Input("a: float")
    .Output("z: float")
    .Attr("bp_i0: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

class WaveSoftmaxDfxOp : public WaveDynFxPointOp {
public:
    typedef std::vector<DynFxPoint> DFXVector;

    explicit WaveSoftmaxDfxOp(OpKernelConstruction* ctx)
    : WaveDynFxPointOp(ctx, {"bp_i0", "bp_o0"}), m_mm_a_dfx(), m_mm_z_dfx()
    {
        if (m_show_banner) {
            printf("WaveSoftmaxDfxOp() init\n");
            m_show_banner = false;
        }
    }

    void Compute(OpKernelContext* context) override
    {
        DCHECK_EQ(1, context->num_inputs());

        const Tensor& tensor_a = context->input(0);
        const TensorShape& a_shape = tensor_a.shape();
        const int last_dim_a = a_shape.dims() - 1; 
        const int n = a_shape.dim_size(last_dim_a);

        TensorShape out_shape(a_shape);
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        auto a_m = tensor_a.flat<float>();
        auto z_m = output->flat<float>();

        m_mm_a_dfx.resize(a_m.size());
        m_mm_z_dfx.resize(z_m.size());

        fxbp a_bp = get_fxbp(true, 0);
        partial_in(a_bp, m_mm_a_dfx, a_m.data());

        for (int i = 0; i < a_m.size(); i += n) {
            softmax(m_mm_a_dfx.data() + i, m_mm_z_dfx.data() + i, n);
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

    void softmax(DynFxPoint *x, DynFxPoint *y, int n)
    {
        int i;
        DynFxPoint max, s, r;

        fxbp bp = x[0].get_fxbp();
        bp.m_wl = 32;
        max.set_fxbp(bp);
        max.set_bits(-32768);
        s.set_fxbp(14, 32);
        s = 0;

        for (i = 0; i < n; i++)
            if (x[i] > max)
                max = x[i];

        for (i = 0; i < n; i++) {
            y[i] = exp(max - x[i]);
            s = s + y[i];
        }

        bp = s.get_fxbp();
        bp.m_bp += __builtin_clz(s.get_value()) - 17;
        bp.m_wl = 16;
        r.set_fxbp(bp);

        r = s;
        r = recip(r);

        for (i = 0; i < n; i++)
            y[i] = y[i] * r;
    }

    // calculate fixed-point exp(-x)
    // this function assumes that x is
    // nonnegative
    DynFxPoint exp(DynFxPoint x)
    {
        int vx, i, r;
        DynFxPoint res, v1, v2, vr;
        fxbp fx;

        res.set_fxbp(14, 16);
        fx = x.get_fxbp();

        vx = x.get_value();
        i = vx;

        if (fx.m_bp > 15)
            r = (i >> (fx.m_bp - 15)) & 0x3ff;
        else
            r = (i << (15 - fx.m_bp)) & 0x3ff;
        if (fx.m_bp > 5)
            i = i >> (fx.m_bp - 5);
        else
            i = i << (5 - fx.m_bp);

        if (i > EXP_LUT_SIZE - 2)
            res.set_bits(0);
        else {
            v1.set_fxbp(14, 16);
            v1.set_bits(exp_lut[i]);
            v2.set_fxbp(14, 16);
            v2.set_bits(exp_lut[i+1]);
            vr.set_fxbp(10, 16);
            vr.set_bits(r);
            v2 = v1 - v2;
            v2 = vr * v2;
            res = v1 - v2;
        }

        return res;
    }

    // calculate fixed-point reciprocal
    // this function assumes that x is normalized
    // i.e. ideal bp is already set by the caller
    DynFxPoint recip(DynFxPoint x)
    {
        int vx, v, i;
        DynFxPoint res, t, two;
        fxbp bp;

        vx = x.get_value();
        i = vx < 0 ? -vx : vx;
        i = (i >> 7) - 128;

        // get initial approximation from the lookup table
        v = recip_lut[i];
        res.set_bits(v);
        bp = x.get_fxbp();
        bp.m_bp = 28 - bp.m_bp;
        res.set_fxbp(bp);

        t.set_fxbp(13, 16);
        two.set_fxbp(13, 16);
        two = 2.0f;

        // refine approximation using Newton's method
        // two iterations are sufficient
        t = res * x;
        t = two - t;
        res = res * t;

        t = res * x;
        t = two - t;
        res = res * t;

        if (vx < 0)
            res = -res;

        return res;
    }
};

bool WaveSoftmaxDfxOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveSoftmaxDfx").Device(DEVICE_CPU), WaveSoftmaxDfxOp);
