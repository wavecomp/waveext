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

#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "Eigen/Core"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "dyn_fx_pt.h"
#include "gemm_dfx.h"
#include "dfx_op_base.h"


using namespace tensorflow;

REGISTER_OP("WaveConv2DDfxGradientInput")
    .Input("input_sizes: int32")
    .Input("filter: float32")
    .Input("out_backprop: float32")
    .Output("output: float32")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    // Dynamic Fixed Point linkage
    // .Attr("bp_i0: string = ''")
    .Attr("bp_i1: string = ''")
    .Attr("bp_i2: string = ''")
    .Attr("bp_o0: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
        TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
        c->set_output(0, s);
        return Status::OK();
    });

    
class WaveConv2dDfxGradientInputOp : public WaveDynFxPointOp {
public:
    typedef TTypes<float>::ConstFlat    FPFlatMatrixC;
    typedef TTypes<float>::Flat         FPFlatMatrix;
    
    explicit WaveConv2dDfxGradientInputOp(OpKernelConstruction* ctx) 
    : WaveDynFxPointOp(ctx, {"bp_i1", "bp_i2", "bp_o0"}) {
        // printf("Calling Wave conv2d_gradient_input() ...\n");
        std::vector<int32> strides;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides));
        m_stride = strides[0];
        // We can only handle a uniform stride across all dimensions for now.
        OP_REQUIRES(ctx, 
            (strides[0] == strides[1] == strides[2] == strides[3] == 1),
            errors::InvalidArgument(
            "wave_conv2d requires the strides attribute to be 1")
        );
        Padding pad;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &pad));
        if (pad == VALID) {
            m_pad = 0;
        } else {
            // If padding is required, set this to 1. It will be recomputed
            // once we get tensor dimensions later on compute().
            m_pad = 1;
        }
        // KS: for right now, we will handle only NHWC. TF seems only to have
        // this implemented internally, so it should be the only thing we see.
        string data_format_str;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
        OP_REQUIRES(ctx, (data_format_str == "NHWC"),
            errors::InvalidArgument(
            "wave_conv2d requires the data_format attribute to be NHWC"));
        if (m_show_banner) {
            printf("WaveConv2dDfxGradientInputOp() init\n");
            m_show_banner = false;
        }        
    }
  
    void Compute(OpKernelContext* context) override {
    
        DCHECK_EQ(3, context->num_inputs());

        const Tensor& t_gradient = context->input(2);    
        const Tensor& t_filter = context->input(1);
        const Tensor& t_sizes = context->input(0);

        const TensorShape& filter_shape = t_filter.shape();

        OP_REQUIRES(context, (filter_shape.dim_size(0) % 2 != 0),
            errors::InvalidArgument(
            "wave_conv2d_vp_gradient_input requires odd filter size"));

        get_conv_params(t_gradient.shape(), filter_shape, t_sizes);
        // out_shape : N H W Ci
        TensorShape out_shape({m_batch, m_height, m_width, m_channel_in});
                
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        gradient_conv2d_vp_in(t_gradient, t_filter, *output);
    }
    
    
private:
    
    int         m_stride;         // For now just one uniform Conv layer supported
    int         m_pad;
    int         m_batch;
    int         m_height;
    int         m_width;
    int         m_channel_in;
    int         m_channel_out;
    int         m_kdim;
    int         m_out_h;
    int         m_out_w;
    
    static bool m_show_banner;
    
    void get_conv_params(const TensorShape& grad_shape, 
                         const TensorShape& filt_shape,
                         const Tensor& fwdin_shape)
    {
        DCHECK_EQ(grad_shape.dims(), 4);
        DCHECK_EQ(filt_shape.dims(), 4);

        auto fwdin_dims = fwdin_shape.flat<int32>();
        DCHECK_EQ(fwdin_dims.size(), 4);
        
        // grad_shape : N Ho Wo Co
        // filt_shape : K K Cin Cout
        // fwdin_shape : N Hi Wi Ci
        m_batch = fwdin_dims(0);
        m_height = fwdin_dims(1);
        m_width = fwdin_dims(2);
        m_channel_in = fwdin_dims(3);
        m_channel_out = grad_shape.dim_size(3);
        m_kdim = filt_shape.dim_size(0);
        
        CHECK_EQ(m_batch, grad_shape.dim_size(0));
        CHECK_EQ(m_channel_in, filt_shape.dim_size(2));
        CHECK_EQ(m_channel_out, filt_shape.dim_size(3));
        
        /*
        int convolutional_out_height(convolutional_layer l)
        {
            return (l.h + 2*l.pad - l.size) / l.stride + 1;
        }

        int convolutional_out_width(convolutional_layer l)
        {
            return (l.w + 2*l.pad - l.size) / l.stride + 1;
        } 
        */
        // Output tensor is also in the NHWC format. H and W are custom based
        // on this convolution attribs.
        m_out_h = grad_shape.dim_size(1);
        m_out_w = grad_shape.dim_size(2);

        if (m_pad == 0) {
            // VALID
            CHECK_EQ(m_height-m_out_h, m_kdim-1);
            CHECK_EQ(m_width-m_out_w, m_kdim-1);
        } else {
            // SAME
            CHECK_EQ(m_height, m_out_h);
            CHECK_EQ(m_width, m_out_w);

            // Update the padding value
            float pad_dim = (m_kdim - 1.f) / 2.f;
            m_pad = ceilf(pad_dim);
            
            assert(m_pad < m_kdim);
        }

#if 0
        printf("gradients: (%d,%d,%d,%d)\n", grad_shape.dim_size(0),
               grad_shape.dim_size(1), grad_shape.dim_size(2), grad_shape.dim_size(3));
        printf("weights: (%d,%d,%d,%d)\n", filt_shape.dim_size(0),
               filt_shape.dim_size(1), filt_shape.dim_size(2), filt_shape.dim_size(3));
        printf("hi x wi : (%d,%d); ho x wo : (%d,%d)\n", m_height, m_width, m_out_h, m_out_w);
        printf("pad: %d\n", m_pad);
#endif
    }
    
    void gradient_conv2d_vp_in(const Tensor& t_grad, const Tensor& t_wts, Tensor& t_out)
    {
        FPFlatMatrixC m_grad_in = t_grad.flat<float>();
        FPFlatMatrixC m_wts_in = t_wts.flat<float>();
        FPFlatMatrix m_conv_out = t_out.flat<float>();
        // Input to col2im
        // DFXMatrix2d  mm_grad(m_out_h*m_out_w, m_channel_out);
        DFXMatrix2d  mm_filters(m_kdim*m_kdim*m_channel_in, m_channel_out);
        // DFXMatrix2d  mm_out(m_kdim*m_kdim*m_channel_in, m_out_h*m_out_w);

        // DFXMatrix2d  c2i(m_height*m_width, m_channel_in);
        
        fxbp bp_filter = get_fxbp(true, 1);
        fxbp bp_grad = get_fxbp(true, 2);
        fxbp bp_out = get_fxbp(false, 0);
        
        // wts_shape : K K Cin Cout
        fp2dfx(bp_filter, mm_filters, m_wts_in.data());

#ifdef OPENMP_OPTIMIZATION
        #pragma omp parallel for
#endif
        for(int i = 0; i < m_batch; ++i) {
            DFXMatrix2d  thread_mm_grad(m_out_h*m_out_w, m_channel_out);
            DFXMatrix2d  thread_mm_out(m_kdim*m_kdim*m_channel_in, m_out_h*m_out_w);
            DFXMatrix2d  thread_c2i(m_height*m_width, m_channel_in);
            
            // grad_in : N Ho Wo Co
            const float* grad_batch = m_grad_in.data() + i * m_channel_out*m_out_h*m_out_w;
            
            // New gradient fxbp for each call; this param is mutable.
            fxbp th_bp_grad(bp_grad);
            fp2dfx(th_bp_grad, thread_mm_grad, grad_batch);
#if 0
            printf("gemm: (%d,%d) * (%d,%d) = (%d,%d)\n", 
                   mm_filters.rows(), mm_filters.cols(),
                   thread_mm_grad.cols(), thread_mm_grad.rows(),
                   thread_mm_out.rows(), thread_mm_out.cols());
#endif
            // (K K Cin, Cout) * (Ho Wo, Co)^T = (K K Cin, Ho Wo)
            mm_nt(true, fxbp(16, 32), mm_filters, mm_filters.rows(), mm_filters.cols(),
                  thread_mm_grad, thread_mm_grad.rows(), thread_mm_grad.cols(),
                  thread_mm_out);

            // Clear the output matrix first.
            dfx_clear(fxbp(16, 32), thread_c2i, thread_c2i.rows(), thread_c2i.cols());
            
            // (Cin K K, Ho Wo) -> (Hi Wi Ci)
            col2im(thread_c2i, thread_mm_out);

            // out : N Hi Wi Ci
            float* conv_out = m_conv_out.data() + i*m_height*m_width*m_channel_in;
            // New out fxbp for each call; this param is mutable.
            fxbp th_out_grad(bp_out);
            dfx2fp(th_out_grad, conv_out, thread_c2i);
        }

    }
    
    void col2im_add_pixel(DFXMatrix2d& im, int row, int col, int channel, 
                          const DynFxPoint& val)
    {
        row -= m_pad;
        col -= m_pad;

        if (row < 0 || col < 0 || row >= m_height || col >= m_width) 
            return;
        // im[col + m_width*(row + m_height*channel)] += val;
        // (H W Cin)
        // im[(row*m_width + col)*m_channel_in + channel] += val;
        im(row*m_width + col, channel) += val;
    }

    void col2im(DFXMatrix2d& m_dest, const DFXMatrix2d& m_src)
    {
        int channels_col = m_channel_in * m_kdim * m_kdim;
        for (int c = 0; c < channels_col; ++c) {
            int c_im = c % m_channel_in;
            int w_offset = (c / m_channel_in) % m_kdim;
            int h_offset = c / m_channel_in / m_kdim;
            for (int h = 0; h < m_out_h; ++h) {
                for (int w = 0; w < m_out_w; ++w) {
                    int im_row = h_offset + h * m_stride;
                    int im_col = w_offset + w * m_stride;
                    // int col_index = (c * m_out_h + h) * m_out_w + w;
                    // (C K K, H W)
                    // float val = data_col[col_index];
                    // float val = m_src(c, h*m_out_w + w);
                    col2im_add_pixel(m_dest, im_row, im_col, c_im,
                        m_src(c, h*m_out_w + w));
                }
            }
        }
    }
};

bool WaveConv2dDfxGradientInputOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveConv2DDfxGradientInput").Device(DEVICE_CPU), WaveConv2dDfxGradientInputOp);
