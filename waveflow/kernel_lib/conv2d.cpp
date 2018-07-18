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
#include <math.h>

#ifdef OPENMP_OPTIMIZATION
#include <omp.h>
#endif

#include "gemm_fp.h"


// #define CONV2D_DEBUG 1

using namespace tensorflow;

REGISTER_OP("WaveConv2D")
    .Input("input: float")
    .Input("filter: float")
    .Output("output: float")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShape);
 
class WaveConv2DOp : public OpKernel {
public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>  FPMatrix2d;
    typedef TTypes<float>::ConstFlat    FPFlatMatrixC;
    typedef TTypes<float>::Flat         FPFlatMatrix;
    
    explicit WaveConv2DOp(OpKernelConstruction* ctx) : OpKernel(ctx),
        m_mm_input(), m_mm_filters(), m_mm_out() {
        // printf("Calling Wave conv2d() ...\n");
        std::vector<int32> strides;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides));
        m_stride = strides[1];
        // We can only handle a uniform stride for HxW.
        OP_REQUIRES(ctx, (strides[1] == strides[2]),
            errors::InvalidArgument(
            "wave_conv2d requires the strides on H and W to be equal")
        );
        
        std::vector<int32> dilations;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations));
        // We can only handle a dialation of 1 across all dimensions for now.
        OP_REQUIRES(ctx, 
            (dilations[0] == dilations[1] == dilations[2] == dilations[3] == 1),
            errors::InvalidArgument(
            "wave_conv2d requires the dilations attribute to be 1")
        );
        Padding pad;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &pad));
        if (pad == VALID) {
            m_pad = 0;
        } else {
            // SAME
            // If padding is required, set this to 1. It will be recomputed
            // once we get tensor dimensions later on compute().
            m_pad = 1;
            OP_REQUIRES(ctx, (m_stride == 1),
                errors::InvalidArgument(
                "wave_conv2d can't support padding=SAME and stride > 1"));
        }
        // KS: for right now, we will handle only NHWC. TF seems only to have
        // this implemented internally, so it should be the only thing we see.
        string data_format_str;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
        OP_REQUIRES(ctx, (data_format_str == "NHWC"),
            errors::InvalidArgument(
            "wave_conv2d requires the data_format attribute to be NHWC"));        
        if (m_show_banner) {
            printf("WaveConv2DOp() init\n");
            m_show_banner = false;
        }        
    }
  
    void Compute(OpKernelContext* context) override {
    
        // some checks to be sure ...
        DCHECK_EQ(2, context->num_inputs());

        const Tensor& t_activations = context->input(0);    
        const Tensor& t_wts = context->input(1);

        // For now we can only deal with square kernels. This can be lifted
        // in the future.
        const TensorShape& wts_shape = t_wts.shape();
        CHECK_EQ(wts_shape.dim_size(0), wts_shape.dim_size(1));

        OP_REQUIRES(context, (wts_shape.dim_size(0) % 2 != 0),
            errors::InvalidArgument(
            "wave_conv2d requires odd filter size"));

        get_conv_params(t_activations.shape(), wts_shape);
        // out_shape : NHWCout
        TensorShape out_shape({m_batch, m_out_h, m_out_w, m_channel_out});
#ifdef CONV2D_DEBUG
        printf("Out shape: (%d,%d,%d,%d)\n", m_batch, m_out_h, m_out_w, m_channel_out);
#endif
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        convolution(t_activations, t_wts, *output);
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
    
    FPMatrix2d  m_mm_input;
    FPMatrix2d  m_mm_filters;
    FPMatrix2d  m_mm_out;
    
    static bool m_show_banner;
    
    void get_conv_params(const TensorShape& act_shape, const TensorShape& wts_shape)
    {
        DCHECK_EQ(act_shape.dims(), 4);
        DCHECK_EQ(wts_shape.dims(), 4);

        // act_shape : N H W Cin
        // wts_shape : K K Cin Cout
        m_batch = act_shape.dim_size(0);
        m_height = act_shape.dim_size(1);
        m_width = act_shape.dim_size(2);
        m_channel_in = act_shape.dim_size(3);
        m_channel_out = wts_shape.dim_size(3);
        m_kdim = wts_shape.dim_size(0);

        CHECK_EQ(m_channel_in, wts_shape.dim_size(2));

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
        if (m_pad == 0) {
            // VALID
            m_out_h = (m_height - m_kdim) / m_stride + 1;
            m_out_w = (m_width - m_kdim) / m_stride + 1;
        } else {
            // SAME
            m_out_h = m_height / m_stride;
            m_out_w = m_width / m_stride;
            
            // Update the padding value
            float pad_dim = (m_kdim - 1.f) / 2.f;
            m_pad = ceilf(pad_dim);
            
            assert(m_pad < m_kdim);
        }
#ifdef CONV2D_DEBUG
        printf("activations: (%d,%d,%d,%d)\n", act_shape.dim_size(0),
               act_shape.dim_size(1), act_shape.dim_size(2), act_shape.dim_size(3));
        printf("weights: (%d,%d,%d,%d)\n", wts_shape.dim_size(0),
               wts_shape.dim_size(1), wts_shape.dim_size(2), wts_shape.dim_size(3));
        printf("h x w : (%d,%d); ho x wo : (%d,%d)\n", m_height, m_width, m_out_h, m_out_w);
#endif
    }
    
    
    float im2col_get_pixel(const float* im, int row, int col, int channel)
    {
        // printf("get_pixel(): row: %d, col: %d, c: %d\n", row, col, channel);
        
        row -= m_pad;
        col -= m_pad;

        if (row < 0 || col < 0 || row >= m_height || col >= m_width) 
            return 0.f;
        // return im[col + width*(row + m_height*channel)];
        // return im[(m_height*channel + row)*width + col];
        // float v = im(row, col, channel);
        // im : H W Cin
        int idx = (row*m_width + col)*m_channel_in + channel;
        // printf("i2c idx: %d\n", idx);
        float v = im[idx];
        return v;
    }


    void im2col(FPMatrix2d& m_dest, const float* m_src)
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
                    // data_col[col_index] = im2col_get_pixel(data_im, batch,
                    //        im_row, im_col, c_im);
                    // H W, Cin
                    // new_h*new_w, k*k*Cin
                    // 
                    m_dest(h*m_out_w + w, c) = im2col_get_pixel(m_src, im_row, im_col, c_im);
                    // printf("m_dest: (%d, %d) = pix (%d,%d,%d)\n", h*m_out_w + w, c,
                    //     im_row, im_col, c_im);
                }
            }
        }
    }
    
    
    void convolution(const Tensor& t_act, const Tensor& t_wts, Tensor& t_out)
    {
        FPFlatMatrixC m_img_in = t_act.flat<float>();
        FPFlatMatrixC m_wts_in = t_wts.flat<float>();
        FPFlatMatrix m_conv_out = t_out.flat<float>();
        // Input to im2col
        // These matrixes will get allocated the first time this is called, but
        // on subsequent calls no more allocations will happen. Eigen avoids a
        // realloc if the dimensions are identical, which it will be for a given
        // op allocated at a specific point in the graph.
        m_mm_input.resize(m_out_h*m_out_w, m_kdim*m_kdim*m_channel_in);
        m_mm_filters.resize(m_kdim*m_kdim*m_channel_in, m_channel_out);
        m_mm_out.resize(m_out_h*m_out_w, m_channel_out);
        
        // wts_shape : K K Cin Cout
        partial_in(m_mm_filters, m_wts_in.data());

#ifdef OPENMP_OPTIMIZATION
        //printf("omp threads: %d\n", get_omp_nthreads());
        
        #pragma omp parallel for
#endif
        for(int i = 0; i < m_batch; ++i) {
            FPMatrix2d  thread_mm_input(m_out_h*m_out_w, m_kdim*m_kdim*m_channel_in);
            FPMatrix2d  thread_mm_out(m_out_h*m_out_w, m_channel_out);

            // img_in : N Hi Wi Ci
            const float* img_batch = m_img_in.data() + i * m_channel_in*m_height*m_width;
            
            // im2col : (Hi Wi Ci) -> (Ho Wo, K K Ci)
            im2col(thread_mm_input, img_batch);

#ifdef CONV2D_DEBUG
            printf("gemm: B: %d/%d, (%d,%d) * (%d,%d) = (%d,%d)\n", i, m_batch,
                   thread_mm_input.rows(), thread_mm_input.cols(),
                   m_mm_filters.rows(), m_mm_filters.cols(),
                   thread_mm_out.rows(), thread_mm_out.cols());
#endif
            // (Ho Wo, K K Ci) * (K K Ci, Co) = (Ho Wo, Co)
            mm_nn(true, thread_mm_input, thread_mm_input.rows(), thread_mm_input.cols(),
                  m_mm_filters, m_mm_filters.rows(), m_mm_filters.cols(), 
                  thread_mm_out);
            
            // out : N Ho Wo Co
            float* conv_out = m_conv_out.data() + i*m_out_h*m_out_w*m_channel_out;
            partial_out(conv_out, thread_mm_out);
        }

    }

    void partial_in(FPMatrix2d& m_out, const float* flat_arr)
    {
        for (int i = 0; i < m_out.rows(); i++) {
            for (int j = 0; j < m_out.cols(); j++) {
                m_out(i, j) = flat_arr[i*m_out.cols() + j];
            }
        }
    }

    void partial_out(float* conv_out, const FPMatrix2d& m_out)
    {
        for (int i = 0; i < m_out.rows(); i++) {
            for (int j = 0; j < m_out.cols(); j++) {
                conv_out[i*m_out.cols() + j] = m_out(i, j);
            }
        }
    }

#ifdef OPENMP_OPTIMIZATION
    int get_omp_nthreads()
    {
        int t = 1;
        #pragma omp parallel
        {
            if (omp_get_thread_num() == 0) {
                t = omp_get_num_threads();
            }
        }
        return t;
    }
#endif
};

bool WaveConv2DOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveConv2D").Device(DEVICE_CPU), WaveConv2DOp);
