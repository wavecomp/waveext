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
#include "tensorflow/core/util/work_sharder.h"

#include "dyn_fx_pt.h"

using namespace tensorflow;

REGISTER_OP("WaveMaxPoolDfx")
    .Attr(
        "T: {float} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("data_format: {'NHWC', 'NCHW', 'NCHW_VECT_C'} = 'NHWC'")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::MaxPoolShape);

class WaveMaxPool2DDfxOp : public OpKernel {
public:
    typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;

    typedef Eigen::Matrix<DynFxPoint, Eigen::Dynamic, Eigen::Dynamic> DFXMatrix2d;

    explicit WaveMaxPool2DDfxOp(OpKernelConstruction *context) : OpKernel(context) {
        string data_format;
        auto status = context->GetAttr("data_format", &data_format);
        if (status.ok()) {
            OP_REQUIRES(context, FormatFromString(data_format, &m_data_format),
                        errors::InvalidArgument("Invalid data format"));
            OP_REQUIRES(
                context, m_data_format == FORMAT_NHWC,
                errors::InvalidArgument("wave_maxpool2d only supports NHWC ",
                                        "on device type ",
                                        DeviceTypeString(context->device_type())));
        } else {
            m_data_format = FORMAT_NHWC;
        }

        OP_REQUIRES_OK(context, context->GetAttr("ksize", &m_ksize));
        OP_REQUIRES(context, m_ksize.size() == 4,
                    errors::InvalidArgument("Sliding window ksize field must "
                                            "specify 4 dimensions"));

        OP_REQUIRES_OK(context, context->GetAttr("strides", &m_stride));
        OP_REQUIRES(context, m_stride.size() == 4,
                    errors::InvalidArgument("Sliding window stride field must "
                                            "specify 4 dimensions"));

        OP_REQUIRES_OK(context, context->GetAttr("padding", &m_padding));

        OP_REQUIRES(context, m_ksize[0] == 1 && m_stride[0] == 1,
                    errors::Unimplemented(
                        "Pooling is not yet supported on the batch dimension."));

        if (m_show_banner) {
            printf("WaveMaxPool2DDfxOp() init\n");
            m_show_banner = false;
        }
    }

    void Compute(OpKernelContext *context) override {
        const Tensor &tensor_in = context->input(0);

        m_depth = GetTensorDim(tensor_in.shape(), FORMAT_NHWC, 'C');
        m_tensor_in_cols = GetTensorDim(tensor_in.shape(), FORMAT_NHWC, 'W');
        m_tensor_in_rows = GetTensorDim(tensor_in.shape(), FORMAT_NHWC, 'H');
        m_tensor_in_batch = GetTensorDim(tensor_in.shape(), FORMAT_NHWC, 'N');
        m_window_rows = GetTensorDim(m_ksize, FORMAT_NHWC, 'H');
        m_window_cols = GetTensorDim(m_ksize, FORMAT_NHWC, 'W');
        m_depth_window = GetTensorDim(m_ksize, FORMAT_NHWC, 'C');
        m_row_stride = GetTensorDim(m_stride, FORMAT_NHWC, 'H');
        m_col_stride = GetTensorDim(m_stride, FORMAT_NHWC, 'W');
        m_depth_stride = GetTensorDim(m_stride, FORMAT_NHWC, 'C');

        // We only support 2D pooling across width/height and depthwise
        // pooling, not a combination.
        OP_REQUIRES(context,
                    (m_depth_window == 1 || (m_window_rows == 1 && m_window_cols == 1)),
                    errors::Unimplemented(
                        "MaxPooling supports exactly one of pooling across depth "
                        "or pooling across width/height."));

        if (m_depth_window == 1) {
            OP_REQUIRES_OK(
                context, GetWindowedOutputSize(m_tensor_in_rows, m_window_rows, m_row_stride,
                                               m_padding, &m_out_height, &m_pad_rows));
            OP_REQUIRES_OK(
                context, GetWindowedOutputSize(m_tensor_in_cols, m_window_cols, m_col_stride,
                                               m_padding, &m_out_width, &m_pad_cols));
            m_pad_depth = 0;
            m_out_depth = m_depth;
        } else {
            // Our current version of depthwise max pooling does not support
            // any padding, and expects the depth_window to equal the
            // depth_stride (no overlapping).
            OP_REQUIRES(
                context, m_depth % m_depth_window == 0,
                errors::Unimplemented("Depthwise max pooling requires the depth "
                                      "window to evenly divide the input depth"));
            OP_REQUIRES(
                context, m_depth_stride == m_depth_window,
                errors::Unimplemented("Depthwise max pooling requires the depth "
                                      "window to equal the depth stride"));

            m_pad_depth = 0;
            m_out_depth = m_depth / m_depth_window;
        }

        if (!context->status().ok()) {
            return;
        }

        TensorShape output_shape;

        if (m_depth_window == 1) {
            // Spatial pooling
            output_shape = ShapeFromFormat(FORMAT_NHWC, m_tensor_in_batch, m_out_height, m_out_width,
                                           m_depth);
        } else {
            // Depthwise pooling
            output_shape = TensorShape(
                {m_tensor_in_batch, m_tensor_in_rows, m_tensor_in_cols, m_out_depth});
        }

        Tensor *tensor_out = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, output_shape, &tensor_out));

        if (m_depth_window > 1) {
            // Validate spec against the current implementation.  A
            // relaxation of these requirements would be ideal.
            OP_REQUIRES(context, m_depth % m_depth_window == 0,
                        errors::Unimplemented(
                            "Depthwise max pooling requires "
                            "the depth window to evenly divide the input depth."));
            OP_REQUIRES(
                context, m_depth_window == m_depth_stride,
                errors::Unimplemented("Depthwise max pooling requires "
                                      "the depth window to equal the depth stride."));

            DepthwiseMaxPool(context, tensor_out, tensor_in);
        } else {
            SpatialMaxPool(context, tensor_out, tensor_in, m_padding);
        }
    }

        // Init a DynFxPoint matrix based on an existing FP matrix.
    void matrix_init(DFXMatrix2d& m_dest,  const Tensor& t_src) {
        ConstEigenMatrixMap m_src(t_src.flat<float>().data(),
                                  m_dest.rows(), m_dest.cols());

        // Compute the ideal BP
        fxbp dest_bp(0, 16);
        dest_bp.set_range_fp(m_src.data(), m_src.data() + m_src.size());

        for (int i = 0; i < m_src.rows(); i++) {
            for (int j = 0; j < m_src.cols(); j++) {
                m_dest(i, j).set_fxbp(dest_bp);
                m_dest(i, j) = m_src(i, j);
            }
        }
     }


      // Convert a DynFxPoint matrix back to an FP tensor.
    void matrix_out(Tensor* t_dest, const DFXMatrix2d& m_src) {
        EigenMatrixMap m_dest(t_dest->flat<float>().data(),
                              m_src.rows(), m_src.cols());

        fxbp out_bp(0, 16);
        out_bp.set_range_dfx(m_src.data(), m_src.data() + m_src.size());

        for (int i = 0; i < m_src.rows(); i++) {
            for (int j = 0; j < m_src.cols(); j++) {
                DynFxPoint v(out_bp);
                v = m_src(i, j);
                m_dest(i, j) = v.to_fp();
            }
        }
    }


    void SpatialMaxPool(OpKernelContext *context, Tensor *tensor_out,
                        const Tensor &tensor_in, const Padding &padding)
    {
        DFXMatrix2d in_mat(m_depth,  m_tensor_in_cols * m_tensor_in_rows *
                                   m_tensor_in_batch);

        matrix_init(in_mat, tensor_in);

        DFXMatrix2d out_mat( m_depth, m_out_width * m_out_height * m_tensor_in_batch);

        // Initializes the output tensor with the lowest (negative) value.
        int32 out_matrix_width = m_out_width * m_out_height * m_tensor_in_batch;
        fxbp out_bp = in_mat(0, 0).get_fxbp();
        for (int32 m = 0; m < m_depth; ++m)
        {
            for (int32 n = 0; n < out_matrix_width; ++n)
            {
                out_mat(m,n).set_fxbp(out_bp);
                out_mat(m,n).set_bits(-32768);
            }
        }

        // The following code basically does the following:
        // 1. Flattens the input and output tensors into two dimensional arrays.
        //    tensor_in_as_matrix:
        //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
        //    output_as_matrix:
        //      depth by (out_width * out_height * tensor_in_batch)
        //
        // 2. Walks through the set of columns in the flattened
        // tensor_in_as_matrix,
        //    and updates the corresponding column(s) in output_as_matrix with the
        //    max value.
        auto shard = [this, &in_mat, &out_mat](int64 start, int64 limit) {

            for (int b = start; b < limit; ++b) {
                for (int h = 0; h < m_tensor_in_rows; ++h) {
                    for (int w = 0; w < m_tensor_in_cols; ++w) {
                        // (h_start, h_end) * (w_start, w_end) is the range that the input
                        // vector projects to.
                        const int hpad = h + m_pad_rows;
                        const int wpad = w + m_pad_cols;
                        const int h_start = (hpad < m_window_rows)
                                            ? 0 : (hpad - m_window_rows) / m_row_stride + 1;
                        const int h_end = std::min<int>(hpad / m_row_stride + 1, m_out_height);
                        const int w_start = (wpad < m_window_cols)
                                            ? 0 : (wpad - m_window_cols) / m_col_stride + 1;
                        const int w_end = std::min<int>(wpad / m_col_stride + 1, m_out_width);
                        const int in_offset = (b * m_tensor_in_rows + h) * m_tensor_in_cols + w;

                        for (int ph = h_start; ph < h_end; ++ph) {
                            for (int pw = w_start; pw < w_end; ++pw) {
                                const int out_offset = (b * m_out_height + ph) * m_out_width + pw;
                                for (int32 i = 0; i < m_depth; ++i) {
                                    if (out_mat(i, out_offset) < in_mat(i, in_offset)) {
                                        out_mat(i, out_offset) = in_mat(i, in_offset);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        //   const DeviceBase::CpuWorkerThreads& worker_threads =
        //       *(context->device()->tensorflow_cpu_worker_threads());

        //   // TODO(andydavis) Consider sharding across batch x rows x cols.
        //   // TODO(andydavis) Consider a higher resolution shard cost model.
        //   const int64 shard_cost =  m_tensor_in_rows * m_tensor_in_cols * m_depth;
        //   Shard(worker_threads.num_threads, worker_threads.workers,
        //         m_tensor_in_batch, shard_cost, shard);

        // Call sharding function directly !!!
        shard(0, m_tensor_in_batch);
        matrix_out(tensor_out, out_mat);
    }

    void DepthwiseMaxPool(OpKernelContext* context, Tensor* tensor_out,
                          const Tensor& tensor_in) {

        DFXMatrix2d in_mat(m_depth_window,  tensor_in.NumElements() / m_depth_window);

        matrix_init(in_mat, tensor_in);

        DFXMatrix2d out_mat(1,tensor_out->NumElements());

        for (int j = 0; j < in_mat.cols(); ++j) {
            out_mat(0, j) = in_mat(0, j);
            for (int i = 1; i < in_mat.rows(); ++i) {
                if (out_mat(0, j) < in_mat(i, j)) {
                    out_mat(0, j) = in_mat(i, j);
                }
            }
        }

        matrix_out(tensor_out, out_mat);
    }

  private:
    std::vector<int32> m_ksize;
    std::vector<int32> m_stride;
    Padding m_padding;
    TensorFormat m_data_format;

    int m_depth;

    int m_tensor_in_cols;
    int m_tensor_in_rows;
    int m_tensor_in_batch;

    int m_window_rows;
    int m_window_cols;
    int m_depth_window;

    int m_row_stride;
    int m_col_stride;
    int m_depth_stride;

    int64 m_out_height;
    int64 m_out_width;
    int m_out_depth;

    int64 m_pad_rows;
    int64 m_pad_cols;
    int m_pad_depth;

  private:
    static bool m_show_banner;
};

bool WaveMaxPool2DDfxOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveMaxPoolDfx").Device(DEVICE_CPU), WaveMaxPool2DDfxOp);
