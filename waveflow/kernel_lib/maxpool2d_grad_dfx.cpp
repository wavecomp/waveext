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
#include "gemm_dfx.h"

using namespace tensorflow;

static const int kInvalidMaxPoolingIndex = -1;

REGISTER_OP("WaveMaxPoolGradDfx")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: realnumbertype = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    });


// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
class WaveMaxPool2DGradDfxOp : public OpKernel {
public:

    typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>> EigenIndexMatrixMap;

    typedef Eigen::Map<Eigen::Matrix<DynFxPoint, Eigen::Dynamic, Eigen::Dynamic>> DFXMatrix2dMap;
    typedef Eigen::Matrix<DynFxPoint, Eigen::Dynamic, Eigen::Dynamic> DFXMatrix2d;

    explicit WaveMaxPool2DGradDfxOp(OpKernelConstruction* context) : OpKernel(context) {
        string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &m_data_format),
                    errors::InvalidArgument("Invalid data format"));
        OP_REQUIRES(
            context, m_data_format == FORMAT_NHWC,
            errors::InvalidArgument("Default WaveMaxPool2DGradOp only supports NHWC ",
                                    "on device type ",
                                    DeviceTypeString(context->device_type())));

        if (context->num_inputs() == 3) {
        OP_REQUIRES_OK(context, context->GetAttr("ksize", &m_ksize));
        OP_REQUIRES(context, m_ksize.size() == 4,
                    errors::InvalidArgument("Sliding window ksize field must "
                                            "specify 4 dimensions"));
        OP_REQUIRES_OK(context, context->GetAttr("strides", &m_stride));
        OP_REQUIRES(context, m_stride.size() == 4,
                    errors::InvalidArgument("Sliding window strides field must "
                                            "specify 4 dimensions"));
        OP_REQUIRES(context, m_ksize[0] == 1 && m_stride[0] == 1,
                    errors::Unimplemented(
                        "Pooling is not yet supported on the batch dimension."));
        OP_REQUIRES(
            context, m_ksize[3] == 1 && m_stride[3] == 1,
            errors::Unimplemented(
                "MaxPoolingGrad is not yet supported on the depth dimension."));
        }

        OP_REQUIRES_OK(context, context->GetAttr("padding", &m_padding));

        if (m_show_banner) {
            printf("WaveMaxPool2DGradDfxOp() init\n");
            m_show_banner = false;
        }
    }

  void Compute(OpKernelContext* context) override {
        const Tensor& tensor_in = context->input(0);
        const Tensor& tensor_out = context->input(1);
        const Tensor& out_backprop = context->input(2);

        // For maxpooling, tensor_in should have 4 dimensions.
        OP_REQUIRES(context, tensor_in.dims() == 4,
                    errors::InvalidArgument("tensor_in must be 4-dimensional"));
        OP_REQUIRES(context, tensor_out.dims() == 4,
                    errors::InvalidArgument("tensor_out must be 4-dimensional"));
        // For maxpooling, out_backprop should have 4 dimensions.
        OP_REQUIRES(context, out_backprop.dims() == 4,
                    errors::InvalidArgument("out_backprop must be 4-dimensional"));

        const TensorShape& output_shape = tensor_in.shape();

        Tensor tensor_out_dup;
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_temp(
                                    {1}, DataTypeToEnum<float>::v(), tensor_out.shape(),
                                    &tensor_out_dup));
        Tensor tensor_out_arg_max;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64>::v(),
                                                    tensor_out.shape(),
                                                    &tensor_out_arg_max));
        std::vector<int32> ksize = m_ksize;
        std::vector<int32> stride = m_stride;
        if (context->num_inputs() == 5) {
            const Tensor& tensor_ksize = context->input(3);
            auto value_ksize = tensor_ksize.flat<int32>();
            ksize.resize(tensor_ksize.shape().num_elements());
            std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

            const Tensor& tensor_stride = context->input(4);
            auto value_stride = tensor_stride.flat<int32>();
            stride.resize(tensor_stride.shape().num_elements());
            std::copy_n(&value_stride(0), stride.size(), stride.begin());
        }

        OP_REQUIRES(context, ksize.size() == 4,
                    errors::InvalidArgument("Sliding window ksize field must "
                                            "specify 4 dimensions"));
        OP_REQUIRES(context, stride.size() == 4,
                    errors::InvalidArgument("Sliding window strides field must "
                                            "specify 4 dimensions"));
        OP_REQUIRES(context, ksize[0] == 1 && stride[0] == 1,
                    errors::Unimplemented(
                        "Pooling is not yet supported on the batch dimension."));
        OP_REQUIRES(
            context, ksize[3] == 1 && stride[3] == 1,
            errors::Unimplemented(
                "MaxPoolingGrad is not yet supported on the depth dimension."));

        m_depth = GetTensorDim(tensor_in.shape(), FORMAT_NHWC, 'C');
        m_tensor_in_cols = GetTensorDim(tensor_in.shape(), FORMAT_NHWC, 'W');
        m_tensor_in_rows = GetTensorDim(tensor_in.shape(), FORMAT_NHWC, 'H');
        m_tensor_in_batch = GetTensorDim(tensor_in.shape(), FORMAT_NHWC, 'N');
        m_window_rows = GetTensorDim(ksize, FORMAT_NHWC, 'H');
        m_window_cols = GetTensorDim(ksize, FORMAT_NHWC, 'W');
        m_depth_window = GetTensorDim(ksize, FORMAT_NHWC, 'C');
        m_row_stride = GetTensorDim(stride, FORMAT_NHWC, 'H');
        m_col_stride = GetTensorDim(stride, FORMAT_NHWC, 'W');
        m_depth_stride = GetTensorDim(stride, FORMAT_NHWC, 'C');

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

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {0}, 0, output_shape, &output));

        SpatialMaxPoolWithArgMaxHelper(context, &tensor_out_dup,
                                      &tensor_out_arg_max, output, tensor_in,
                                     out_backprop);
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

    void matrix_init(DFXMatrix2d& m_dest)
    {
        dfx_clear(fxbp(16, 32), m_dest, m_dest.rows(), m_dest.cols());
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

    // Check that 0 <= index < limit using a single comparison, assuming
    // that 0 <= limit if Index is signed.  Intended for use in performance
    // critical contexts where 0 <= index < limit is almost always true.
    EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool FastBoundsCheck(const int index,
                                                            const int limit) {
        static_assert(std::is_integral<int>::value && std::is_integral<int>::value,
                        "FastBoundsCheck can only be used on integer types.");
        typedef typename std::make_unsigned<decltype(index + limit)>::type UIndex;
        return TF_PREDICT_TRUE(static_cast<UIndex>(index) <
                                static_cast<UIndex>(limit));
    }

    void SpatialMaxPoolWithArgMaxHelper(
        OpKernelContext* context, Tensor* output, Tensor* output_arg_max,
        Tensor* input_backprop, const Tensor& tensor_in, const Tensor& out_backprop) {

        EigenIndexMatrixMap out_arg_max_mat(output_arg_max->flat<int64>().data(), m_depth,
                                            m_out_width * m_out_height * m_tensor_in_batch);

        DFXMatrix2d in_mat(m_depth,  m_tensor_in_cols * m_tensor_in_rows * m_tensor_in_batch);
        matrix_init(in_mat, tensor_in);

        DFXMatrix2d out_mat(m_depth, m_out_width * m_out_height * m_tensor_in_batch);
        matrix_init(out_mat, *output);

        DFXMatrix2d in_backprop_mat(m_depth,  m_tensor_in_cols * m_tensor_in_rows * m_tensor_in_batch);
        matrix_init(in_backprop_mat);

        DFXMatrix2d out_backprop_mat(m_depth, m_tensor_in_cols * m_tensor_in_rows * m_tensor_in_batch);
        matrix_init(out_backprop_mat, out_backprop);

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

        const int32 output_image_size = m_out_height * m_out_width * m_depth;
        EigenIndexMatrixMap out_arg_max_shard(
            out_arg_max_mat.data(), 1, m_tensor_in_batch * output_image_size);
        out_arg_max_shard.setConstant(kInvalidMaxPoolingIndex);


        // The following code basically does the following:
        // 1. Flattens the input and output tensors into two dimensional arrays.
        //    tensor_in_as_matrix:
        //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
        //    output_as_matrix:
        //      depth by (out_width * out_height * tensor_in_batch)
        //
        // 2. Walks through the set of columns in the flattened tensor_in_as_matrix,
        //    and updates the corresponding column(s) in output_as_matrix with the
        //    max value.
        auto shard = [this, &in_mat, &out_mat, &out_arg_max_mat, &input_backprop,
                        &output_arg_max, &out_backprop, &in_backprop_mat,
                        &out_backprop_mat](int64 start, int64 limit) {

            for (int64 b = start; b < limit; ++b) {
                for (int h = 0; h < m_tensor_in_rows; ++h) {
                    for (int w = 0; w < m_tensor_in_cols; ++w) {
                        // (h_start, h_end) * (w_start, w_end) is the range that the input
                        // vector projects to.
                        const int hpad = h + m_pad_rows;
                        const int wpad = w + m_pad_cols;
                        const int h_start =
                            (hpad < m_window_rows) ? 0 : (hpad - m_window_rows) / m_row_stride + 1;
                        const int h_end = std::min<int>(hpad / m_row_stride + 1, m_out_height);
                        const int w_start =
                            (wpad < m_window_cols) ? 0 : (wpad - m_window_cols) / m_col_stride + 1;
                        const int w_end = std::min<int>(wpad / m_col_stride + 1, m_out_width);
                        // compute elementwise max
                        const int64 in_index = (b * m_tensor_in_rows + h) * m_tensor_in_cols + w;
                        for (int ph = h_start; ph < h_end; ++ph) {
                            const int64 out_index_base = (b * m_out_height + ph) * m_out_width;
                            for (int pw = w_start; pw < w_end; ++pw) {
                                const int64 out_index = out_index_base + pw;
                                /// NOTES(zhengxq): not using the eigen matrix operation for
                                /// now.
                                for (int d = 0; d < m_depth; ++d) {
                                    const DynFxPoint& input_ref = in_mat.coeffRef(d, in_index);
                                    DynFxPoint& output_ref = out_mat.coeffRef(d, out_index);
                                    int64& out_arg_max_ref = out_arg_max_mat.coeffRef(d, out_index);
                                    if (output_ref < input_ref ||
                                        out_arg_max_ref == kInvalidMaxPoolingIndex) {
                                        output_ref = input_ref;
                                        int64 input_offset = in_index * m_depth + d;
                                        out_arg_max_ref = input_offset;
                                    }

                                }
                            }
                        }
                    }
                }
            }

            if (input_backprop != nullptr) {
                auto out_arg_max_flat = output_arg_max->flat<int64>();

                DFXMatrix2dMap input_backprop_flat(in_backprop_mat.data(), 1,
                                              in_backprop_mat.rows() * in_backprop_mat.cols());

                DFXMatrix2dMap out_backprop_flat(out_backprop_mat.data(), 1,
                                               out_backprop_mat.rows() * out_backprop_mat.cols());

                // Initialize output to 0.
                const int64 in_size = m_tensor_in_rows * m_tensor_in_cols * m_depth;
                const int64 in_start = start * in_size;
                const int64 in_end = limit * in_size;

                // Backpropagate.
                const int out_size = m_out_height * m_out_width * m_depth;
                const int out_start = start * out_size;
                const int out_end = limit * out_size;
                for (int index = out_start; index < out_end; ++index) {
                    int input_backprop_index = out_arg_max_flat(index);
                    // Although this check is in the inner loop, it is worth its value
                    // so we don't end up with memory corruptions. Our benchmark shows that
                    // the performance impact is quite small
                    //CHECK(input_backprop_index >= in_start && input_backprop_index < in_end)
                    FastBoundsCheck(input_backprop_index - in_start, in_end - in_start);
                    input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
                }
            }
        };

//   const DeviceBase::CpuWorkerThreads& worker_threads =
//       *(context->device()->tensorflow_cpu_worker_threads());
//   const int64 shard_cost = m_tensor_in_rows * m_tensor_in_cols *
//                            m_depth * m_window_rows *
//                            m_window_cols;
//   Shard(worker_threads.num_threads, worker_threads.workers,
//         m_tensor_in_batch, shard_cost, shard);

    // Call sharding function directly !!!
    shard(0, m_tensor_in_batch);
    matrix_out(input_backprop, in_backprop_mat);
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

bool WaveMaxPool2DGradDfxOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveMaxPoolGradDfx").Device(DEVICE_CPU), WaveMaxPool2DGradDfxOp);
