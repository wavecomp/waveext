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


REGISTER_OP("WaveAvgPoolGradDfx")
    .Input("orig_input_shape: int32")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });



// The operation to compute AvgPool gradients.
// It takes two inputs:
//   - The original input tensor shape
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
class WaveAvgPoolingGradDfxOp : public OpKernel {
public:
    typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;

    typedef Eigen::Matrix<DynFxPoint, Eigen::Dynamic, Eigen::Dynamic> DFXMatrix2d;

    explicit WaveAvgPoolingGradDfxOp(OpKernelConstruction* context) : OpKernel(context) {
        string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &m_data_format),
                    errors::InvalidArgument("Invalid data format"));
        OP_REQUIRES(
            context, m_data_format == FORMAT_NHWC,
            errors::InvalidArgument("Default AvgPoolingGradOp only supports NHWC ",
                                    "on device type ",
                                    DeviceTypeString(context->device_type())));
        OP_REQUIRES_OK(context, context->GetAttr("ksize", &m_ksize));
        OP_REQUIRES(context, m_ksize.size() == 4,
                    errors::InvalidArgument("Sliding window ksize field must "
                                            "specify 4 dimensions"));
        OP_REQUIRES_OK(context, context->GetAttr("strides", &m_stride));
        OP_REQUIRES(context, m_stride.size() == 4,
                    errors::InvalidArgument("Sliding window strides field must "
                                            "specify 4 dimensions"));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &m_padding));
        OP_REQUIRES(context, m_ksize[0] == 1 && m_stride[0] == 1,
                    errors::Unimplemented(
                        "Pooling is not yet supported on the batch dimension."));


        if (m_show_banner) {
                printf("WaveAvgPoolingGradDfxOp() init\n");
                m_show_banner = false;
        }
    }

    Status GetBroadcastSize(const int index, const int in_size, const int ksize,
                        const int stride, const int pad_size, int* bindex,
                        int* bsize) {
        // Cannot have index beyond the input size.
        if (index * stride > in_size) {
            return errors::InvalidArgument(
                "index * stride must be less than or equal to input size");
        }
        *bindex = index * stride;
        *bsize = ksize;
        if (*bindex < pad_size) {
            // If the current index is in the padding area, start broadcast  from index
            // 0 with broadcast size reduced by padding size.
            *bsize = ksize + *bindex - pad_size;
            *bindex = 0;
        } else {
            // Otherwise, start broadcast from current index reduced by padding size.
            *bindex -= pad_size;
        }
        if (*bindex + ksize > in_size) {
            *bsize = std::min((in_size - *bindex), ksize);
        }
        return Status::OK();
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

    void Compute(OpKernelContext* context) override {
        const Tensor& tensor_in_shape = context->input(0);
        const Tensor& out_backprop = context->input(1);
        // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
        OP_REQUIRES(
            context,
            tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
            errors::InvalidArgument("out_backprop must be 1-dimensional and 4 "
                                    "elements"));
        // For avgpooling, out_backprop should have 4 dimensions.
        OP_REQUIRES(context, out_backprop.dims() == 4,
                    errors::InvalidArgument("out_backprop must be 4-dimensional"));
        const int64 out_backprop_batch = out_backprop.dim_size(0);
        const int64 out_backprop_rows = out_backprop.dim_size(1);
        const int64 out_backprop_cols = out_backprop.dim_size(2);
        const int64 out_backprop_depth = out_backprop.dim_size(3);

        TensorShape output_shape;
        auto shape_vec = tensor_in_shape.vec<int32>();
        for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
            output_shape.AddDim(shape_vec(i));
        }
        const int64 in_rows = output_shape.dim_size(1);
        const int64 in_cols = output_shape.dim_size(2);

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
       // output->flat<float>().setZero();

        const int window_rows = m_ksize[1];
        const int window_cols = m_ksize[2];
        const int depth_window = m_ksize[3];

        const int row_stride = m_stride[1];
        const int col_stride = m_stride[2];

        // We (will) use different code for spatial pooling and
        // non-spatial pooling.
        //
        // Spatial pooling is when depth_window = 1
        OP_REQUIRES(context, depth_window == 1,
                    errors::Unimplemented("Non-spatial pooling is not "
                                        "yet supported. Volunteers? :)"));

        int64 out_height, out_width, pad_rows, pad_cols;
        OP_REQUIRES_OK(context,
                    GetWindowedOutputSize(in_rows, window_rows, row_stride,
                                            m_padding, &out_height, &pad_rows));
        OP_REQUIRES_OK(context,
                    GetWindowedOutputSize(in_cols, window_cols, col_stride,
                                            m_padding, &out_width, &pad_cols));

        // const float* out_backprop_ptr = out_backprop.flat<float>().data();
        // float* input_backprop_ptr = output->flat<float>().data();

        DFXMatrix2d out_backprop_mat(1,  out_backprop_batch * out_backprop_rows *
                                        out_backprop_cols * out_backprop_depth);
        matrix_init(out_backprop_mat, out_backprop);

        DFXMatrix2d input_backprop_mat( 1, output_shape.dim_size(0) * output_shape.dim_size(1) *
                                          output_shape.dim_size(2) * output_shape.dim_size(3));
       // matrix_init(input_backprop_mat, *output);
        matrix_init(input_backprop_mat);

        //  const int32 output_image_size = m_out_height * m_out_width * m_depth;
        // EigenIndexMatrixMap out_arg_max_shard(
        //     out_arg_max_mat.data(), 1, m_tensor_in_batch * output_image_size);
        // out_arg_max_shard.setConstant(kInvalidMaxPoolingIndex);

        auto shard = [this, context, &out_backprop_mat, &input_backprop_mat,
                    out_backprop_rows, out_backprop_cols, out_backprop_depth,
                    in_rows, in_cols, window_rows, window_cols, row_stride,
                    col_stride, pad_rows, pad_cols](int64 start, int64 limit) {
            for (int64 b = start; b < limit; ++b) {
                for (int64 r = 0; r < out_backprop_rows; ++r) {
                    // Calculates row broadcast size.  For SAME padding, current
                    // index could be in the padding area, and r*row_stride +
                    // window_rows could be beyond the input tensor's boundary. In
                    // such cases, change the starting index and reduce the
                    // broadcast size.
                    int rindex, rsize;
                    OP_REQUIRES_OK(context,
                                    GetBroadcastSize(r, in_rows, window_rows, row_stride,
                                                    pad_rows, &rindex, &rsize));
                    for (int64 c = 0; c < out_backprop_cols; ++c) {
                        // Calculates col broadcast size.  For SAME padding, current
                        // index could be in the padding area, and c*col_stride +
                        // window_cols could be beyond the input tensor's boundary. In
                        // such cases, change the starting index and reduce the
                        // broadcast size.
                        int cindex, csize;
                        OP_REQUIRES_OK(context,
                                    GetBroadcastSize(c, in_cols, window_cols, col_stride,
                                                        pad_cols, &cindex, &csize));

                        //float divide_coeff(1.0 / (rsize * csize));

                        // TODO: This should be implemented differently
                        DynFxPoint divide_coeff = DynFxPoint((float)1.0 / (rsize * csize));

                        int64 output_index =
                            (b * out_backprop_rows + r) * out_backprop_cols + c;
                        for (int64 r_dst = rindex; r_dst < rindex + rsize; ++r_dst) {
                            for (int64 c_dst = cindex; c_dst < cindex + csize; ++c_dst) {
                                int64 input_index = (b * in_rows + r_dst) * in_cols + c_dst;
                                int64 output_offset = output_index * out_backprop_depth;
                                int64 input_offset = input_index * out_backprop_depth;
                                for (int64 d = 0; d < out_backprop_depth; ++d) {
                                    input_backprop_mat(0,input_offset) += out_backprop_mat(0,output_offset) * divide_coeff;
                                    ++output_offset;
                                    ++input_offset;
                                }
                            }
                        }
                    }
                }
            }
        };

        // const DeviceBase::CpuWorkerThreads& worker_threads =
        //     *(context->device()->tensorflow_cpu_worker_threads());
        // const int64 shard_cost =
        //     window_rows * window_cols * depth_window * in_rows * in_rows * in_cols;
        // Shard(worker_threads.num_threads, worker_threads.workers,
        //       out_backprop_batch, shard_cost, shard);

        shard(0, out_backprop_batch);
        matrix_out(output, input_backprop_mat);
  }

private:
    std::vector<int32> m_ksize;
    std::vector<int32> m_stride;
    Padding m_padding;
    TensorFormat m_data_format;

private:
    static bool m_show_banner;
};

bool WaveAvgPoolingGradDfxOp::m_show_banner = true;

REGISTER_KERNEL_BUILDER(Name("WaveAvgPoolGradDfx").Device(DEVICE_CPU), WaveAvgPoolingGradDfxOp);
