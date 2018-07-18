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

#include "gemm_fp.h"


// #define TRACE_ENABLE 1

using namespace tensorflow;

REGISTER_OP("WaveMatMul")
    .Input("a: float")
    .Input("b: float")
    .Output("z: float")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    // .Attr("T: float")
    .SetShapeFn(shape_inference::MatMulShape);

 
class WaveMatMulOp : public OpKernel {
public:
    explicit WaveMatMulOp(OpKernelConstruction* context) 
        : OpKernel(context), m_first_time(true) {
        OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
        OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
        if (m_show_banner) {
            printf("WaveMatMulOp() init\n");
            m_show_banner = false;
        }        
    }
  
    void PrintInstance(const TensorShape& a_shape, const TensorShape& b_shape) {
#ifdef TRACE_ENABLE
        if (m_first_time) {
            const char* attr[2] = {"", "^T "};
            printf("WaveMatMul(%s): (%d,%d)%s x (%d,%d)%s\n", name().c_str(),
                   a_shape.dim_size(0), a_shape.dim_size(1), attr[(int)transpose_a_],
                   b_shape.dim_size(0), b_shape.dim_size(1), attr[(int)transpose_b_]);
            m_first_time = false;
        }
#endif
    }
    
    void Compute(OpKernelContext* context) override {
    
        // some checks to be sure ...
        DCHECK_EQ(2, context->num_inputs());

        const Tensor& tensor_a = context->input(0);    
        const Tensor& tensor_b = context->input(1);

        // check shapes of input and weights
        const TensorShape& a_shape = tensor_a.shape();
        const TensorShape& b_shape = tensor_b.shape();

        // check dimensions
        DCHECK_EQ(a_shape.dims(), 2);
        DCHECK_EQ(b_shape.dims(), 2);

        // create output shape
        TensorShape out_shape(get_output_shape(a_shape, b_shape));
                
        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        // Debug
        PrintInstance(a_shape, b_shape);
        
        auto a_m = tensor_a.matrix<float>();
        auto b_m = tensor_b.matrix<float>();
        auto z_m = output->matrix<float>();
        
        if (!transpose_a_ && !transpose_b_) {
            mm_nn(true, a_m, a_shape.dim_size(0), a_shape.dim_size(1), 
                  b_m, b_shape.dim_size(0), b_shape.dim_size(1), z_m);
        } else if (transpose_a_ && !transpose_b_) {
            mm_tn(true, a_m, a_shape.dim_size(0), a_shape.dim_size(1), 
                  b_m, b_shape.dim_size(0), b_shape.dim_size(1), z_m);
        } else if (!transpose_a_ && transpose_b_) {
            mm_nt(true, a_m, a_shape.dim_size(0), a_shape.dim_size(1), 
                  b_m, b_shape.dim_size(0), b_shape.dim_size(1), z_m);
        } else {
            mm_tt(true, a_m, a_shape.dim_size(0), a_shape.dim_size(1), 
                  b_m, b_shape.dim_size(0), b_shape.dim_size(1), z_m);
        }
    }
private:
    
    bool transpose_a_;
    bool transpose_b_;
    
    TensorShape get_output_shape(const TensorShape& a_shape, const TensorShape& b_shape)
    {
        if (!transpose_a_ && !transpose_b_) {
            DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(0));
            return TensorShape({a_shape.dim_size(0), b_shape.dim_size(1)});
        } else if (transpose_a_ && !transpose_b_) {
            DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(0));
            return TensorShape({a_shape.dim_size(1), b_shape.dim_size(1)});
        } else if (!transpose_a_ && transpose_b_) {
            DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(1));
            return TensorShape({a_shape.dim_size(0), b_shape.dim_size(0)});
        } else {
            DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(1));
            return TensorShape({a_shape.dim_size(1), b_shape.dim_size(0)});
        } 
    }

private:
    static bool m_show_banner;
    bool m_first_time;
};

bool WaveMatMulOp::m_show_banner = true;
// bool WaveMatMulOp::m_first_time = true;
 
REGISTER_KERNEL_BUILDER(Name("WaveMatMul").Device(DEVICE_CPU), WaveMatMulOp);
