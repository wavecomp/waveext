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

#include <stdlib.h>

using namespace tensorflow;

REGISTER_OP("WaveConfig")
    .Attr("rounding_mode: string = ''")
    .Attr("config2: string = ''")
    .Attr("config3: string = ''")
    .SetShapeFn(shape_inference::UnknownShape);

class WaveConfigOp : public OpKernel {
public:
    // configuration fields
    static string m_rounding_mode;
    static string m_config2;
    static string m_config3;


    explicit WaveConfigOp(OpKernelConstruction* context)
        : OpKernel(context) {

        string tmp_string;
        OP_REQUIRES_OK(context, context->GetAttr("rounding_mode", &tmp_string));
        if (!tmp_string.empty())
            m_rounding_mode = tmp_string;

        OP_REQUIRES_OK(context, context->GetAttr("config2", &tmp_string));
        if (!tmp_string.empty())
            m_config2 = tmp_string;

        OP_REQUIRES_OK(context, context->GetAttr("config3", &tmp_string));
        if (!tmp_string.empty())
            m_config3 = tmp_string;

#if 0
        // validity check can be done here or in destination operators
        if (!m_rounding_mode.compare("stochastic")) {
            printf("WaveConfigOp: rounding mode is stochastic\n");
        } else if(!m_rounding_mode.compare("convergent")) {
            printf("WaveConfigOp: rounding mode is convergent\n");
        } else {

        }
#endif
        printf("WaveConfigOp: rounding_mode = %s, config2 = %s, config3 = %s\n",
               m_rounding_mode.c_str(), m_config2.c_str(), m_config3.c_str());
    }

    void Compute(OpKernelContext* context) override {}
};

string WaveConfigOp::m_rounding_mode = "";
string WaveConfigOp::m_config2 = "";
string WaveConfigOp::m_config3 = "";

REGISTER_KERNEL_BUILDER(Name("WaveConfig").Device(DEVICE_CPU), WaveConfigOp);