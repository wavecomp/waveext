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
#pragma once

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "Eigen/Core"

#include <stdlib.h>
#include <string>
#include <vector>

#include "dyn_fx_pt.h"


using namespace tensorflow;

/* Base class for Dynamic Fixed Point operators. This will take care of parsing
 * any relevant attributes and keeping some appropriate binary point.
 * 
 * This class is desined to make it easy for a custom operator to get the 
 * Dynamic Fixed Point for a given I/O. I/O must be referenced by index, since the TF
 * C++ API isn't sophisticated enough to refer to I/O by name. 
 * 
 * Users must pass the name of the BP attributes through to the constructor.
 * The attributes given will be associated with a BP setting from some user-
 * declared BP generator operation. This base class handles the linkage between
 * this consumer operation and the BP generator operation. Any BP I/O without a
 * generator will return an fxbp of uninitialized, so the caller may handle this
 * however they decide.
 */
class WaveDynFxPointOp : public OpKernel {
public:
    typedef std::vector<std::string>    StringVec;
    typedef std::vector<DynFxPoint>     DFXVector;
    typedef Eigen::Matrix<DynFxPoint, Eigen::Dynamic, Eigen::Dynamic>  DFXMatrix2d;

    explicit WaveDynFxPointOp(OpKernelConstruction* ctx, const StringVec& attrs);
    
protected:
    
    fxbp        get_fxbp(bool isInput, int n) const;
    std::string get_vp_key(bool isInput, int n) const;

    void        partial_in(fxbp& dest_bp, DFXVector& m_out, const float* flat_arr);
    void        partial_out(fxbp& out_bp, float* conv_out, const DFXVector& m_out);
    void        partial_in(fxbp& dest_bp, int32_t* m_out, const float* flat_arr, int n);
    void        partial_out(fxbp& out_bp, float* conv_out, const int32_t* m_out, int32_t src_bp, int n);
    void        partial_out(float* conv_out, const int32_t* m_out, int32_t src_bp, int n);
    int32_t     stochastic_rounding(int32_t v, int32_t r);
    
    StringVec   m_iLookup;
    StringVec   m_oLookup;
    bool        m_use_stochastic_round;
    
private:
    
    bool        get_iospec(const std::string& attr, bool& isInput, int& n);
    bool        get_static_bp(const std::string& attr, fxbp& bp) const;
    bool        get_dynamic_bp(const std::string& attr, fxbp& bp) const;
};

    