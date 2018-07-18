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
#include "tensorflow/core/framework/tensor.h"


#include <string>
#include <unordered_map>

#include "dyn_fx_pt.h"


class DynFxPointRegistry;


/* This class implemented a key/Tensor registry for ops to use. It's designed
 * to connect ops in a Tensorflow graph together by key name. Technically, it
 * can be used for any Tensor/Name binding. In practice, it's used to associate
 * graph ops which generate a BP based on some criteria, with downstream 
 * consumer ops.
 */
class DynFxPointRegistry
{

public:
    
    void register_dfx(const std::string& n, tensorflow::Tensor* bp);
    tensorflow::Tensor* find_dfx(const std::string& n) const;
    
    // Singleton constructor
    static DynFxPointRegistry* get_registry();
    
private:
    
    typedef std::unordered_map<const std::string, tensorflow::Tensor*, std::hash<std::string>> TensorBPMap;
    
    explicit DynFxPointRegistry() {}
    
    TensorBPMap         m_tensor_bp_map;
};
