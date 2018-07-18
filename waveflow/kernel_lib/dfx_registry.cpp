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
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <unordered_map>

#include "dfx_registry.h"


// #define DFX_REGISTRY_DEBUG 1

// The registry is a global singleton.
static DynFxPointRegistry* g_registry = NULL;

// Singleton generator.
DynFxPointRegistry* DynFxPointRegistry::get_registry()
{
    if (!g_registry) {
        g_registry = new DynFxPointRegistry();
    }
    return g_registry;
}


void DynFxPointRegistry::register_dfx(const std::string& n, tensorflow::Tensor* t)
{
    // Ensure no overwrites. Duplicates are a programming error.
    // assert(m_tensor_bp_map.find(n) == m_tensor_bp_map.end());
    assert(n.size() > 0);
    assert(t);
    int wtf = t->dims();
    assert(t->dims() == 1);
    assert(t->dim_size(0) == 2);
    assert(DataTypeIsInteger(t->dtype()));
#ifdef DFX_REGISTRY_DEBUG
    printf("Adding tensor [0x%16x](%s,%s) to the registry.\n", (unsigned long)t,
           n.c_str(), 
           t->DebugString().c_str());
#endif
    m_tensor_bp_map[n] = t;
}


tensorflow::Tensor* DynFxPointRegistry::find_dfx(const std::string& n) const
{
    auto e = m_tensor_bp_map.find(n);
    if (e == m_tensor_bp_map.end()) {
        return NULL;
    }
    tensorflow::Tensor* t = e->second;
#ifdef DFX_REGISTRY_DEBUG
    printf("Found tensor [0x%16x](%s,%s) in the registry.\n", (unsigned long)t,
           n.c_str(), 
           t->DebugString().c_str());
#endif
    return t;
}
