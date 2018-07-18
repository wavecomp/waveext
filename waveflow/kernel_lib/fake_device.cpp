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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/allocator.h"

#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
// #include "tensorflow/core/common_runtime/scoped_allocator.h"
// #include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"

#include <assert.h>
#include <vector>


using namespace tensorflow;


namespace tensorflow {

    TF_EXPORT const char* const DEVICE_DPU = "DPU";
};

// Dummy
class ScopedAllocatorMgr;

namespace tensorflow {

    // CPU device implementation.
    class DPUDevice : public LocalDevice {
    public:
        DPUDevice(const SessionOptions& options, const string& name,
                         Bytes memory_limit, const DeviceLocality& locality,
                         Allocator* allocator)
        : LocalDevice(options, Device::BuildDeviceAttributes(
            name, DEVICE_DPU, memory_limit, locality)),
            allocator_(allocator) {}
              // scoped_allocator_mgr_(new ScopedAllocatorMgr(name)) {}
        ~DPUDevice() override {}

        void Compute(OpKernel* op_kernel, OpKernelContext* context) override
        {
            // When TraceMe profiling is off (which is the default), the
            // following TraceMe constructor is simply a conditional test of
            // false value. Measurements show that its overhead is negligible.
            port::Tracing::TraceMe trace_me(op_kernel->name(), op_kernel->type_string(),
                                            op_kernel->IsExpensive());
            if (port::Tracing::IsActive()) {
                // TODO(pbar) We really need a useful identifier of the graph node.
                const uint64 id = Hash64(op_kernel->name());
                port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                                    id);
                op_kernel->Compute(context);
            } else {
                op_kernel->Compute(context);
            }
        }
        
        Allocator* GetAllocator(AllocatorAttributes attr) override {
            return allocator_;
        }
        
        /*
        Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                      int64 step_id) override {
            return nullptr;
        }
        ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
            assert(0);
            return NULL;
        }
        */
        
        Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override
        {
            if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
                Tensor parsed(tensor_proto.dtype());
                /* FIXME DPU_ALLOCATOR */
                if (parsed.FromProto(cpu_allocator(), tensor_proto)) {
                    *tensor = std::move(parsed);
                    return Status::OK();
                }
            }
            return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                        ProtoDebugString(tensor_proto));
        }

        Status Sync() override { return Status::OK(); }

    private:
        Allocator* allocator_;  // Not owned
        // std::unique_ptr<ScopedAllocatorMgr> scoped_allocator_mgr_;
    };
};



namespace tensorflow {

    class DPUDeviceFactory : public DeviceFactory {
    public:
        Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                             std::vector<Device*>* devices) override {
          // TODO(zhifengc/tucker): Figure out the number of available CPUs
          // and/or NUMA configuration.
          int n = 1;
          auto iter = options.config.device_count().find("DPU");
          if (iter != options.config.device_count().end()) {
            n = iter->second;
          }
          for (int i = 0; i < n; i++) {
            string name = strings::StrCat(name_prefix, "/device:DPU:", i);
            devices->push_back(new DPUDevice(
                    options, name, Bytes(256 << 20), DeviceLocality(), cpu_allocator()));
          }

          return Status::OK();
        }
    };

    // REGISTER_LOCAL_DEVICE_FACTORY("DPU", DPUDeviceFactory, 60);

}  // namespace tensorflow
