# Copyright (c) 2010-2018 Wave Computing, Inc. and its applicable licensors.   
# All rights reserved; provided, that any files identified as open source shall
# be governed by the specific open source license(s) applicable to such files. 
#
# For any files associated with distributions under the Apache 2.0 license, 
# full attribution to The Apache Software Foundation is given via the license 
# below.
#
# PURPOSE
#      Common import module for all Wave custom ops.
# 
# Author          : Ken Shiring
# Created On      : 02/28/2018
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


file(GLOB CPP_SOURCES "*.cpp")

set(CPU_SOURCES ${CPP_SOURCES})

add_library(wf_kernel_lib SHARED ${CPU_SOURCES})

# if (CMAKE_BUILD_TYPE MATCHES Debug)
# 	add_compile_options("-fsanitize=address")
# 	target_link_libraries(wf_kernel_lib "-lasan")
# endif ()


include_directories(${TF_INCLUDE})
include_directories("${TF_INCLUDE}/external/nsync/public")
target_link_libraries(wf_kernel_lib "${TF_LIBDIR}/libtensorflow_framework.so")
