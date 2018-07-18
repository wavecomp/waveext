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

cmake_minimum_required (VERSION 3.5)
project (waveflow)

# if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# 	message(FATAL_ERROR "The C++ compiler must be clang.")
# endif()
# if (NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
# 	message(FATAL_ERROR "The C compiler must be clang.")
# endif()


SET( CMAKE_CXX_STANDARD 11 )
SET( CMAKE_CXX_STANDARD_REQUIRED ON )

# KS: add checks for cuDNN here when ready.

add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")

# These switches must be set carefully. The assumed base machine
# will be SSE 4.2 (for now).
add_compile_options("-fPIC")
add_compile_options("-m64")
add_compile_options("-msse4.2")

if (OPENMP_OPTIMIZATION MATCHES "True")
	find_package(OpenMP)
	if (OPENMP_FOUND)
		add_compile_options("-DOPENMP_OPTIMIZATION")
	    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	endif()
endif ()


if (CMAKE_BUILD_TYPE MATCHES Debug)
	add_compile_options("-fno-omit-frame-pointer")
endif ()

# Query Tensorflow to get the includes and library deps
function (tf_get_env)
	execute_process(COMMAND python3 -W ignore -c "import tensorflow as tf; print(tf.__version__)"
				    OUTPUT_VARIABLE TF_VERSION
				    OUTPUT_STRIP_TRAILING_WHITESPACE)
	set(TF_VERSION ${TF_VERSION} PARENT_SCOPE)
	set(TF_REQ_VERSION "1.8.0")
	if (NOT TF_VERSION VERSION_EQUAL TF_REQ_VERSION)
		message(FATAL_ERROR " Unsupported Tensoflow version ${TF_VERSION} found, ${TF_REQ_VERSION} required.")
	endif ()
	execute_process(COMMAND python3 -W ignore -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
				    OUTPUT_VARIABLE TF_INCLUDE
				    OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(COMMAND python3 -W ignore -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
				    OUTPUT_VARIABLE TF_LIBDIR
				    OUTPUT_STRIP_TRAILING_WHITESPACE)
	set(TF_INCLUDE ${TF_INCLUDE} PARENT_SCOPE)
	set(TF_LIBDIR ${TF_LIBDIR} PARENT_SCOPE)

	# These functions should also be used at some point; for future improvement to make compile flag settings
	# more failproof.
	#
	# execute_process(COMMAND python3 -W ignore -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
	# 			    OUTPUT_VARIABLE TF_LINK_FLAGS
	# 			    OUTPUT_STRIP_TRAILING_WHITESPACE)
	# execute_process(COMMAND python3 -W ignore -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
	# 			    OUTPUT_VARIABLE TF_COMPILE_FLAGS
	# 			    OUTPUT_STRIP_TRAILING_WHITESPACE)

	# message(INFO " CFLAGS: " ${TF_COMPILE_FLAGS})
	# message(INFO " LFLAGS: " ${TF_LINK_FLAGS})

	# add_compile_options(${TF_COMPILE_FLAGS})
	# add_compile_options(${TF_LINK_FLAGS})
endfunction(tf_get_env)

tf_get_env()
# message(INFO " Version: " ${TF_VERSION})
# message(INFO " Includes: " ${TF_INCLUDE})
# message(INFO " Libs: " ${TF_LIBDIR})

add_subdirectory("waveflow/kernel_lib/")


