# $HEADER$
# $LICENSE$


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
