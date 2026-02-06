CMAKE_MINIMUM_REQUIRED(VERSION 3.22)

set(PRJ cute_learning)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(${PRJ} CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})

message("CMAKE_CURRENT_PATH: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_PREFIX: ${CMAKE_PREFIX_PATH}")

# 全局包含目录
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm-ffi/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/sglang/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS "csrc/*.cpp" "csrc/*.cu")
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "include/*.h" "include/*.hpp" "include/*.cuh")

add_executable(
    ${PRJ}
    ${headers}
    ${srcs}
)

# Debug模式设置
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode enabled")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-G")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g")
endif()