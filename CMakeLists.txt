cmake_minimum_required(VERSION 3.24)
project(paged_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Test for the CPU reference attention
add_executable(test_attention_cpu tests/test_attention_cpu.cpp src/attention_cpu.cpp)
target_include_directories(test_attention_cpu PRIVATE src)

# Test for the GPU attention, checked against the CPU one
add_executable(test_attention_gpu tests/test_attention_gpu.cu src/attention_gpu.cu src/attention_cpu.cpp)
target_include_directories(test_attention_gpu PRIVATE src)