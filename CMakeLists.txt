cmake_minimum_required(VERSION 3.24)
project(paged_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra)

# Test for the CPU reference attention
add_executable(test_attention_cpu tests/test_attention_cpu.cpp src/attention_cpu.cpp)
target_include_directories(test_attention_cpu PRIVATE src)