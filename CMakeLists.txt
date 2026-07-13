cmake_minimum_required(VERSION 3.18)

project(cuda_sgemm_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3090: Compute Capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(device_info src/device_info.cu)

target_link_libraries(
    device_info
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    device_info
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(
    sgemm_stage71
    src/sgemm_stage71.cu
)

target_link_libraries(
    sgemm_stage71
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    sgemm_stage71
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
