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

add_executable(
    sgemm_stage72_tiled
    src/sgemm_stage72_tiled.cu
)

target_link_libraries(
    sgemm_stage72_tiled
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    sgemm_stage72_tiled
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(
    sgemm_stage73_register1d
    src/sgemm_stage73_register1d.cu
)

target_link_libraries(
    sgemm_stage73_register1d
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    sgemm_stage73_register1d
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(
    sgemm_stage74_register2d
    src/sgemm_stage74_register2d.cu
)

target_link_libraries(
    sgemm_stage74_register2d
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    sgemm_stage74_register2d
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(
    sgemm_stage75_vectorized
    src/sgemm_stage75_vectorized.cu
)

target_link_libraries(
    sgemm_stage75_vectorized
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    sgemm_stage75_vectorized
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(
    sgemm_stage76_double_buffer
    src/sgemm_stage76_double_buffer.cu
)

target_link_libraries(
    sgemm_stage76_double_buffer
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    sgemm_stage76_double_buffer
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(
    sgemm_stage77_dispatch
    src/sgemm_stage77_dispatch.cu
)

target_link_libraries(
    sgemm_stage77_dispatch
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    sgemm_stage77_dispatch
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# ============================================================
# Stage 7.8: Reusable SGEMM static library
# ============================================================

add_library(
    sgemm_dispatch
    STATIC
    src/sgemm_dispatch.cu
)

target_include_directories(
    sgemm_dispatch
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    sgemm_dispatch
    PUBLIC
    CUDA::cudart
)

target_compile_options(
    sgemm_dispatch
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

set_target_properties(
    sgemm_dispatch
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

add_executable(
    sgemm_stage78_library_smoke
    tests/sgemm_stage78_library_smoke.cu
)

target_link_libraries(
    sgemm_stage78_library_smoke
    PRIVATE
    sgemm_dispatch
    CUDA::cudart
)

target_compile_options(
    sgemm_stage78_library_smoke
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

include(GNUInstallDirs)

install(
    TARGETS sgemm_dispatch
    EXPORT sgemm_dispatchTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    FILES
    include/sgemm_dispatch.h
    include/cuda_check.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT sgemm_dispatchTargets
    FILE sgemm_dispatchTargets.cmake
    NAMESPACE cuda_sgemm::
    DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/sgemm_dispatch
)


# ============================================================
# Stage 7.9: CMake package configuration
# ============================================================

include(CMakePackageConfigHelpers)

set(
    SGEMM_DISPATCH_PACKAGE_VERSION
    "0.1.0"
)

set(
    SGEMM_DISPATCH_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/sgemm_dispatch"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sgemm_dispatchConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/sgemm_dispatchConfig.cmake"
    INSTALL_DESTINATION
    "${SGEMM_DISPATCH_CMAKE_INSTALL_DIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sgemm_dispatchConfigVersion.cmake"
    VERSION
    "${SGEMM_DISPATCH_PACKAGE_VERSION}"
    COMPATIBILITY
    SameMajorVersion
)

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sgemm_dispatchConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sgemm_dispatchConfigVersion.cmake"
    DESTINATION
    "${SGEMM_DISPATCH_CMAKE_INSTALL_DIR}"
)


# ============================================================
# Stage 7.10: CTest integration
# ============================================================

include(CTest)

option(
    SGEMM_ENABLE_GPU_TESTS
    "Enable CTest cases that require an NVIDIA GPU"
    OFF
)

if(BUILD_TESTING)
    if(CMAKE_CONFIGURATION_TYPES)
        set(
            STAGE710_TEST_CONFIG
            "$<CONFIG>"
        )
    elseif(CMAKE_BUILD_TYPE)
        set(
            STAGE710_TEST_CONFIG
            "${CMAKE_BUILD_TYPE}"
        )
    else()
        set(
            STAGE710_TEST_CONFIG
            "Release"
        )
    endif()

    set(
        STAGE710_CUDA_HOST_COMPILER
        "${CMAKE_CUDA_HOST_COMPILER}"
    )

    if(NOT STAGE710_CUDA_HOST_COMPILER)
        set(
            STAGE710_CUDA_HOST_COMPILER
            "${CMAKE_CXX_COMPILER}"
        )
    endif()

    add_test(
        NAME
        stage710_package_consumer_build

        COMMAND
        "${CMAKE_COMMAND}"
        "-DSTAGE710_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        "-DSTAGE710_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DSTAGE710_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DSTAGE710_CUDA_HOST_COMPILER=${STAGE710_CUDA_HOST_COMPILER}"
        "-DSTAGE710_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
        "-DSTAGE710_CONFIG=${STAGE710_TEST_CONFIG}"
        "-DSTAGE710_GENERATOR=${CMAKE_GENERATOR}"
        -P
        "${CMAKE_SOURCE_DIR}/cmake/stage710_package_ci_test.cmake"
    )

    set_tests_properties(
        stage710_package_consumer_build
        PROPERTIES
        LABELS
        "ci;package;compile-only"
        TIMEOUT
        600
        PASS_REGULAR_EXPRESSION
        "CI_PACKAGE_TEST_PASS = true"
    )

    if(SGEMM_ENABLE_GPU_TESTS)
        add_test(
            NAME
            stage710_gpu_float4

            COMMAND
            $<TARGET_FILE:sgemm_stage78_library_smoke>
            127
            132
            116
            1
            aligned
        )

        add_test(
            NAME
            stage710_gpu_dimension_fallback

            COMMAND
            $<TARGET_FILE:sgemm_stage78_library_smoke>
            127
            131
            113
            1
            aligned
        )

        add_test(
            NAME
            stage710_gpu_pointer_fallback

            COMMAND
            $<TARGET_FILE:sgemm_stage78_library_smoke>
            127
            132
            116
            1
            misalign-a
        )

        set_tests_properties(
            stage710_gpu_float4
            stage710_gpu_dimension_fallback
            stage710_gpu_pointer_fallback
            PROPERTIES
            LABELS
            "gpu;runtime"
            RESOURCE_LOCK
            "nvidia_gpu"
            TIMEOUT
            120
            PASS_REGULAR_EXPRESSION
            "Mismatches = 0"
        )
    endif()
endif()
