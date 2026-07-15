cmake_minimum_required(VERSION 3.18)

project(
    sgemm_dispatch_consumer
    LANGUAGES
    CXX
    CUDA
)

find_package(
    sgemm_dispatch
    0.1
    CONFIG
    REQUIRED
)

get_target_property(
    SGEMM_TARGET_TYPE
    cuda_sgemm::sgemm_dispatch
    TYPE
)

get_target_property(
    SGEMM_INCLUDE_DIRECTORIES
    cuda_sgemm::sgemm_dispatch
    INTERFACE_INCLUDE_DIRECTORIES
)

get_target_property(
    SGEMM_LINK_LIBRARIES
    cuda_sgemm::sgemm_dispatch
    INTERFACE_LINK_LIBRARIES
)

message(
    STATUS
    "sgemm_dispatch version: ${sgemm_dispatch_VERSION}"
)

message(
    STATUS
    "sgemm_dispatch target type: ${SGEMM_TARGET_TYPE}"
)

message(
    STATUS
    "sgemm_dispatch include directories: ${SGEMM_INCLUDE_DIRECTORIES}"
)

message(
    STATUS
    "sgemm_dispatch link libraries: ${SGEMM_LINK_LIBRARIES}"
)

add_executable(
    sgemm_dispatch_consumer
    main.cu
)

target_link_libraries(
    sgemm_dispatch_consumer
    PRIVATE
    cuda_sgemm::sgemm_dispatch
)

set_target_properties(
    sgemm_dispatch_consumer
    PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
