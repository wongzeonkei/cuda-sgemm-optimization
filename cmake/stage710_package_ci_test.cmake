cmake_minimum_required(VERSION 3.18)

set(required_variables
    STAGE710_SOURCE_DIR
    STAGE710_BINARY_DIR
    STAGE710_CXX_COMPILER
    STAGE710_CUDA_HOST_COMPILER
    STAGE710_CUDA_ARCHITECTURES
    STAGE710_CONFIG
    STAGE710_GENERATOR
)

foreach(variable IN LISTS required_variables)
    if(
        NOT DEFINED ${variable}
        OR "${${variable}}" STREQUAL ""
    )
        message(
            FATAL_ERROR
            "Missing required variable: ${variable}"
        )
    endif()
endforeach()


function(run_checked description)
    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE command_status
        OUTPUT_VARIABLE command_stdout
        ERROR_VARIABLE command_stderr
    )

    message(
        STATUS
        "===== ${description}: stdout =====\n"
        "${command_stdout}"
    )

    if(NOT "${command_stderr}" STREQUAL "")
        message(
            STATUS
            "===== ${description}: stderr =====\n"
            "${command_stderr}"
        )
    endif()

    if(NOT command_status EQUAL 0)
        message(
            FATAL_ERROR
            "${description} failed with status "
            "${command_status}"
        )
    endif()
endfunction()


set(
    test_root
    "${STAGE710_BINARY_DIR}/stage710_ctest_package"
)

set(
    original_prefix
    "${test_root}/install"
)

set(
    original_consumer_build
    "${test_root}/consumer_original"
)

set(
    relocated_prefix
    "${test_root}/relocated"
)

set(
    relocated_consumer_build
    "${test_root}/consumer_relocated"
)

file(
    REMOVE_RECURSE
    "${test_root}"
)

file(
    MAKE_DIRECTORY
    "${test_root}"
)


run_checked(
    "Install SGEMM package"
    "${CMAKE_COMMAND}"
    --install
    "${STAGE710_BINARY_DIR}"
    --prefix
    "${original_prefix}"
    --config
    "${STAGE710_CONFIG}"
)


set(expected_installed_files
    "include/sgemm_dispatch.h"
    "include/cuda_check.h"
    "lib/libsgemm_dispatch.a"
    "lib/cmake/sgemm_dispatch/sgemm_dispatchConfig.cmake"
    "lib/cmake/sgemm_dispatch/sgemm_dispatchConfigVersion.cmake"
    "lib/cmake/sgemm_dispatch/sgemm_dispatchTargets.cmake"
)

foreach(relative_path IN LISTS expected_installed_files)
    if(
        NOT EXISTS
        "${original_prefix}/${relative_path}"
    )
        message(
            FATAL_ERROR
            "Missing installed file: "
            "${original_prefix}/${relative_path}"
        )
    endif()
endforeach()


run_checked(
    "Configure original package Consumer"
    "${CMAKE_COMMAND}"
    -S
    "${STAGE710_SOURCE_DIR}/tests/stage79_cmake_consumer"
    -B
    "${original_consumer_build}"
    -G
    "${STAGE710_GENERATOR}"
    "-DCMAKE_BUILD_TYPE=${STAGE710_CONFIG}"
    "-DCMAKE_PREFIX_PATH=${original_prefix}"
    "-DCMAKE_CXX_COMPILER=${STAGE710_CXX_COMPILER}"
    "-DCMAKE_CUDA_HOST_COMPILER=${STAGE710_CUDA_HOST_COMPILER}"
    "-DCMAKE_CUDA_ARCHITECTURES=${STAGE710_CUDA_ARCHITECTURES}"
)


run_checked(
    "Build original package Consumer"
    "${CMAKE_COMMAND}"
    --build
    "${original_consumer_build}"
    --config
    "${STAGE710_CONFIG}"
    --parallel
    "2"
)


file(
    COPY
    "${original_prefix}/"
    DESTINATION
    "${relocated_prefix}"
)


run_checked(
    "Configure relocated package Consumer"
    "${CMAKE_COMMAND}"
    -S
    "${STAGE710_SOURCE_DIR}/tests/stage79_cmake_consumer"
    -B
    "${relocated_consumer_build}"
    -G
    "${STAGE710_GENERATOR}"
    "-DCMAKE_BUILD_TYPE=${STAGE710_CONFIG}"
    "-DCMAKE_PREFIX_PATH=${relocated_prefix}"
    "-DCMAKE_CXX_COMPILER=${STAGE710_CXX_COMPILER}"
    "-DCMAKE_CUDA_HOST_COMPILER=${STAGE710_CUDA_HOST_COMPILER}"
    "-DCMAKE_CUDA_ARCHITECTURES=${STAGE710_CUDA_ARCHITECTURES}"
)


run_checked(
    "Build relocated package Consumer"
    "${CMAKE_COMMAND}"
    --build
    "${relocated_consumer_build}"
    --config
    "${STAGE710_CONFIG}"
    --parallel
    "2"
)


set(link_files)

file(
    GLOB_RECURSE
    response_files
    "${relocated_consumer_build}/*linkLibs.rsp"
)

file(
    GLOB_RECURSE
    link_text_files
    "${relocated_consumer_build}/*link.txt"
)

list(
    APPEND
    link_files
    ${response_files}
    ${link_text_files}
)

if(
    EXISTS
    "${relocated_consumer_build}/build.ninja"
)
    list(
        APPEND
        link_files
        "${relocated_consumer_build}/build.ninja"
    )
endif()

if(NOT link_files)
    message(
        FATAL_ERROR
        "No Consumer link description file was found."
    )
endif()


set(all_link_text "")

foreach(link_file IN LISTS link_files)
    file(
        READ
        "${link_file}"
        current_link_text
    )

    string(
        APPEND
        all_link_text
        "\n"
        "${current_link_text}"
    )
endforeach()


set(
    original_library
    "${original_prefix}/lib/libsgemm_dispatch.a"
)

set(
    relocated_library
    "${relocated_prefix}/lib/libsgemm_dispatch.a"
)

string(
    FIND
    "${all_link_text}"
    "${relocated_library}"
    relocated_library_index
)

if(relocated_library_index EQUAL -1)
    message(
        FATAL_ERROR
        "Relocated Consumer does not link the relocated library: "
        "${relocated_library}"
    )
endif()

string(
    FIND
    "${all_link_text}"
    "${original_library}"
    original_library_index
)

if(NOT original_library_index EQUAL -1)
    message(
        FATAL_ERROR
        "Relocated Consumer still references original library: "
        "${original_library}"
    )
endif()


message(
    STATUS
    "Original package Consumer build passed."
)

message(
    STATUS
    "Relocated package Consumer build passed."
)

message(
    STATUS
    "Relocated library: ${relocated_library}"
)

message(
    STATUS
    "CI_PACKAGE_TEST_PASS = true"
)
