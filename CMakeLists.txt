cmake_minimum_required(VERSION 3.20)
project(yuuka LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YUUKA_BUILD_TESTS "Build native tests" ON)
option(YUUKA_ENABLE_HIP "Build the HIP backend" ON)

if(YUUKA_ENABLE_HIP)
    if(DEFINED ENV{ROCM_PATH})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
    endif()
    file(GLOB _yuuka_rocm_hip_dirs "/opt/rocm/*/lib/cmake/hip")
    list(APPEND CMAKE_PREFIX_PATH ${_yuuka_rocm_hip_dirs})
    find_package(hip REQUIRED)
    enable_language(HIP)
    set_source_files_properties(backend/hip/bridge/bridge.cpp PROPERTIES LANGUAGE HIP)
    add_library(yuuka_hip SHARED
        backend/hip/bridge/bridge.cpp
        kernels/hip/gemv/gemv_quant.hip
        kernels/hip/ops/elementwise.hip
    )
    target_include_directories(yuuka_hip PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/hip/bridge
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/hip
    )
    target_link_libraries(yuuka_hip PRIVATE hip::device)
    target_compile_options(yuuka_hip PRIVATE -Wall -Wextra -Werror)
    set_target_properties(yuuka_hip PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    install(TARGETS yuuka_hip LIBRARY DESTINATION lib)

    if(YUUKA_BUILD_TESTS)
        enable_testing()
        add_executable(yuuka_hip_abi_test tests/hip/abi_smoke.cpp)
        target_link_libraries(yuuka_hip_abi_test PRIVATE yuuka_hip)
        add_test(NAME hip_abi_smoke COMMAND yuuka_hip_abi_test)
        set_tests_properties(hip_abi_smoke PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
