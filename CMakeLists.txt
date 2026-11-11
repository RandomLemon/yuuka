cmake_minimum_required(VERSION 3.20)
project(yuuka LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YUUKA_BUILD_TESTS "Build native tests" ON)
option(YUUKA_BUILD_BENCH "Build native hardware benchmarks" ON)
option(YUUKA_ENABLE_HIP "Build the HIP backend" ON)

if(YUUKA_ENABLE_HIP)
    if(DEFINED ENV{ROCM_PATH})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
    endif()
    file(GLOB _yuuka_rocm_cmake_dirs "/opt/rocm/*/lib/cmake/hip"
         "/opt/rocm/*/lib/cmake/rocblas" "/opt/rocm/*/lib/cmake/rocwmma")
    list(APPEND CMAKE_PREFIX_PATH ${_yuuka_rocm_cmake_dirs})
    find_package(hip REQUIRED)
    find_package(rocblas REQUIRED)
    find_package(rocwmma REQUIRED)
    find_package(OpenMP REQUIRED)
    if(NOT TARGET OpenMP::omp)
        add_library(OpenMP::omp INTERFACE IMPORTED)
        set_target_properties(OpenMP::omp PROPERTIES
            INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)
    endif()
    enable_language(HIP)
    set_source_files_properties(backend/hip/bridge/bridge.cpp PROPERTIES LANGUAGE HIP)
    add_library(yuuka_hip SHARED
        backend/hip/bridge/bridge.cpp
        kernels/hip/attention/attention.hip
        kernels/hip/embedding/embedding.hip
        kernels/hip/gemv/gemv_quant.hip
        kernels/hip/gemm/gemm.hip
        kernels/hip/gemm/wmma_gemm.hip
        kernels/hip/gemm/gemm_blas.hip
        kernels/hip/ops/elementwise.hip
        kernels/hip/ops/norm.hip
        kernels/hip/ops/rope.hip
        kernels/hip/prefill/prefill.hip
        kernels/hip/ssm/ssm.hip
    )
    target_include_directories(yuuka_hip PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/hip/bridge
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/hip
    )
    target_link_libraries(yuuka_hip PRIVATE hip::device roc::rocblas roc::rocwmma)
    target_compile_options(yuuka_hip PRIVATE -Wall -Wextra -Werror)
    set_target_properties(yuuka_hip PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    install(TARGETS yuuka_hip LIBRARY DESTINATION lib)

    if(YUUKA_BUILD_BENCH)
        add_executable(yuuka_hardware_probe bench/hip/hardware_probe.hip)
        target_link_libraries(yuuka_hardware_probe PRIVATE hip::device roc::rocblas roc::rocwmma)

        add_executable(yuuka_gemv_sweep bench/hip/gemv_sweep.cpp)
        target_link_libraries(yuuka_gemv_sweep PRIVATE yuuka_hip)

        add_executable(yuuka_prefill_sweep bench/hip/prefill_sweep.cpp)
        target_link_libraries(yuuka_prefill_sweep PRIVATE yuuka_hip)
    endif()

    if(YUUKA_BUILD_TESTS)
        enable_testing()
        add_executable(yuuka_hip_abi_test tests/hip/abi_smoke.cpp)
        target_link_libraries(yuuka_hip_abi_test PRIVATE yuuka_hip)
        add_test(NAME hip_abi_smoke COMMAND yuuka_hip_abi_test)
        set_tests_properties(hip_abi_smoke PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(yuuka_hip_gemv_test tests/hip/gemv_test.cpp)
        target_link_libraries(yuuka_hip_gemv_test PRIVATE yuuka_hip)
        add_test(NAME hip_gemv_correctness COMMAND yuuka_hip_gemv_test)
        set_tests_properties(hip_gemv_correctness PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(yuuka_hip_ops_test tests/hip/ops_test.cpp)
        target_link_libraries(yuuka_hip_ops_test PRIVATE yuuka_hip)
        add_test(NAME hip_ops_correctness COMMAND yuuka_hip_ops_test)
        set_tests_properties(hip_ops_correctness PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(yuuka_hip_gemm_test tests/hip/gemm_test.cpp)
        target_link_libraries(yuuka_hip_gemm_test PRIVATE yuuka_hip)
        add_test(NAME hip_gemm_correctness COMMAND yuuka_hip_gemm_test)
        set_tests_properties(hip_gemm_correctness PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(yuuka_hip_prefill_test tests/hip/prefill_test.cpp)
        target_link_libraries(yuuka_hip_prefill_test PRIVATE yuuka_hip)
        add_test(NAME hip_prefill_correctness COMMAND yuuka_hip_prefill_test)
        set_tests_properties(hip_prefill_correctness PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(yuuka_hip_paged_test tests/hip/paged_test.cpp)
        target_link_libraries(yuuka_hip_paged_test PRIVATE yuuka_hip)
        add_test(NAME hip_paged_graph_correctness COMMAND yuuka_hip_paged_test)
        set_tests_properties(hip_paged_graph_correctness PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
