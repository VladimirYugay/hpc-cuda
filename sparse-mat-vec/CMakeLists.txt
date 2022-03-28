cmake_minimum_required(VERSION 3.5)
project(sparse)

set(CMAKE_CXX_STANDARD 11)

set(SM "sm_60" CACHE STRING "model of Nvidia Streaming Multiprocessor")
set(SM_OPTIONS "sm_20" "sm_30" "sm_50" "sm_60" "sm_70")
set_property(CACHE SM PROPERTY STRINGS ${SM_OPTIONS})


set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)
message(STATUS "Boost version: ${Boost_VERSION}")


find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -std=c++11;
        -arch=${SM};
        -Xptxas -v;
        --compiler-options -fPIC;
        -Wno-deprecated-gpu-targets
        -D_FORCE_INLINES)

cuda_add_library(core src/util.cpp
                      src/aux.cpp
                      src/external/mmio.c
                      src/solvers/pagerank/pr.cu
                      src/solvers/heat_transfer/ht_driver.cpp
                      src/solvers/heat_transfer/ht_cusparse.cu
                      src/solvers/heat_transfer/ht_ellpack.cu
                      src/solvers/heat_transfer/ht_band.cu
                      src/solvers/kernels/kernels.cu)

target_include_directories(core PUBLIC src
                                       src/solvers
                                       src/external
                                       ${CUDA_INCLUDE_DIRS})

target_link_libraries(core ${CUDA_LIBRARIES}
                            Boost::program_options
                            Boost::filesystem
                            ${CUDA_cusparse_LIBRARY})
target_compile_options(core PRIVATE -Wno-deprecated-gpu-targets -fPIC)
cuda_add_cublas_to_target(core ${CMAKE_PROJECT_NAME})


cuda_add_executable(${CMAKE_PROJECT_NAME} src/main.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} core)


# Optional Sections
set(WITH_TEST OFF)
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    message(STATUS "PkgConfig Found")
    pkg_check_modules(GMock QUIET gmock)
    if (GMock_FOUND)
        message(STATUS "GMock Found")
        set(WITH_TEST ON)
    endif()
endif()


if (WITH_TEST)
    message(STATUS "compiling with tests...")
    cuda_add_executable(tests tests/main.cpp
                              tests/test_aux.cpp
                              tests/host_tests.cpp
                              tests/gpu_tests.cu)
    target_link_libraries(tests core ${GMock_LIBRARIES} ${CUDA_LIBRARIES})
    target_include_directories(tests PRIVATE src ${GMock_LIBRARIES} ${CUDA_INCLUDE_DIRS})

    enable_testing()
    add_test(NAME all_tests COMMAND tests)
    configure_file(tests/test_matrix.mtx ${CMAKE_CURRENT_BINARY_DIR}/test_matrix.mtx COPYONLY)
else()
    message(STATUS "compiling without tests. Google test/mock framework is not found")
endif()



find_package(pybind11 QUIET)
if (pybind11_FOUND)
    message(STATUS "Compiling with PYBINDING")
    pybind11_add_module(heat_transfer binding/interface.cpp)
    target_link_libraries(heat_transfer PRIVATE core)

    configure_file(binding/ht_runner.py ${CMAKE_CURRENT_BINARY_DIR}/ht_runner.py COPYONLY)
else()
    message(STATUS "Compiling without PYBINDING")
endif()
