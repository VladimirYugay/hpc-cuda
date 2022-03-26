cmake_minimum_required(VERSION 3.5)
project(mm)

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
                     -g
                     -G
                     -std=c++11;
                     -arch=${SM};
                     -Xptxas -v;
                     --compiler-options -fPIC;
                     -Wno-deprecated-gpu-targets
                     -D_FORCE_INLINES)

cuda_add_library(core src/driver.cu
                      src/util.cu
                      src/kernels/parallel.cu
                      src/kernels/serial.cu)

target_include_directories(core PUBLIC src ${CUDA_INCLUDE_DIRS})
target_link_libraries(core ${CUDA_LIBRARIES} Boost::program_options)
target_compile_options(core PRIVATE -Wno-deprecated-gpu-targets)
cuda_add_cublas_to_target(core ${CMAKE_PROJECT_NAME})

cuda_add_executable(${CMAKE_PROJECT_NAME} src/main.cu)
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
    add_executable(tests tests/main.cpp tests/tests.cpp)
    target_link_libraries(tests core ${GMock_LIBRARIES})
    target_include_directories(tests PRIVATE src ${GMock_LIBRARIES})

    enable_testing()
    add_test(NAME all_tests COMMAND tests)
else()
    message(STATUS "compiling without tests. Google test/mock framework is not found")
endif()


find_package(pybind11 QUIET)
if (pybind11_FOUND)
    message(STATUS "Compiling with PYBINDING")
    pybind11_add_module(dense_hpc_aa binding/interface.cpp)
    target_link_libraries(dense_hpc_aa PRIVATE core)

    configure_file(binding/runner.py ${CMAKE_CURRENT_BINARY_DIR}/runner.py COPYONLY)
else()
    message(STATUS "Compiling without PYBINDING")
endif()