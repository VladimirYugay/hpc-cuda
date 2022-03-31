cmake_minimum_required(VERSION 3.5)
project(stencil)

set(CMAKE_CXX_STANDARD 11)

set(SM "sm_86" CACHE STRING "model of Nvidia Streaming Multiprocessor")
set(SM_OPTIONS "sm_50" "sm_60" "sm_70" "sm_86")
set_property(CACHE SM PROPERTY STRINGS ${SM_OPTIONS})


find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                    -std=c++11;
                    -arch=${SM};
                    -Xptxas -v;
                    -O3;
                    -lineinfo;
                    --compiler-options -fPIC;
                    -Wno-deprecated-gpu-targets
                    -D_FORCE_INLINES)


cuda_add_library(core src/solver.cpp
                      src/aux.cpp
                      src/implementations/cpu.cpp
                      src/implementations/gpu.cu
                      src/kernels/kernels.cu)

target_include_directories(core PUBLIC src ${CUDA_INCLUDE_DIRS})
target_compile_options(core PRIVATE -Wno-deprecated-gpu-targets)
target_compile_options(core PUBLIC -fPIC)

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


find_package(pybind11 REQUIRED)
pybind11_add_module(interface src/interface.cpp)
target_link_libraries(interface PRIVATE core)

install(TARGETS interface
        DESTINATION ${CMAKE_SOURCE_DIR}/solver
        PERMISSIONS OWNER_READ
                    OWNER_EXECUTE
                    GROUP_READ
                    GROUP_EXECUTE
                    WORLD_READ
                    WORLD_EXECUTE)


add_executable(profiling profiling/main.cpp)
target_link_libraries(profiling core)


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
                        tests/host_tests.cpp
                        tests/gpu_tests.cu)
    target_link_libraries(tests core ${GMock_LIBRARIES} ${CUDA_LIBRARIES})
    target_include_directories(tests PRIVATE src ${GMock_LIBRARIES} ${CUDA_INCLUDE_DIRS})

    enable_testing()
    add_test(NAME all_tests COMMAND tests)
else()
    message(STATUS "compiling without tests. Google test/mock framework is not found")
endif()
