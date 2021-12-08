cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -g -l png")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(cpu src/cpu.cpp)
target_compile_features(cpu PUBLIC cxx_std_17)
target_link_libraries(cpu PRIVATE PNG::PNG)

add_executable(gpu src/gpu.cpp)
target_compile_features(gpu PUBLIC cxx_std_17)
target_link_libraries(gpu PRIVATE PNG::PNG)



