cmake_minimum_required(VERSION 3.6)

project(
  FasterDP
  VERSION 1.0
  LANGUAGES CXX C)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if (DEFINED _GLIBCXX_USE_CXX11_ABI)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
endif ()

# Dependencies
set(WITH_PERF_TOOL OFF)
set(ZMQ_BUILD_TESTS OFF)
set(ENABLE_CPACK OFF)
add_subdirectory(libzmq) 
set(CPPZMQ_BUILD_TESTS OFF)
add_subdirectory(cppzmq) 
set(protobuf_BUILD_SHARED_LIBS ON)
set(protobuf_BUILD_TESTS OFF)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(protobuf) 

# set(CMAKE_C_FLAGS "-g -march=native")
# set(CMAKE_CXX_FLAGS "-g -march=native")
set(CMAKE_C_FLAGS "-g -march=broadwell")
set(CMAKE_CXX_FLAGS "-g -march=broadwell")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-g -O3")
set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}")
set(CUDNN_LIBRARY_PATH ${PYTHON_LIB_PATH}/site-packages/torch/lib)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories(${CMAKE_CURRENT_BINARY_DIR}/protobuf)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${PYTHON_LIB_PATH}/site-packages/torch/include)
include_directories(${PYTHON_LIB_PATH}/site-packages/torch/include/torch/csrc/api/include)
include_directories(cppzmq)
include_directories(json/single_include)

add_library(fasterdp_core SHARED 
  src/engine/core.cpp
  src/engine/core_module_load.cpp
  src/engine/core_module_api.cpp
  src/engine/task.cpp
  src/engine/threadpool.cpp
  src/engine/shm_manager.cpp
  src/engine/comm_manager.cpp
  src/engine/telemetry_client.cpp
  src/engine/telemetry_server.cpp
  src/engine/modules/d2h_copy.cpp
  src/engine/modules/cpu_gather.cpp
  src/engine/modules/compress.cpp
  src/engine/modules/grad_exchange.cpp
  src/engine/modules/cpu_optimize.cpp
  src/engine/modules/h2d_copy.cpp
  src/engine/modules/null.cpp
  src/engine/modules/barrier_checker.cpp
  src/optim/sgd.cpp 
  src/optim/adam.cpp
  src/optim/sgd_naive.cpp
  src/compress/topk.cpp
  src/compress/randomk.cpp
  src/compress/thresholdv.cpp
  src/compress/thresholdv16.cpp
  src/misc/hello.cpp
  src/misc/simdxorshift128plus.c
  src/misc/ipaddr.cpp
)

add_subdirectory(src/proto)

add_executable(fasterdp_engine src/engine/entry.cpp)

set_source_files_properties( test/test_malloc_chatgpt.c PROPERTIES COMPILE_FLAGS "-O0" )


add_subdirectory(pybind11)
pybind11_add_module(fasterdp src/python/pybind.cpp)
# configure_file(python/*.pyi ${CMAKE_CURRENT_BINARY_DIR}/stubs/fasterdp COPYONLY)


# See https://github.com/pytorch/pytorch/issues/38122
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(fasterdp_core PUBLIC 
  ${TORCH_LIBRARIES} 
  ${TORCH_PYTHON_LIBRARY} 
  "python${PYTHON_VERSION}"
  libzmq
  libprotobuf
  ${Protobuf_LIBRARIES}
  fasterdp_protobuf
)

set_property(TARGET fasterdp_core PROPERTY CXX_STANDARD 14)

target_link_libraries(fasterdp PUBLIC fasterdp_core)
target_link_libraries(fasterdp_engine PUBLIC fasterdp_core)

add_custom_command(
        TARGET fasterdp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/src/python/fasterdp
                ${CMAKE_CURRENT_BINARY_DIR}/stubs/fasterdp)