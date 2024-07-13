include("${CMAKE_SOURCE_DIR}/protobuf/cmake/protobuf-generate.cmake")

include_directories(${CMAKE_SOURCE_DIR}/protobuf/src)
include_directories(${CMAKE_SOURCE_DIR}/protobuf/third_party/abseil-cpp)
link_directories(${CMAKE_BINARY_DIR}/protobuf)

add_library(fasterdp_protobuf SHARED message.proto)
protobuf_generate(
    LANGUAGE cpp
    TARGET fasterdp_protobuf
)
protobuf_generate(
    LANGUAGE python
    TARGET fasterdp_protobuf
)

target_link_libraries(fasterdp_protobuf PUBLIC protobuf)