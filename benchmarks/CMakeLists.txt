cmake_minimum_required(VERSION 3.16)
project(rfdetr_benchmark VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
endif()

# ONNX Runtime
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime directory")
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_c_api.h")
    message(FATAL_ERROR "Set -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

add_executable(rfdetr_benchmark rfdetr_benchmark.cpp)
target_include_directories(rfdetr_benchmark PRIVATE
    "${CMAKE_SOURCE_DIR}/../include"
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(rfdetr_benchmark PRIVATE ${OpenCV_LIBS})

if(UNIX AND NOT APPLE)
    target_link_libraries(rfdetr_benchmark PRIVATE
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" pthread dl)
    set_target_properties(rfdetr_benchmark PROPERTIES
        BUILD_RPATH   "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
elseif(APPLE)
    target_link_libraries(rfdetr_benchmark PRIVATE
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    set_target_properties(rfdetr_benchmark PROPERTIES
        BUILD_RPATH   "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
elseif(WIN32)
    target_link_libraries(rfdetr_benchmark PRIVATE
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif()

message(STATUS "")
message(STATUS "=== RFDETRs-CPP Benchmark ===")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_DIR}")
message(STATUS "")
