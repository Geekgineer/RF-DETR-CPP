# ============================================================================
# RFDETRs-CPP — C++ ONNX Runtime Inference for RF-DETR
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(RFDETRs-CPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Options
# ============================================================================
option(BUILD_TESTS        "Build parity test suite"                                        OFF)
option(BUILD_EXAMPLES     "Build example applications"                                     OFF)
option(BUILD_BENCHMARKS   "Build benchmark suite"                                          OFF)
option(ENABLE_WARNINGS    "Enable compiler warnings"                                       ON)
option(ENABLE_NATIVE_ARCH "Enable -march=native (ties binary to build CPU; not portable)" OFF)

# ============================================================================
# ONNX Runtime discovery (mirrors YOLOs-CPP)
# ============================================================================
if(NOT DEFINED ONNXRUNTIME_DIR)
    if(WIN32)
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-gpu-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime"
            "$ENV{ONNXRUNTIME_DIR}"
        )
    elseif(APPLE)
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-osx-arm64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-osx-x86_64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime"
            "/opt/onnxruntime"
            "$ENV{ONNXRUNTIME_DIR}"
        )
    else()
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime"
            "/opt/onnxruntime"
            "$ENV{ONNXRUNTIME_DIR}"
        )
    endif()

    foreach(path ${ONNXRUNTIME_SEARCH_PATHS})
        if(EXISTS "${path}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_DIR "${path}")
            break()
        endif()
    endforeach()
endif()

if(ONNXRUNTIME_DIR AND NOT IS_ABSOLUTE "${ONNXRUNTIME_DIR}")
    get_filename_component(ONNXRUNTIME_DIR "${ONNXRUNTIME_DIR}" ABSOLUTE
                           BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(NOT ONNXRUNTIME_DIR OR NOT EXISTS "${ONNXRUNTIME_DIR}/include")
    message(FATAL_ERROR
        "ONNX Runtime not found!\n"
        "Set ONNXRUNTIME_DIR: cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime\n"
        "Or download from: https://github.com/microsoft/onnxruntime/releases")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")

# ============================================================================
# OpenCV
# ============================================================================
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ============================================================================
# Threading
# ============================================================================
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ============================================================================
# Compiler flags
# ============================================================================
if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT MSVC)
    add_compile_options(-O3)
    if(ENABLE_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()
endif()

# ============================================================================
# ONNX Runtime library
# ============================================================================
if(UNIX AND NOT APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    message(STATUS "Platform: Linux")
elseif(APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    message(STATUS "Platform: macOS")
elseif(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    message(STATUS "Platform: Windows")
endif()

# ============================================================================
# Include directories
# ============================================================================
include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${ONNXRUNTIME_DIR}/include"
)

# ============================================================================
# Helper: add an rfdetrs executable with correct link/rpath settings
# ============================================================================
function(add_rfdetrs_executable name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE "${ONNXRUNTIME_DIR}/include")
    target_link_libraries(${name} PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        Threads::Threads
    )
    if(WIN32)
        set(ORT_DLL "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
        if(EXISTS "${ORT_DLL}")
            add_custom_command(TARGET ${name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_DLL}" "$<TARGET_FILE_DIR:${name}>")
        endif()
    elseif(UNIX AND NOT APPLE)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH   "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
    elseif(APPLE)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH   "@executable_path;${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib")
    endif()
endfunction()

# ============================================================================
# Main executables
# ============================================================================
add_rfdetrs_executable(rfdetr_image_inference          src/rfdetr_image_inference.cpp)
add_rfdetrs_executable(rfdetr_video_inference          src/rfdetr_video_inference.cpp)
add_rfdetrs_executable(rfdetr_camera_inference         src/rfdetr_camera_inference.cpp)
add_rfdetrs_executable(rfdetr_batch_inference          src/rfdetr_batch_inference.cpp)
add_rfdetrs_executable(rfdetr_seg_image_inference      src/rfdetr_seg_image_inference.cpp)
add_rfdetrs_executable(rfdetr_keypoint_image_inference src/rfdetr_keypoint_image_inference.cpp)

# ============================================================================
# Tests (optional)
# ============================================================================
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ============================================================================
# Examples (optional)
# ============================================================================
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Benchmarks (optional)
# ============================================================================
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== RFDETRs-CPP Configuration ===")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:         ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime:   ${ONNXRUNTIME_DIR}")
message(STATUS "  Build Tests:    ${BUILD_TESTS}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "")
