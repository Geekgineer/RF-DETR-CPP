cmake_minimum_required(VERSION 3.16)
project(rfdetrs_examples)

find_package(OpenCV REQUIRED)
include_directories("../include/")

set(EXAMPLES
    example_image_det
    example_video_det
    example_camera_det
)

foreach(ex ${EXAMPLES})
    add_executable(${ex} ${ex}.cpp)
    target_include_directories(${ex} PRIVATE "${ONNXRUNTIME_DIR}/include")
    target_link_libraries(${ex} PRIVATE ${OpenCV_LIBS})
    if(UNIX AND NOT APPLE)
        target_link_libraries(${ex} PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
        set_target_properties(${ex} PROPERTIES
            BUILD_RPATH   "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
    elseif(APPLE)
        target_link_libraries(${ex} PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
        set_target_properties(${ex} PROPERTIES
            BUILD_RPATH   "@executable_path;${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib")
    elseif(WIN32)
        target_link_libraries(${ex} PRIVATE "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    endif()
endforeach()
