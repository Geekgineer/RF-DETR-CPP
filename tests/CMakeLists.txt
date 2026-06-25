# ============================================================================
# RFDETRs-CPP Tests
# ============================================================================
cmake_minimum_required(VERSION 3.16)

include(FetchContent)

# ============================================================================
# GoogleTest
# ============================================================================
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ============================================================================
# nlohmann/json (header-only)
# ============================================================================
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

include(GoogleTest)

# Base paths passed into test binaries as compile definitions
set(BASE_PATH_DETECTION    "${CMAKE_SOURCE_DIR}/tests/detection/")
set(BASE_PATH_SEGMENTATION "${CMAKE_SOURCE_DIR}/tests/segmentation/")
set(BASE_PATH_KEYPOINT     "${CMAKE_SOURCE_DIR}/tests/keypoint/")

# ============================================================================
# Helper: add a test executable
# ============================================================================
function(add_rfdetrs_test name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${ONNXRUNTIME_DIR}/include"
    )
    target_link_libraries(${name} PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        Threads::Threads
        GTest::gtest_main
        nlohmann_json::nlohmann_json
    )
    target_compile_definitions(${name} PRIVATE
        BASE_PATH_DETECTION="${BASE_PATH_DETECTION}"
        BASE_PATH_SEGMENTATION="${BASE_PATH_SEGMENTATION}"
        BASE_PATH_KEYPOINT="${BASE_PATH_KEYPOINT}"
    )
    if(UNIX AND NOT APPLE)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH   "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
    elseif(APPLE)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH   "@executable_path;${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib")
    endif()
    gtest_discover_tests(${name})
endfunction()

# ============================================================================
# Test: C++ inference runner (produces results_cpp.json)
# ============================================================================
add_executable(inference_rfdetr_cpp
    ${CMAKE_SOURCE_DIR}/tests/detection/inference_rfdetr_cpp.cpp
)
target_include_directories(inference_rfdetr_cpp PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${ONNXRUNTIME_DIR}/include"
)
target_link_libraries(inference_rfdetr_cpp PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    Threads::Threads
    nlohmann_json::nlohmann_json
)
if(UNIX AND NOT APPLE)
    set_target_properties(inference_rfdetr_cpp PROPERTIES
        BUILD_RPATH   "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
elseif(APPLE)
    set_target_properties(inference_rfdetr_cpp PROPERTIES
        BUILD_RPATH   "@executable_path;${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib")
endif()

# ============================================================================
# Test: Unit tests — math helpers + preprocessing (no ONNX model required)
# ============================================================================
add_rfdetrs_test(unit_tests
    ${CMAKE_SOURCE_DIR}/tests/unit/test_core.cpp
)

# ============================================================================
# Test: Parity comparison (GTest)
# ============================================================================
add_rfdetrs_test(compare_results
    ${CMAKE_SOURCE_DIR}/tests/detection/compare_results.cpp
)

# ============================================================================
# Test: Batch inference — RFDETRDetector on multiple images (library API)
# ============================================================================
add_rfdetrs_test(test_batch
    ${CMAKE_SOURCE_DIR}/tests/batch/test_batch.cpp
)

# ============================================================================
# Test: Video inference — RFDETRDetector on synthetic video frames (library API)
# ============================================================================
add_rfdetrs_test(test_video
    ${CMAKE_SOURCE_DIR}/tests/video/test_video.cpp
)

# ============================================================================
# Test: Segmentation — RFDETRSegDetector structural tests (library API)
# ============================================================================
add_rfdetrs_test(test_seg
    ${CMAKE_SOURCE_DIR}/tests/segmentation/test_seg.cpp
)

# ============================================================================
# Test: Keypoint — RFDETRKeypointDetector structural tests (library API)
# ============================================================================
add_rfdetrs_test(test_keypoint
    ${CMAKE_SOURCE_DIR}/tests/keypoint/test_keypoint.cpp
)

# ============================================================================
# CTest integration: seed results_cpp.json before parity tests run.
# gtest_discover_tests creates one CTest entry per test-case, so we cannot
# apply FIXTURES_REQUIRED to the entire suite from here.  Instead we expose
# a named "generate" step:
#
#   ctest -R generate_cpp_results   # step 1 — writes results_cpp.json
#   ctest -R ParityFixture          # step 2 — all parity assertions
#   ctest -R unit_tests             # independent unit tests (no model needed)
#   ctest -R BatchFixture           # batch API tests (needs detection model)
#   ctest -R VideoFixture           # video API tests (needs detection model + codec)
#   ctest -R SegFixture             # seg tests (needs seg model in tests/segmentation/models/)
#   ctest -R KeypointFixture        # keypoint tests (needs kp model in tests/keypoint/models/)
# ============================================================================
add_test(NAME generate_cpp_results
    COMMAND inference_rfdetr_cpp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ============================================================================
# CTest smoke test: rfdetr_batch_inference binary on real test images.
# Verifies the binary loads a model, processes a directory, and exits 0.
# Only registered when the test model is present at configure time.
# ============================================================================
set(_RFDETR_TEST_MODEL "${CMAKE_SOURCE_DIR}/tests/detection/models/rfdetr-base.onnx")
set(_RFDETR_TEST_IMGS  "${CMAKE_SOURCE_DIR}/tests/detection/data/images")

if(EXISTS "${_RFDETR_TEST_MODEL}")
    add_test(NAME batch_binary_smoke
        COMMAND $<TARGET_FILE:rfdetr_batch_inference>
                "${_RFDETR_TEST_MODEL}"
                "${_RFDETR_TEST_IMGS}"
                "" 0.5 300 cpu
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set_tests_properties(batch_binary_smoke PROPERTIES
        TIMEOUT 120
        PASS_REGULAR_EXPRESSION "SUMMARY"
    )
endif()
