cmake_minimum_required(VERSION 3.21)

project(tabletop_object_recognition VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn)
find_package(SQLite3 REQUIRED)

function(enable_project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(tabletop_recognition_core
    src/core/observability.cpp
    src/core/pipeline.cpp
    src/core/frame_source/sources.cpp
    src/core/preprocess/color_score_preprocessor.cpp
    src/core/segmentation/kmeans_segmenter.cpp
    src/core/morphology/basic_morphology.cpp
    src/core/components/connected_component_analyzer.cpp
    src/core/features/shape_feature_extractor.cpp
    src/core/classifier/nearest_neighbor_classifier.cpp
    src/core/tracker/centroid_tracker.cpp
    src/core/renderer/opencv_renderer.cpp
    src/core/storage/sqlite_run_storage.cpp
)

target_include_directories(tabletop_recognition_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_include_directories(tabletop_recognition_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(tabletop_recognition_core PUBLIC ${OpenCV_LIBS})
target_link_libraries(tabletop_recognition_core PUBLIC SQLite::SQLite3)

enable_project_warnings(tabletop_recognition_core)

add_executable(tabletop_recognition_live src/app/live_main.cpp)
add_executable(tabletop_recognition_eval src/app/eval_main.cpp)
add_executable(tabletop_recognition_train src/app/train_main.cpp)

foreach(app tabletop_recognition_live tabletop_recognition_eval tabletop_recognition_train)
    target_link_libraries(${app} PRIVATE tabletop_recognition_core)
    enable_project_warnings(${app})
endforeach()

install(TARGETS tabletop_recognition_live tabletop_recognition_eval tabletop_recognition_train RUNTIME DESTINATION bin)

include(CTest)

if(BUILD_TESTING)
    find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)

    function(configure_vision_test target label)
        set(test_temp_dir "${CMAKE_CURRENT_BINARY_DIR}/test-tmp/${target}")
        file(MAKE_DIRECTORY "${test_temp_dir}")
        set_tests_properties(${target} PROPERTIES
            LABELS "${label}"
            TIMEOUT 120
            ENVIRONMENT "TMPDIR=${test_temp_dir};TMP=${test_temp_dir};TEMP=${test_temp_dir}"
        )
    endfunction()

    function(add_vision_test target source label)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE tabletop_recognition_core)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        )
        target_compile_definitions(${target} PRIVATE VISION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        enable_project_warnings(${target})
        # Keep assertions active when the production targets use a release build.
        if(MSVC)
            target_compile_options(${target} PRIVATE /UNDEBUG)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -UNDEBUG)
        endif()

        add_test(NAME ${target} COMMAND ${target})
        configure_vision_test(${target} ${label})
    endfunction()

    function(add_vision_python_test target source label)
        add_test(NAME ${target}
            COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/${source}" ${ARGN}
        )
        configure_vision_test(${target} ${label})
    endfunction()

    add_vision_test(test_types tests/unit/test_types.cpp unit)
    add_vision_test(test_segmentation_threshold tests/unit/test_segmentation_threshold.cpp unit)
    add_vision_test(test_morphology tests/unit/test_morphology.cpp unit)
    add_vision_test(test_region_geometry tests/unit/test_region_geometry.cpp unit)
    add_vision_test(test_classifier_distance tests/unit/test_classifier_distance.cpp unit)
    add_vision_test(test_observability tests/unit/test_observability.cpp unit)
    add_vision_test(test_storage tests/unit/test_storage.cpp unit)
    add_vision_python_test(test_evaluation tests/unit/test_evaluation.py unit)
    add_vision_python_test(test_dataset_tools tests/unit/test_dataset_tools.py unit)

    add_vision_test(test_pipeline_smoke tests/integration/test_pipeline_smoke.cpp integration)
    add_vision_test(test_pipeline_fixed_image tests/integration/test_pipeline_fixed_image.cpp integration)
    add_vision_test(test_training_persistence tests/integration/test_training_persistence.cpp integration)
    add_vision_python_test(test_eval_cli tests/integration/test_eval_cli.py integration
        --binary $<TARGET_FILE:tabletop_recognition_eval>
    )
    add_vision_python_test(test_experiment_workflow tests/integration/test_experiment_workflow.py integration
        --binary $<TARGET_FILE:tabletop_recognition_eval>
        --train-binary $<TARGET_FILE:tabletop_recognition_train>
    )

    add_vision_test(test_pipeline_regression tests/regression/test_pipeline_regression.cpp regression)
    add_vision_test(test_pipeline_performance tests/performance/test_pipeline_performance.cpp performance)
endif()
