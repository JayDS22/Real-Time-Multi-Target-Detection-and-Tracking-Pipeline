cmake_minimum_required(VERSION 3.14)
project(imgpipeline
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Image detection and tracking pipeline — C++ translation of Python prototype"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compiler warnings — treat warnings seriously
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# option to enable profiling instrumentation
option(ENABLE_PROFILING "Add -pg flag for gprof" OFF)
if(ENABLE_PROFILING)
    add_compile_options(-pg)
    add_link_options(-pg)
endif()

# ---- core library (all pipeline modules) ----
add_library(imgpipe_core STATIC
    src/background_model.cpp
    src/morphology.cpp
    src/detection_extractor.cpp
    src/tracker.cpp
    src/pipeline.cpp
    src/image_io.cpp
)
target_include_directories(imgpipe_core PUBLIC include)

# ---- demo executable ----
add_executable(imgpipeline_demo src/main.cpp)
target_link_libraries(imgpipeline_demo PRIVATE imgpipe_core)

# ---- unit tests ----
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE imgpipe_core)
enable_testing()
add_test(NAME pipeline_tests COMMAND test_pipeline)

# ---- benchmark ----
add_executable(benchmark_pipeline benchmarks/benchmark_pipeline.cpp)
target_link_libraries(benchmark_pipeline PRIVATE imgpipe_core)

# installation
install(TARGETS imgpipeline_demo benchmark_pipeline DESTINATION bin)
install(TARGETS imgpipe_core DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/imgpipe)
