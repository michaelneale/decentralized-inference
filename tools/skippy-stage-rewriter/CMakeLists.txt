cmake_minimum_required(VERSION 3.20)

project(skippy_stage_rewriter LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

add_executable(skippy-stage-rewriter
    main.cpp)

target_compile_features(skippy-stage-rewriter PRIVATE cxx_std_17)
target_include_directories(skippy-stage-rewriter SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS})
target_compile_definitions(skippy-stage-rewriter PRIVATE
    ${LLVM_DEFINITIONS})

target_link_libraries(skippy-stage-rewriter PRIVATE
    clang-cpp
    LLVM)

include(CTest)
if (BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME skippy-stage-rewriter-fixture
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rewriter.py
            $<TARGET_FILE:skippy-stage-rewriter>
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/source)
endif()
