cmake_minimum_required(VERSION 3.20.0)
project(mlir-example)


set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
# Debug模式
set(CMAKE_BUILD_TYPE "Debug")

set(LLVM_DIR /root/depend/llvm/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR /root/depend/llvm/llvm-project/build/lib/cmake/mlir)


find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(HandleLLVMOptions)


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/build/include)
include_directories(${PROJECT_SOURCE_DIR}/build/python)

link_directories(${LLVM_LIBRARY_DIR})
add_subdirectory(include)
add_subdirectory(python)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${extension_libs}
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    LLVMCore
    LLVMSupport
    MLIRAnalysis
    MLIRPass
    # MLIRSideEffectInterfaces
    MLIRTransforms
    )

set(DEPENDS
  ExTdGen
)
set(SRC_LIST
    src/Rewrite.cpp
    src/Dialect.cpp
    src/Ops.cpp
    src/ShapeInferPass.cpp
    src/Conversion.cpp
)
set(TEST_NAME
    test/example.cpp
)
add_executable(example ${TEST_NAME} ${SRC_LIST})
add_dependencies(example ${DEPENDS})
target_link_libraries(example PRIVATE ${LIBS})

set(TEST_NAME
    test/read.cpp
)
add_executable(read ${TEST_NAME} ${SRC_LIST})
add_dependencies(read ${DEPENDS})
target_link_libraries(read PRIVATE ${LIBS})