include(AddMLIRPython)

set(EX_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(EX_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python")


declare_mlir_python_sources(EXMLIRPythonSources)
declare_mlir_python_sources(EXMLIRPythonSources.Dialects
  ROOT_DIR "${EX_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT EXMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT EXMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/EX.td
  SOURCES dialects/EX.py
  DIALECT_NAME EX)

set(_source_components
  EXMLIRPythonSources
)
add_mlir_python_modules(EXMLIRPythonModules
  ROOT_PREFIX "${EX_MLIR_PYTHON_PACKAGES_DIR}/mlir"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES ${_source_components}
  )


install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "*.py"
  PATTERN "xlsx_to_py.py" EXCLUDE
  PATTERN "test_*" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
)
# install(
#   DIRECTORY ./ DESTINATION python FILES_MATCHING
#   USE_SOURCE_PERMISSIONS
#   PATTERN "test_onnx.py"
# )