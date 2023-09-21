#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"

using namespace mlir;
using namespace mlir::EX;

#include "Dialect.cpp.inc"

void EXDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
        >();
}

// 通过宏控制引入范围
#define GET_OP_CLASSES
#include "Ops.cpp.inc"