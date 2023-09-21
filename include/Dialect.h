#ifndef MLIR_EX_DIALECT_H_
#define MLIR_EX_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinOps.h"

#include "ShapeInfer.h"

#include "Dialect.h.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"

#endif // MLIR_EX_DIALECT_H_
