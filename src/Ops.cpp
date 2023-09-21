#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::EX;

void mlir::EX::ConvOp::shape_inference()
{
}

void mlir::EX::ReluOp::shape_inference()
{
}

void mlir::EX::AddOp::shape_inference()
{
}

void mlir::EX::GlobalAveragePoolOp::shape_inference()
{
}

void mlir::EX::FlattenOp::shape_inference()
{
}

void mlir::EX::GemmOp::shape_inference()
{
}

void mlir::EX::MaxPoolOp::shape_inference()
{
}
