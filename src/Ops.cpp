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
    getResult().setType(getInput().getType());
}

void mlir::EX::ReluOp::shape_inference()
{
    getResult().setType(getInput().getType());
}

void mlir::EX::AddOp::shape_inference()
{
    getResult().setType(getInput1().getType());
}

void mlir::EX::GlobalAveragePoolOp::shape_inference()
{
    getResult().setType(getInputs().getType());
}

void mlir::EX::FlattenOp::shape_inference()
{
    getResult().setType(getInput().getType());
}

void mlir::EX::GemmOp::shape_inference()
{
    getResult().setType(getA().getType());
}

void mlir::EX::MaxPoolOp::shape_inference()
{
    // getResult().setType(getInput().getType());
}
