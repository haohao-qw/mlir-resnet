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
    auto type = getOperand(0).getType();
    getResult().setType(type.cast<mlir::TensorType>());
    llvm::outs() << "ConvOp shape_inferenc()\n";
}

void mlir::EX::AddOp::shape_inference()
{
    auto type = getOperand(0).getType();
    getResult().setType(type.cast<mlir::TensorType>());
    llvm::outs() << "AddOp shape_inferenc()\n";
}

void mlir::EX::ReluOp::shape_inference()
{
    llvm::outs() << "ReluOp shape_inferenc()\n";
}

void mlir::EX::GlobalAveragePoolOp::shape_inference()
{
    llvm::outs() << "GlobalAveragePoolOp shape_inferenc()\n";
}

void mlir::EX::FlattenOp::shape_inference()
{
    llvm::outs() << "FlattenOp shape_inferenc()\n";
}

void mlir::EX::GemmOp::shape_inference()
{
    auto type = getOperand(0).getType();
    getResult().setType(type.cast<mlir::TensorType>());
    llvm::outs() << "GemmOp shape_inferenc()\n";
}

void mlir::EX::MaxPoolOp::shape_inference()
{
    llvm::outs() << "MaxPoolOp shape_inferenc()\n";
}