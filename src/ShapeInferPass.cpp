#include "mlir/Pass/Pass.h"
#include "Dialect.h"
#include "ShapeInfer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/InitAllDialects.h"
#include "Passes.h"

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace EX;

#include "ShapeInfer.cpp.inc"

namespace
{
    struct ShapeInferencePass
        : public mlir::PassWrapper<ShapeInferencePass, OperationPass<mlir::func::FuncOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

        void runOnOperation() override
        {
            auto f = getOperation();

            f.walk([&](mlir::Operation *op)
                   {
                       if (auto shapeOp = dyn_cast<ShapeInterface>(op)){
                            shapeOp.shape_inference();
                       } });
        }
    };
} // namespace

std::unique_ptr<mlir::Pass> mlir::EX::createShapeInferencePass()
{
    return std::make_unique<ShapeInferencePass>();
}
