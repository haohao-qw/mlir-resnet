#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "Dialect.h"
#include <numeric>
using namespace mlir;
using namespace EX;

namespace
{
#include "Rewrite.inc"
} // namespace

struct SimplifyConstantOp : public mlir::OpRewritePattern<EX::ConstantOp>
{
    SimplifyConstantOp(mlir::MLIRContext *context)
        : OpRewritePattern<EX::ConstantOp>(context, /*benefit=*/1) {}

    mlir::LogicalResult
    matchAndRewrite(EX::ConstantOp op,
                    mlir::PatternRewriter &rewriter) const override
    {
        llvm::outs() << "matchAndRewrite\n";
        return success();
    }
};

void EX::ConstantOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context)
{
    llvm::outs() << "add matchAndRewrite\n";
    results.add<SimplifyConstantOp>(context);
}