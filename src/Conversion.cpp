#include "mlir/IR/BuiltinDialect.h"
#include "Dialect.h"
#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
static MemRefType convertTensorToMemRef(RankedTensorType type)
{
    return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter)
{
    auto alloc = rewriter.create<memref::AllocOp>(loc, type);
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration)
{
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    auto loc = op->getLoc();

    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, tensorType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs)
        {
            Value valueToStore = processIteration(nestedBuilder, operands, ivs);
            nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                        ivs);
        });

    rewriter.replaceOp(op, alloc);
}

namespace
{
    template <typename BinaryOp, typename LoweredBinaryOp>
    struct BinaryOpLowering : public ConversionPattern
    {
        BinaryOpLowering(MLIRContext *ctx)
            : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            lowerOpToLoops(op, operands, rewriter,
                           [loc](OpBuilder &builder, ValueRange memRefOperands,
                                 ValueRange loopIvs)
                           {
                               typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                               auto loadedLhs = builder.create<affine::AffineLoadOp>(
                                   loc, binaryAdaptor.getLhs(), loopIvs);
                               auto loadedRhs = builder.create<affine::AffineLoadOp>(
                                   loc, binaryAdaptor.getRhs(), loopIvs);

                               return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                                      loadedRhs);
                           });
            return success();
        }
    };
    using AddOpLowering = BinaryOpLowering<EX::AddOp, arith::AddFOp>;

    struct ConstantOpLowering : public OpRewritePattern<EX::ConstantOp>
    {
        using OpRewritePattern<EX::ConstantOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(EX::ConstantOp op,
                                      PatternRewriter &rewriter) const final
        {
            DenseElementsAttr constantValue = op.getValue().dyn_cast<DenseElementsAttr>();
            Location loc = op.getLoc();

            auto tensorType = llvm::cast<RankedTensorType>(op.getType());
            auto memRefType = convertTensorToMemRef(tensorType);
            auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

            auto valueShape = memRefType.getShape();
            SmallVector<Value, 8> constantIndices;

            if (!valueShape.empty())
            {
                for (auto i : llvm::seq<int64_t>(
                         0, *std::max_element(valueShape.begin(), valueShape.end())))
                    constantIndices.push_back(
                        rewriter.create<arith::ConstantIndexOp>(loc, i));
            }
            else
            {
                constantIndices.push_back(
                    rewriter.create<arith::ConstantIndexOp>(loc, 0));
            }

            SmallVector<Value, 2> indices;
            auto valueIt = constantValue.value_begin<FloatAttr>();
            std::function<void(uint64_t)> storeElements = [&](uint64_t dimension)
            {
                if (dimension == valueShape.size())
                {
                    rewriter.create<affine::AffineStoreOp>(
                        loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
                        llvm::ArrayRef(indices));
                    return;
                }

                for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i)
                {
                    indices.push_back(constantIndices[i]);
                    storeElements(dimension + 1);
                    indices.pop_back();
                }
            };

            storeElements(/*dimension=*/0);

            rewriter.replaceOp(op, alloc);
            return success();
        }
    };

    struct ReluOpLowering : public ConversionPattern
    {
        ReluOpLowering(MLIRContext *ctx)
            : ConversionPattern(EX::ReluOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            lowerOpToLoops(op, operands, rewriter,
                           [loc](OpBuilder &builder, ValueRange memRefOperands,
                                 ValueRange loopIvs)
                           {
                               EX::ReluOpAdaptor ReluAdaptor(memRefOperands);
                               Value input = ReluAdaptor.getInput();

                               SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                               return builder.create<affine::AffineLoadOp>(loc, input,
                                                                           reverseIvs);
                           });
            return success();
        }
    };

} // namespace

namespace
{
    struct ExConversionPass
        : public PassWrapper<ExConversionPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExConversionPass)

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<affine::AffineDialect, func::FuncDialect,
                            memref::MemRefDialect>();
        }
        void runOnOperation() final;
    };
} // namespace

void ExConversionPass::runOnOperation()
{
    ConversionTarget target(getContext());

    target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                           arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect>();

    target.addIllegalDialect<EX::EXDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering, ConstantOpLowering, ReluOpLowering>(
        &getContext());

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::EX::createConversionPass()
{
    return std::make_unique<ExConversionPass>();
}