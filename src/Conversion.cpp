#include "Dialect.h"
#include "Passes.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include <vector>
using namespace mlir;
using namespace EX;

static MemRefType convertTensorToMemRef(RankedTensorType type)
{
    return MemRefType::get(type.getShape(), type.getElementType());
}

// 内存分配
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
    auto tensorType = mlir::RankedTensorType::get({1, 1}, rewriter.getF32Type());
    auto loc = op->getLoc();

    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, memRefType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs)
        {
            Value valueToStore = processIteration(nestedBuilder, operands, ivs);
            nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                        ivs);
        });

    rewriter.replaceOp(op, alloc);
}

// 两种重写方式，都可以作为模式被应用：
// 1.继承ConversionPattern
// 2.继承OpRewritePattern
namespace
{

    static mlir::RankedTensorType getRankedFromOp(Operation *op)
    {
        auto origin_type = op->getResult(0).getType();
        auto ty = origin_type.cast<mlir::RankedTensorType>();
        auto output_type = RankedTensorType::get(ty.getShape(), ty.getElementType());
        return output_type;
    }

    // Add Conv Flatten GlobalAveragePool Relu Gemm算子到Linalg
    struct AddLowering : public ConversionPattern
    {
        AddLowering(MLIRContext *ctx)
            : ConversionPattern(EX::AddOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const final
        {
            // 由于TOSA有对应的表示，因此不用create，直接replace就行
            auto output_type = getRankedFromOp(op);
            std::vector<Value> vec;
            for (auto in : op->getOperands())
            {
                vec.push_back(in);
            }
            rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, output_type, vec);
            return success();
        }
    };

    struct ConvLowering : public ConversionPattern
    {
        ConvLowering(MLIRContext *ctx)
            : ConversionPattern(EX::ConvOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            auto output_type = getRankedFromOp(op);
            std::vector<NamedAttribute> attrs;
            return success();
        }
    };

    struct FlattenLowering : public ConversionPattern
    {
        FlattenLowering(MLIRContext *ctx)
            : ConversionPattern(EX::FlattenOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            return success();
        }
    };

    struct GlobalAveragePoolLowering : public ConversionPattern
    {
        GlobalAveragePoolLowering(MLIRContext *ctx)
            : ConversionPattern(EX::GlobalAveragePoolOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            return success();
        }
    };

    struct ReluLowering : public ConversionPattern
    {
        ReluLowering(MLIRContext *ctx)
            : ConversionPattern(EX::ReluOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            return success();
        }
    };

    struct GemmLowering : public ConversionPattern
    {
        GemmLowering(MLIRContext *ctx)
            : ConversionPattern(EX::GemmOp::getOperationName(), 1, ctx) {}

        LogicalResult
        matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            return success();
        }
    };

    // struct ConstantLowering : public ConversionPattern
    // {
    //     ConstantLowering(MLIRContext *ctx)
    //         : ConversionPattern(EX::ConstantOp::getOperationName(), 1, ctx) {}

    //     LogicalResult
    //     matchAndRewrite(Operation *op, ArrayRef<Value> operands,
    //                     ConversionPatternRewriter &rewriter) const final
    //     {
    //         auto loc = op->getLoc();
    //         return success();
    //     }
    // };

    // 二元算子 ConversionPattern
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
                                   loc, binaryAdaptor.getA(), memRefOperands);
                               auto loadedRhs = builder.create<affine::AffineLoadOp>(
                                   loc, binaryAdaptor.getB(), memRefOperands);
                               return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                                      loadedRhs);
                           });
            return success();
        }
    };
    // using AddOpLowering = BinaryOpLowering<EX::AddOp, arith::AddFOp>;

    // constant 算子
    struct ConstantOpLowering : public OpRewritePattern<EX::ConstantOp>
    {
        using OpRewritePattern<EX::ConstantOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(EX::ConstantOp op,
                                      PatternRewriter &rewriter) const final
        {
            DenseElementsAttr constantValue = op.getValue().dyn_cast<mlir::DenseElementsAttr>();
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

} // namespace

// 流程
namespace
{
    struct EXConversion : public PassWrapper<EXConversion, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EXConversion)

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                            memref::MemRefDialect, tosa::TosaDialect>();
        }
        void runOnOperation() final;
    };
} // namespace

void EXConversion::runOnOperation()
{
    // 定义转换目标
    ConversionTarget target(getContext());
    // 这里设置转换到的目标dialect都是合法的，也就是允许这些dialect出现在最终生成的mlir文件中
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect, affine::AffineDialect, tosa::TosaDialect,
                           memref::MemRefDialect>();
    // 设置非法dialect，也就是不允许这些dialect出现在最终生成的mlir文件中，如果转换后还存在这些dialect的操作，会报错
    target.addIllegalDialect<mlir::EX::EXDialect>();
    // 可以设置部分op为合法，不进行转换
    // 合法Op，本例子只有addop和constantop
    target.addLegalOp<mlir::EX::GemmOp, mlir::EX::ConvOp, mlir::EX::FlattenOp, mlir::EX::GlobalAveragePoolOp,
                      mlir::EX::ReluOp, mlir::func::FuncOp, mlir::EX::ConstantOp, mlir::func::ReturnOp>();

    // 添加定义的转换模式
    RewritePatternSet patterns(&getContext());
    patterns.add<AddLowering>(
        &getContext());

    // 执行转换
    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::EX::createConversionPass()
{
    return std::make_unique<EXConversion>();
}