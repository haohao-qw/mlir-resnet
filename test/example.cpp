#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

// #include "llvm/ADT/ScopedHashTable.h"
// #include "llvm/ADT/StringRef.h"
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/ErrorOr.h"
// #include "llvm/Support/MemoryBuffer.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/Passes.h"

#include "Dialect.h"
#include "Passes.h"
#include <vector>
#include <string>

using namespace mlir;
using namespace mlir::EX;
namespace cl = llvm::cl;

struct FileLocation
{
  std::string filename;
  int line;
  int column;
};

mlir::Location loc(mlir::OpBuilder &builder, const FileLocation &loc)
{
  return mlir::FileLineColLoc::get(builder.getStringAttr(loc.filename), loc.line,
                                   loc.column);
}

mlir::Location getNameLoc(mlir::OpBuilder &builder, const std::string &name)
{
  return mlir::NameLoc::get(builder.getStringAttr(name));
}

void Init()
{
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
}

void RegisterDialect(MLIRContext &context)
{
  DialectRegistry registry;
  registry.insert<mlir::EX::EXDialect>();

  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

int Test()
{
  Init();
  mlir::MLIRContext context;
  RegisterDialect(context);

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::ModuleOp theModule;

  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  FileLocation localtion{"../test/test.txt", 0, 0};

  // 设置插入点
  builder.setInsertionPointToEnd(theModule.getBody());

  // 构造lc，对应源文件位置信息
  // auto lc = loc(builder, localtion);
  auto lc = getNameLoc(builder, "main");

  // 构造FuncOp,需要指定参数类型和返回值类型，返回类型需要和returnOp的返回值类型一致，否则生成的mlir文件无法被正确读取。
  auto itype = mlir::RankedTensorType::get({1, 3, 224, 224}, builder.getF64Type());
  llvm::SmallVector<mlir::Type, 2> InputArg(1, itype);
  // 1*1000的返回
  auto otype = mlir::RankedTensorType::get({1, 1000}, builder.getF64Type());
  llvm::SmallVector<mlir::Type, 2> OutputArg(1, otype);
  auto funcType = builder.getFunctionType(InputArg, OutputArg);
  // auto funcType = builder.getFunctionType(InputArg, std::nullopt);
  llvm::StringRef funname = "main";
  auto funcOp = builder.create<mlir::func::FuncOp>(lc, funname, funcType);

  // 获取funcop的body，以此设置插入点，之后的内容都会插入到funcop的body中
  mlir::Block *efblock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(efblock);

  // 获取funcop的args
  auto output = mlir::UnrankedTensorType::get(builder.getF32Type());
  auto args = efblock->getArguments();
  mlir::Value arg0 = args[0];
  mlir::Value arg1 = args[0];
  lc = getNameLoc(builder, "add");
  auto argret = builder.create<AddOp>(lc, output, arg0, arg1);

  std::vector<float> vec;
  for (int i = 0; i < 2048; i++)
  {
    vec.push_back(static_cast<float>(i));
  }
  auto vec_shape = mlir::RankedTensorType::get({1, 8, 8, 32}, builder.getF32Type());
  auto vec_attr = mlir::DenseElementsAttr::get(vec_shape, llvm::ArrayRef(vec));
  lc = getNameLoc(builder, "constant");
  auto vec_retop = builder.create<mlir::EX::ConstantOp>(lc, output, Attribute(vec_attr));
  builder.create<AddOp>(lc, output, vec_retop, argret);

  // ConstantOp
  auto op_consant = builder.create<mlir::EX::ConstantOp>(lc, output, Attribute(vec_attr));

  // ConvOp
  ArrayRef<int64_t> stride_attr{1, 1};
  auto stride = builder.getI64ArrayAttr(stride_attr);
  ArrayRef<int64_t> kernel_shape_attr{3, 3};
  auto kernel_shape = builder.getI64ArrayAttr(kernel_shape_attr);
  ArrayRef<int64_t> padding_attr{1, 1, 1, 1};
  auto padding = builder.getI64ArrayAttr(padding_attr);
  ArrayRef<int64_t> dilations_attr{1, 1};
  auto dilations = builder.getI64ArrayAttr(dilations_attr);
  lc = getNameLoc(builder, "conv");
  auto op_conv = builder.create<mlir::EX::ConvOp>(lc, output, op_consant, op_consant, op_consant, stride, kernel_shape, padding, 1, dilations, "NOTSET");

  // ReluOp
  lc = getNameLoc(builder, "relu");
  auto op_relu = builder.create<mlir::EX::ReluOp>(lc, output, op_conv);

  // MaxPoolOp
  auto auto_pad = builder.getStringAttr("NOTSET");
  auto ceil_mode = builder.getI64IntegerAttr(0);
  auto count_include_pad = builder.getBoolAttr(false);
  auto p = builder.getI64IntegerAttr(1);
  auto pads = builder.getI64ArrayAttr(padding_attr);
  auto storage_order = builder.getI64IntegerAttr(0);
  lc = getNameLoc(builder, "maxpool");
  auto op_maxpool = builder.create<mlir::EX::MaxPoolOp>(lc, output, output, op_relu, auto_pad, ceil_mode, dilations, kernel_shape, p, pads, storage_order, stride);

  // GlobalAveragePoolOp
  lc = getNameLoc(builder, "globalAveragepool");
  auto op_globalaveragepool = builder.create<mlir::EX::GlobalAveragePoolOp>(lc, output, op_maxpool.getResult(0));

  // FlattenOp
  lc = getNameLoc(builder, "flatten");
  auto op_flatten = builder.create<mlir::EX::FlattenOp>(lc, output, op_globalaveragepool);

  // GemmOp
  auto alpha = builder.getF64FloatAttr(1.0);
  auto beta = builder.getF64FloatAttr(1.0);
  auto tranA = builder.getI64IntegerAttr(0);
  auto tranB = builder.getI64IntegerAttr(0);
  auto retype = mlir::RankedTensorType::get({1, 1000}, builder.getF64Type());
  lc = getNameLoc(builder, "gemm");
  auto op_gemm = builder.create<mlir::EX::GemmOp>(lc, retype, op_flatten, op_flatten, op_flatten, alpha, beta, tranA, tranB);

  // 构造ReturnOp，和funcop的返回值类型一致
  mlir::ValueRange retVals{op_gemm};
  lc = getNameLoc(builder, "end");
  builder.create<mlir::func::ReturnOp>(lc, retVals);
  // 设置插入点，funcop到此body结束
  builder.setInsertionPointToEnd(efblock);
  builder.setInsertionPointToEnd(theModule.getBody());

  // Apply Pass
  module = theModule;
  mlir::PassManager pm(module.get()->getName());
  pm.addPass(mlir::EX::createConversionPass());

  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(mlir::EX::createShapeInferencePass());
  // optPM.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*module)))
    return 4;

  // save file
  std::error_code ec;
  llvm::StringRef filename{"output.mlir"};
  llvm::raw_fd_ostream file(filename, ec);
  module->print(file);
}

int Test2()
{
  Init();
  mlir::MLIRContext context;
  RegisterDialect(context);

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::ModuleOp theModule;

  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  // 设置插入点
  builder.setInsertionPointToEnd(theModule.getBody());

  // 构造lc，对应源文件位置信息
  auto lc = getNameLoc(builder, "main");

  // 构造FuncOp,需要指定参数类型和返回值类型，返回类型需要和returnOp的返回值类型一致，否则生成的mlir文件无法被正确读取。
  auto itype = mlir::RankedTensorType::get({1, 3, 224, 224}, builder.getF64Type());
  llvm::SmallVector<mlir::Type, 2> InputArg(1, itype);
  // 1*1000的返回
  auto otype = mlir::RankedTensorType::get({1, 1000}, builder.getF64Type());
  llvm::SmallVector<mlir::Type, 2> OutputArg(1, otype);
  // auto funcType = builder.getFunctionType(InputArg, OutputArg);
  auto funcType = builder.getFunctionType(InputArg, std::nullopt);
  llvm::StringRef funname = "main";
  auto funcOp = builder.create<mlir::func::FuncOp>(lc, funname, funcType);

  // 获取funcop的body，以此设置插入点，之后的内容都会插入到funcop的body中
  mlir::Block *efblock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(efblock);

  auto output = UnrankedTensorType::get(builder.getF64Type());

  // 获取funcop的args
  // auto args = efblock->getArguments();
  // mlir::Value arg0 = args[0];
  // lc = loc(builder, "args");
  // auto arg_op = builder.create<AddOp>(lc, output, arg0, arg0);

  std::vector<float> vec;
  for (int i = 0; i < 2048; i++)
  {
    vec.push_back(static_cast<float>(i));
  }
  auto vec_shape = mlir::RankedTensorType::get({1, 8, 8, 32}, builder.getF32Type());
  auto vec_attr = mlir::DenseElementsAttr::get(vec_shape, llvm::ArrayRef(vec));
  lc = getNameLoc(builder, "constant");
  auto consant_op = builder.create<ConstantOp>(lc, vec_shape, Attribute(vec_attr));
  lc = getNameLoc(builder, "add");
  auto add_op = builder.create<AddOp>(lc, output, consant_op, consant_op);
  lc = getNameLoc(builder, "constant");
  auto constantop = builder.create<ConstantOp>(lc, vec_shape, Attribute(vec_attr));
  // // ConvOp
  ArrayRef<int64_t> stride_attr{1, 1};
  auto stride = builder.getI64ArrayAttr(stride_attr);
  ArrayRef<int64_t> kernel_shape_attr{3, 3};
  auto kernel_shape = builder.getI64ArrayAttr(kernel_shape_attr);
  ArrayRef<int64_t> padding_attr{1, 1, 1, 1};
  auto padding = builder.getI64ArrayAttr(padding_attr);
  ArrayRef<int64_t> dilations_attr{1, 1};
  auto dilations = builder.getI64ArrayAttr(dilations_attr);
  lc = getNameLoc(builder, "conv");
  auto op_conv = builder.create<mlir::EX::ConvOp>(lc, output, consant_op, consant_op, consant_op, stride, kernel_shape, padding, 1, dilations, "NOTSET");

  // // ReluOp

  // lc = getNameLoc(builder, "relu");
  // auto relu_op = builder.create<ReluOp>(lc, output, conv_op);

  // // GlobalAveragePoolOp
  // lc = getNameLoc(builder, "GlobalAveragePool");
  // auto global_average_pool_op = builder.create<GlobalAveragePoolOp>(lc, output, relu_op);

  // // FlattenOp
  // lc = getNameLoc(builder, "Flatten");
  // auto flatten_op = builder.create<FlattenOp>(lc, output, global_average_pool_op);

  // // GemmOp
  // auto alpha = builder.getF64FloatAttr(1.0);
  // auto beta = builder.getF64FloatAttr(1.0);
  // auto transA = builder.getI64IntegerAttr(0);
  // auto transB = builder.getI64IntegerAttr(0);
  // lc = getNameLoc(builder, "Gemm");
  // auto gemm_op = builder.create<GemmOp>(lc, otype, flatten_op, add_op, add_op, alpha, beta, transA, transB);

  // // 构造ReturnOp，和funcop的返回值类型一致
  // mlir::ValueRange retop = {gemm_op};
  // lc = getNameLoc(builder, "end");
  // builder.create<mlir::func::ReturnOp>(lc, retop);
  builder.create<mlir::func::ReturnOp>(lc);
  // 设置插入点，funcop到此body结束
  builder.setInsertionPointToEnd(efblock);
  builder.setInsertionPointToEnd(theModule.getBody());

  // Apply Pass
  module = theModule;
  mlir::PassManager pm(module.get()->getName());
  pm.addPass(mlir::EX::createConversionPass());

  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(mlir::EX::createShapeInferencePass());
  // optPM.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*module)))
    return 4;

  // save file
  std::error_code ec;
  llvm::StringRef filename{"output.mlir"};
  llvm::raw_fd_ostream file(filename, ec);
  module->print(file);
  return 0;
}

int main(int argc, char **argv)
{
  // Test();
  Test2();
  return 0;
}
