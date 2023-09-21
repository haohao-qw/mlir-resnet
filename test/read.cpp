#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
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

#include "llvm/ADT/ScopedHashTable.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialect.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <string>

#include <unistd.h>

using namespace mlir;
using namespace mlir::EX;
namespace cl = llvm::cl;

void print_op(mlir::Operation &op, unsigned level);
void print_region(mlir::Region &region, unsigned level);
void print_block(mlir::Block &block, unsigned level);

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

// get方法，获取op的attr，其原理都是通过对op获取到的attr进行转换为具体类型，然后获取到构造时传入的值。
static void getOpAttr(mlir::Operation &op, const std::string &opname, const std::string &attrname)
{
  std::string curnode = op.getName().getStringRef().str();
  if (opname != curnode)
    return;

  auto attr = op.getAttr(attrname);
  if (attr)
  {
    // 获取该属性的值，转换为create的类型，如constant是DenseElementsAttr
    auto denseAttr = attr.dyn_cast<mlir::DenseElementsAttr>();
    // 转为一个范围
    auto tensor = denseAttr.getValues<mlir::FloatAttr>();
    // 获取shape
    auto shape = denseAttr.getType().cast<mlir::RankedTensorType>().getShape();
    // 获取类型
    auto type = denseAttr.getType().cast<mlir::RankedTensorType>().getElementType();

    // shape
    llvm::outs() << "shape: ";
    for (auto dim : shape)
    {
      llvm::outs() << dim << " ";
    }
    // 打印类型
    llvm::outs() << "\n type:" << type << "\n";
    auto size = denseAttr.getType().cast<mlir::RankedTensorType>().getNumElements();
    for (int i = 0; i < size; i++)
    {
      // 可能会有精度丢失，float转double，0.1可能变成0.100000000001
      auto value = tensor[i].getValueAsDouble();
      llvm::outs() << value << " ";
    }
    llvm::outs() << "\n";
  }
}

static void getStrAttr(mlir::Operation &op, const std::string &opname, const std::string &attrname)
{
  // 当前node名称，
  std::string curnode = op.getName().getStringRef().str();
  if (opname != curnode)
    return;

  auto attr = op.getAttr(attrname);
  if (attr)
  {
    auto strAttr = attr.dyn_cast<mlir::StringAttr>();
    auto str = strAttr.getValue();
    llvm::outs() << "str: " << str << "\n";
  }
}

static void getI64ArrayAttr(mlir::Operation &op, const std::string &opname, const std::string &attrname)
{
  std::string curnode = op.getName().getStringRef().str();
  if (opname != curnode)
    return;

  auto attr = op.getAttr(attrname);
  if (attr)
  {
    auto arrayAttr = attr.dyn_cast<mlir::ArrayAttr>();
    auto array = arrayAttr.getValue();
    llvm::outs() << "array: ";
    for (auto i : array)
    {
      auto i64Attr = i.dyn_cast<mlir::IntegerAttr>();
      auto i64 = i64Attr.getInt();
      llvm::outs() << i64 << " ";
    }
    llvm::outs() << "\n";
  }
}

static void getSI64Attr(mlir::Operation &op, const std::string &opname, const std::string &attrname)
{
  std::string curnode = op.getName().getStringRef().str();
  if (opname != curnode)
    return;

  auto attr = op.getAttr(attrname);
  if (attr)
  {
    auto i64Attr = attr.dyn_cast<mlir::IntegerAttr>();
    auto i64 = i64Attr.getSInt();
    llvm::outs() << "i64: " << i64 << "\n";
  }
}

// op入参和出参的类型，如conv的输入是tensor，输出也是tensor，或者f32等等
static void getOpType(mlir::Operation &op)
{
  size_t op_num = op.getNumOperands();
  for (size_t i = 0; i < op_num; i++)
  {
    auto operand = op.getOperand(i);
    auto type = operand.getType();
    llvm::outs() << "operand shape: " << type << "\n";
  }
  // 返回的type
  size_t result_num = op.getNumResults();
  for (size_t i = 0; i < result_num; i++)
  {
    auto result = op.getResult(i);
    auto type = result.getType();
    llvm::outs() << "result shape: " << type << "\n";
  }
}

void print_op(mlir::Operation *op, unsigned level = 0)
{
  for (unsigned i = 0; i < level; ++i)
  {
    llvm::outs() << "  ";
  }
  unsigned region_num = op->getNumRegions();
  llvm::outs() << " -> Visting op: " << op->getName() << " with " << region_num << " region(s)"
               << "\n";

  // 获取op的attr具体信息
  getOpAttr(*op, "EX.Constant", "value");
  getStrAttr(*op, "EX.Conv", "auto_pad");
  getI64ArrayAttr(*op, "EX.Conv", "dilations");
  getSI64Attr(*op, "EX.Conv", "group");
  getI64ArrayAttr(*op, "EX.Conv", "kernel_shape");
  getI64ArrayAttr(*op, "EX.Conv", "pads");
  getI64ArrayAttr(*op, "EX.Conv", "strides");
  getOpType(*op);
  for (mlir::Region &region : op->getRegions())
  {
    print_region(region, level + 1);
  }
}

void print_region(mlir::Region &region, unsigned level = 0)
{
  for (unsigned i = 0; i < level; ++i)
  {
    llvm::outs() << "  ";
  }
  size_t block_num = region.getBlocks().size();
  llvm::outs() << " -> Region with " << block_num << " block(s)"
               << "\n";
  for (mlir::Block &block : region.getBlocks())
  {
    print_block(block, level + 1);
  }
}

void print_block(mlir::Block &block, unsigned level = 0)
{
  for (unsigned i = 0; i < level; ++i)
  {
    llvm::outs() << "  ";
  }
  size_t op_num = block.getOperations().size();
  llvm::outs() << " -> Block with " << op_num << " op(s), with successor(s) " << block.getNumSuccessors() << "\n";
  for (mlir::Operation &op : block.getOperations())
  {
    print_op(&op, level + 1);
  }
}

// 读取mlir文件，如果能正常被解析，说明mlir文件是正确的。
static void loadMLIR(const std::string inputFilename, mlir::MLIRContext &context,
                     mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input)
  {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module)
  {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    exit(1);
  }
}

int main(int argc, char **argv)
{
  // 读取测试
  mlir::MLIRContext rcontext;
  RegisterDialect(rcontext);
  mlir::OwningOpRef<mlir::ModuleOp> rmodule;
  loadMLIR("output.mlir", rcontext, rmodule);
  print_op(rmodule.get());

  return 0;
}
