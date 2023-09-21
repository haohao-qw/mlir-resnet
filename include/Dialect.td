#ifndef EX_DIALECT
#define EX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Pass/PassBase.td"

def EX_Dialect : Dialect {
  let name = "EX";
  let cppNamespace = "::mlir::EX";
  let dependentDialects = ["mlir::func::FuncDialect"];
}


#endif // EX_DIALECT
