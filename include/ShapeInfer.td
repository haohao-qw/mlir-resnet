#ifndef EX_MLIR_SHAPEINTERFACE
#define EX_MLIR_SHAPEINTERFACE

include "mlir/IR/OpBase.td"

def ShapeInterface : OpInterface<"ShapeInterface"> {
  let description = [{
  }];
//   let cppNamespace = "mlir::EX";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"shape_inference",
        /*args=*/(ins)
      >,
  ];
}
#endif // EX_MLIR_SHAPEINTERFACE