#ifndef OPS
#define OPS

include "Dialect.td"
include "ShapeInfer.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AnyTensorOrNone : AnyTypeOf<[AnyTensor, NoneType]>;

class Ex_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<EX_Dialect, mnemonic, traits>;


def ConstantOp : Ex_BaseOp<"Constant"> {
    let summary = "Constant OP";
    let description = [{
    }];
    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

class Ex_Op<string mnemonic,list<Trait> traits=[]>:
      Ex_BaseOp<mnemonic,!listconcat(traits,
          [DeclareOpInterfaceMethods<ShapeInterface>])>;

def ConvOp : Ex_Op<"Conv"> {
    let summary = "Conv OP";
    let description = [{
    }];
    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$weight,
        AnyTensorOrNone:$bias,
        I64ArrayAttr:$stride,
        I64ArrayAttr:$kernel_shape,
        I64ArrayAttr:$padding,
        DefaultValuedAttr<I64Attr, "1">:$group,
        I64ArrayAttr:$dilations,
        DefaultValuedAttr<StrAttr, "\"NOTSET\"">:$auto_pad
    );
    let results = (outs AnyTensor:$output);
}

def ReluOp: Ex_Op<"Relu"> {
    let summary = "Relu OP";
    let description = [{
    }];
    let arguments = (ins
        AnyTensor:$input
    );
    let results = (outs AnyTensor:$output);
}


def MaxPoolOp : Ex_Op<"MaxPool"> {
    let summary = "MaxPool OP";
    let description = [{
    }];
    let arguments = (ins
        AnyTensor:$input,
        DefaultValuedAttr<StrAttr, "\"NOTSET\"">:$auto_pad,
        DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$kernel_shape,
        DefaultValuedAttr<I64Attr, "2">:$p,
        I64ArrayAttr:$pads,
        DefaultValuedAttr<I64Attr, "0">:$storage_order,
        I64ArrayAttr:$strides
    );
    let results = (outs
        AnyTensor:$output,
        AnyTensorOrNone:$indices
    );
}

def AddOp : Ex_Op<"Add"> {
    let summary = "Add OP";
    let description =[{
    }];
    let arguments = (ins
        AnyTensor:$input1,
        AnyTensor:$input2
    );
    let results = (outs AnyTensor:$output);
}

def GlobalAveragePoolOp: Ex_Op<"GlobalAveragePool"> {
    let summary = "GlobalAveragePool OP";
    let description = [{
    }];
    let arguments = (ins AnyTensor:$inputs);
    let results = (outs AnyTensor:$output);
}

def FlattenOp: Ex_Op<"Flatten"> {
    let summary = "Flatten OP";
    let description = [{
    }];
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def GemmOp: Ex_Op<"Gemm"> {
    let summary = "Gemm OP";
    let description = [{
    }];
    let arguments = (ins
        AnyTensor:$A,
        AnyTensor:$B,
        AnyTensorOrNone:$C,
        DefaultValuedAttr<F64Attr, "1.0">:$alpha,
        DefaultValuedAttr<F64Attr, "1.0">:$beta,
        DefaultValuedAttr<I64Attr, "0">:$transA,
        DefaultValuedAttr<I64Attr, "0">:$transB
    );
    let results = (outs AnyTensor:$output);
}

#endif //
