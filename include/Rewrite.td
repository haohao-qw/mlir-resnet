#ifndef EX_REWRITE
#define EX_REWRITE

include "mlir/IR/PatternBase.td"
include "Ops.td"

// 第一个是要匹配的模式，第二个是替换的模式
// def TestOpPattern : Pat<(TestOp $arg),
//                                    (ShowOp $arg)>;

// 添加一些约束条件
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def AddOpPattern : Pat<(AddOp $arg1,$arg2),
                                   (AddOp $arg2,$arg1),
                                   [(TypesAreIdentical $arg1, $arg2)]>;

#endif // EX_WRITE