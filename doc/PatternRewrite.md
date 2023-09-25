### match and write
DRR方式：  
```cpp
def OptPattern : Pat<(SourceOpName(SourceOpName $arg)),
                                   (ResultOpName $arg)>;
```
SourceOpName和ResultOpName都是Op的名字，$arg是Op的参数。第一个是匹配模式，第二个是重写模式。第三个可以是约束条件，只要在为真时才会进行重写。  

```cpp
class Pat<
    dag sourcePattern, dag resultPattern,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)> :
  Pattern<sourcePattern, [resultPattern], additionalConstraints, benefitAdded>;
```
sourcePattern代表用来匹配的模式，resultPattern是改写后的模式，他们的格式都是 dag ，即 (operator arg0, arg1, ...) ；additionalConstraints用于增加额外限制条件，benefitsAdded是默认给了该匹配模式一个优先级。
Pat是个单输出的模式，它继承自Pattern，Pattern是个多输出的模式：
```cpp
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
```

添加约束条件：
```cpp
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

调用NativeCodeCall
```cpp
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

参考链接：https://www.zhihu.com/people/xaioyuxu/posts