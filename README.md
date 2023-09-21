## ex1:约束和属性 
### 约束：  
https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints  
概念：限制和规范的表示。如限制某个值为大于5的int32类型的数值或者限制某个数值为FP32；限制多个操作数的结果类型必须和第一个操作数的类型相同；operations的内在属性。  
以上三种分别对应单实体约束、多实体约束和特征。  
单实体约束通过继承TypeConstraint和AttrConstraints用于定义约束。
多实体约束通过继承PredOpTrait定义约束。  
特征通过继承NativeOpTrait定义约束。  
指定新的约束：  
通过谓词Cpred进行描述，通过(conjunction: And, disjunction: Or, negation: Neg, substitution: SubstLeaves, concatenation: Concat）构造复杂谓词。Cpred成立时约束才生效，否则不起作用。  
也可以通过将任何返回bool类型的C++表达式放到cpred中进行构建。为了和C++进行交互，用$_builder, $_op, and $_self充当勾子。在谓词非常复杂的情况下，可以通过C++函数进行描述声明，CPred充当调用的角色进行简化编写。  

### 属性：
概念：属性是操作的编译时已知常量,粒度更细的约束。  
属性装饰器用于指定附加属性，如DefaultValuedAttr（为属性指定一个默认值）、OptionalAttr（指定属性是可选的）、Confined（将原始约束构造为更复杂的约束）。  
枚举属性：某些属性只能从预定义的枚举中获取，ods提供IntEnumAttr、BitEnumAttr进行定义。

# mlir-resnet
