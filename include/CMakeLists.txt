set(LLVM_TARGET_DEFINITIONS 
    Dialect.td
)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS 
    Ops.td
)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS ShapeInfer.td)
mlir_tablegen(ShapeInfer.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInfer.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS Rewrite.td)
mlir_tablegen(Rewrite.inc -gen-rewriters)

add_public_tablegen_target(ExTdGen)