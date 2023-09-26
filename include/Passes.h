#ifndef EX_PASSES_H
#define EX_PASSES_H

#include <memory>

namespace mlir
{
    class Pass;

    namespace EX
    {
        std::unique_ptr<Pass> createShapeInferencePass();

        std::unique_ptr<mlir::Pass> createConversionPass();
    } // namespace t
} // namespace mlir

#endif // EX_PASSES_H
