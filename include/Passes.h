#ifndef EX_PASSES_H_
#define EX_PASSES_H_

#include <memory>

namespace mlir
{
    class Pass;
    namespace EX
    {
        std::unique_ptr<Pass> createShapeInferencePass();

        std::unique_ptr<Pass> createConversionPass();
    } // namespace t
} // namespace mlir

#endif // EX_PASSES_H
