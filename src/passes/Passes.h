//===- Passes.h - NN dialect passes --------------------------*- C++ -*-===//
//
// This file declares the passes for the NN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef NN_PASSES_H
#define NN_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nn {

// Pass declarations
std::unique_ptr<mlir::Pass> createShapeInferencePass();

} // namespace nn
} // namespace mlir

#endif // NN_PASSES_H
