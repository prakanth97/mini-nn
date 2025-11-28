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
std::unique_ptr<mlir::Pass> createLowerToLinalgPass();
std::unique_ptr<mlir::Pass> createLowerToLoopsPass();
std::unique_ptr<mlir::Pass> createLowerLoopsToCPUPass();
std::unique_ptr<mlir::Pass> createLowerLoopsToGPUPass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
std::unique_ptr<mlir::Pass> createGPUMemoryTransferPass();
std::unique_ptr<mlir::Pass> createGPUHostRegister();

} // namespace nn
} // namespace mlir

#endif // NN_PASSES_H
