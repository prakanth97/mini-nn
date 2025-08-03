//===- MLIRGen.h - MLIR Generation from NN AST -----------------*- C++ -*-===//
//
// This file declares the MLIRGen interface for converting NN AST to MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRGEN_H
#define MLIRGEN_H

#include "mlir/IR/BuiltinOps.h"
#include <memory>

// Forward declare AST nodes
class Program;

namespace mlir {
namespace nn {

/// Emit IR for the given NN program, returns a newly created module or nullptr on failure.
OwningOpRef<ModuleOp> mlirGen(MLIRContext &context, Program &program);

} // namespace nn
} // namespace mlir

#endif // MLIRGEN_H
