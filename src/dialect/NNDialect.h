//===- NNDialect.h - NN dialect --------------------------------*- C++ -*-===//
//
// This file declares the NN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef NN_DIALECT_H
#define NN_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"

namespace mlir {
namespace nn {

// Forward declarations
class NNDialect;

} // namespace nn
} // namespace mlir

#include "dialect.h.inc"

#define GET_OP_CLASSES
#include "ops.h.inc"

#endif // NN_DIALECT_H
