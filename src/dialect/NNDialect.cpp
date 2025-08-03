//===- NNDialect.cpp - NN dialect --------------------------------*- C++ -*-===//
//
// This file implements the NN dialect.
//
//===----------------------------------------------------------------------===//

#include "NNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::nn;

#include "dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// NN dialect.
//===----------------------------------------------------------------------===//

void NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ops.cpp.inc"
      >();
}

// //===----------------------------------------------------------------------===//
// // Type and Attribute parsing/printing (using defaults)
// //===----------------------------------------------------------------------===//

// mlir::Type NNDialect::parseType(mlir::DialectAsmParser &parser) const {
//   // For now, just return nullptr as we don't have custom types
//   return nullptr;
// }

// void NNDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
//   // For now, do nothing as we don't have custom types
// }

// mlir::Attribute NNDialect::parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const {
//   // For now, just return nullptr as we don't have custom attributes
//   return nullptr;
// }

// void NNDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {
//   // For now, do nothing as we don't have custom attributes
// }

// #define GET_OP_CLASSES
// #include "ops.cpp.inc"
