//===- NNDialect.cpp - NN dialect --------------------------------*- C++ -*-===//
//
// This file implements the NN dialect.
//
//===----------------------------------------------------------------------===//

#include "NNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/Casting.h"

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

//===----------------------------------------------------------------------===//
// Type and Attribute parsing/printing (using defaults)
//===----------------------------------------------------------------------===//

mlir::Type NNDialect::parseType(mlir::DialectAsmParser &parser) const {
  // For now, just return nullptr as we don't have custom types
  return nullptr;
}

void NNDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  // For now, do nothing as we don't have custom types
}

mlir::Attribute NNDialect::parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const {
  // For now, just return nullptr as we don't have custom attributes
  return nullptr;
}

void NNDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {
  // For now, do nothing as we don't have custom attributes
}

//===----------------------------------------------------------------------===//
// DenseOp 
//===----------------------------------------------------------------------===//

void DenseOp::inferShapes() {
  // Get the input and weight tensor types
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = llvm::dyn_cast<RankedTensorType>(getWeight().getType());

  // Get the input and weight shapes
  auto inputShape = inputType.getShape();
  auto weightShape = weightType.getShape();
  
  // For dense operation: output = input * weight^T + bias
  // Input shape: [batch_size, input_features]
  // Weight shape: [input_features, output_features]
  // Output shape: [batch_size, output_features]
  
  if (inputShape.size() < 2 || weightShape.size() != 2) {
    return; // Invalid shapes, let the verifier handle this
  }
  
  // Create the output shape: [batch_size, output_features]
  SmallVector<int64_t> outputShape;
  outputShape.append(inputShape.begin(), inputShape.end() - 1); // All dims except last
  outputShape.push_back(weightShape[1]); // output_features from weight
  
  // Create the output tensor type
  auto elementType = inputType.getElementType();
  auto outputType = RankedTensorType::get(outputShape, elementType);
  
  // Set the inferred type on the result
  getResult().setType(outputType);
}

//===----------------------------------------------------------------------===//
// Conv1D
//===----------------------------------------------------------------------===//

void Conv1DOp::inferShapes() {
  // Get the input and weight tensor types
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType()); 
  auto kernelType = llvm::dyn_cast<RankedTensorType>(getKernel().getType()); 
  
  // Get the input and weight shapes
  auto inputShape = inputType.getShape();
  auto kernelShape = kernelType.getShape();
  
  // For Conv1D operation: output = conv1d(input, weight) + bias
  // Input shape: [batch_size, input_length, input_channels]
  // Weight shape: [kernel_size, input_channels, output_channels]
  // Output shape: [batch_size, output_length, output_channels]
  
  if (inputShape.size() != 3 || kernelShape.size() != 3) {
    return; // Invalid shapes, let the verifier handle this
  }
  
  // Calculate output length based on convolution formula
  int64_t outputLength = inputShape[1] - kernelShape[0] + 1;
  
  // Create the output shape: [batch_size, output_length, output_channels]
  SmallVector<int64_t> outputShape = {inputShape[0], outputLength, kernelShape[2]};
  
  // Create the output tensor type
  auto elementType = inputType.getElementType();
  auto outputType = RankedTensorType::get(outputShape, elementType);
  
  // Set the inferred type on the result
  getResult().setType(outputType);
}

//===----------------------------------------------------------------------===//
// Conv2D
//===----------------------------------------------------------------------===//

void Conv2DOp::inferShapes() {
  // Get the input and weight tensor types
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType()); 
  auto kernelType = llvm::dyn_cast<RankedTensorType>(getKernel().getType());
  
  // Get the input and weight shapes
  auto inputShape = inputType.getShape();
  auto kernelShape = kernelType.getShape();
  
  // For Conv2D operation: output = conv2d(input, weight) + bias
  // Input shape: [batch_size, height, width, input_channels]
  // Weight shape: [kernel_height, kernel_width, input_channels, output_channels]
  // Output shape: [batch_size, output_height, output_width, output_channels]
  
  if (inputShape.size() != 4 || kernelShape.size() != 4) {
    return; // Invalid shapes, let the verifier handle this
  }
  
  // Calculate output height and width based on convolution formula
  int64_t outputHeight = inputShape[1] - kernelShape[0] + 1;
  int64_t outputWidth = inputShape[2] - kernelShape[1] + 1;
  
  // Create the output shape: [batch_size, output_height, output_width, output_channels]
  SmallVector<int64_t> outputShape = {inputShape[0], outputHeight, outputWidth, kernelShape[3]};
  
  // Create the output tensor type
  auto elementType = inputType.getElementType();
  auto outputType = RankedTensorType::get(outputShape, elementType);
  
  // Set the inferred type on the result
  getResult().setType(outputType);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::inferShapes() {
  // Get the input tensor type
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  
  // Get the input shape
  auto inputShape = inputType.getShape();
  
  // For transpose operation: output = transpose(input)
  // Transpose swaps the last two dimensions of the tensor
  
  if (inputShape.size() < 2) {
    return; // Invalid shape, let the verifier handle this
  }
  
  // Create the output shape by swapping the last two dimensions
  SmallVector<int64_t> outputShape(inputShape);
  std::swap(outputShape[outputShape.size() - 1], outputShape[outputShape.size() - 2]);
  
  // Create the output tensor type
  auto elementType = inputType.getElementType();
  auto outputType = RankedTensorType::get(outputShape, elementType);
  
  // Set the inferred type on the result
  getResult().setType(outputType);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

void MatmulOp::inferShapes() {
  // Get the input tensor types
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  
  // Get the input shapes
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  
  // For matmul operation: output = matmul(lhs, rhs)
  // LHS shape: [batch_size, m, n]
  // RHS shape: [batch_size, n, p]
  // Output shape: [batch_size, m, p]
  
  if (lhsShape.size() != 3 || rhsShape.size() != 3 || lhsShape[0] != rhsShape[0]) {
    return; // Invalid shapes, let the verifier handle this
  }
  
  // Create the output shape: [batch_size, m, p]
  SmallVector<int64_t> outputShape = {lhsShape[0], lhsShape[1], rhsShape[2]};
  
  // Create the output tensor type
  auto elementType = lhsType.getElementType();
  auto outputType = RankedTensorType::get(outputShape, elementType);
  
  // Set the inferred type on the result
  getResult().setType(outputType);
}


#define GET_OP_CLASSES
#include "ops.cpp.inc"
