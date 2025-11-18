//===- MLIRGen.cpp - MLIR Generation from NN AST ---------------*- C++ -*-===//
//
// This file implements the MLIRGen interface for converting NN AST to MLIR.
//
//===----------------------------------------------------------------------===//

#include "MLIRGen.h"
#include "../parser/ast.h"
#include "NNDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"

using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::dyn_cast;
using llvm::isa;
using llvm::StringRef;

namespace {

/// Implementation of a simple MLIR emission from the NN AST.
class MLIRGenImpl {
private:
  mlir::MLIRContext &context;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;

  /// A mapping for the functions that have been code generated
  llvm::StringMap<mlir::nn::FuncOp> functionMap;

  /// A mapping for named values to the operations that produce them.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  /// Layer definitions mapping layer names to their configuration
  llvm::StringMap<LayerDefinition *> layerMap;

public:
  MLIRGenImpl(mlir::MLIRContext &context)
      : context(context), builder(&context) {}

  /// Public API: convert the AST for a NN program to MLIR.
  mlir::ModuleOp mlirGen(Program &program) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    // Process layer definitions first
    for (auto &stmt : program.statements) {
      if (auto *layerDef = dyn_cast<LayerDefinition>(stmt.get())) {
        layerMap[layerDef->name] = layerDef;
      }
    }

    // Process function definitions
    for (auto &stmt : program.statements) {
      if (auto *funcDef = dyn_cast<FunctionDefinition>(stmt.get())) {
        mlir::nn::FuncOp func = mlirGen(*funcDef);
        if (!func)
          return nullptr;
        theModule.push_back(func);
      }
    }

    // Verify the module after we have finished constructing it
    if (failed(verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// Create an MLIR type from a NN AST type.
  mlir::Type getType(::Type *nnType) {
    if (nnType->primitiveType == "float") {
      if (nnType->dimensions.empty()) {
        return builder.getF32Type();
      } else {
        // Convert dimensions to MLIR tensor shape
        SmallVector<int64_t, 4> shape;
        for (const auto &dim : nnType->dimensions) {
          if (dim == "?") {
            shape.push_back(mlir::ShapedType::kDynamic);
          } else {
            shape.push_back(std::stoll(dim));
          }
        }
        return mlir::RankedTensorType::get(shape, builder.getF32Type());
      }
    }
    // Default to f32 for unknown types
    return builder.getF32Type();
  }

  /// Create layer parameters as function arguments
  SmallVector<mlir::Type> getLayerParameterTypes(LayerDefinition *layerDef) {
    SmallVector<mlir::Type> paramTypes;

    if (layerDef->layerType == "dense") {
      // Dense layer needs weight and bias parameters
      // Get dimensions from layer parameters
      if (layerDef->parameters.size() >= 2) {
        auto *inputDim =
            dyn_cast<NumberLiteral>(layerDef->parameters[0].get());
        auto *outputDim =
            dyn_cast<NumberLiteral>(layerDef->parameters[1].get());

        if (inputDim && outputDim) {
          // Weight matrix: [input_dim, output_dim]
          SmallVector<int64_t> weightShape = {std::stoll(inputDim->value),
                                              std::stoll(outputDim->value)};
          paramTypes.push_back(
              mlir::RankedTensorType::get(weightShape, builder.getF32Type()));

          // Bias vector: [output_dim]
          SmallVector<int64_t> biasShape = {std::stoll(outputDim->value)};
          paramTypes.push_back(
              mlir::RankedTensorType::get(biasShape, builder.getF32Type()));
        }
      }
    }

    return paramTypes;
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::nn::FuncOp mlirGen(FunctionDefinition &funcDef) {

    // Create the input types for the function.
    SmallVector<mlir::Type> argTypes;
    SmallVector<mlir::Location> argLocs;

    // Add original function parameters
    for (auto &param : funcDef.parameters) {
      argTypes.push_back(getType(param.first.get()));
      argLocs.push_back(builder.getUnknownLoc());
    }

    // Add layer parameters (weights and biases) for each layer used in the
    // function
    SmallVector<std::string> layerParamNames;
    for (auto &stmt : funcDef.body) {
      if (auto *assign = dyn_cast<Assignment>(stmt.get())) {
        if (auto *call =
                dyn_cast<FunctionCall>(assign->value.get())) {
          // Check if this is a layer call
          if (layerMap.find(call->functionName) != layerMap.end()) {
            LayerDefinition *layerDef = layerMap[call->functionName];
            auto layerTypes = getLayerParameterTypes(layerDef);
            for (size_t i = 0; i < layerTypes.size(); ++i) {
              argTypes.push_back(layerTypes[i]);
              argLocs.push_back(builder.getUnknownLoc());
              layerParamNames.push_back(call->functionName + "_param" +
                                        std::to_string(i));
            }
          }
        }
      }
    }

    // Create the function type
    auto funcType = mlir::FunctionType::get(&context, argTypes,
                                            getType(funcDef.returnType.get()));

    // Create the NN function by manually building the operation state
    // to avoid the ambiguous build method calls
    auto loc = builder.getUnknownLoc();
    auto funcName = builder.getStringAttr(funcDef.name);
    auto funcTypeAttr = mlir::TypeAttr::get(funcType);

    auto function =
        builder.create<mlir::nn::FuncOp>(loc, funcName, funcTypeAttr);

    // Add the function to our function map
    functionMap[funcDef.name] = function;

    // Create the function body
    auto &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);

    // Declare the function arguments in the symbol table
    size_t paramIndex = 0;
    for (auto &param : funcDef.parameters) {
      symbolTable.insert(param.second, entryBlock.getArgument(paramIndex++));
    }

    // Declare layer parameters in symbol table
    for (const auto &paramName : layerParamNames) {
      symbolTable.insert(paramName, entryBlock.getArgument(paramIndex++));
    }

    // Emit the body of the function
    for (auto &stmt : funcDef.body) {
      if (mlir::failed(mlirGen(*stmt))) {
        function.erase();
        return nullptr;
      }
    }

    return function;
  }

  /// Emit a statement, this can be a return statement or an assignment.
  mlir::LogicalResult mlirGen(Statement &stmt) {
    if (auto *ret = dyn_cast<ReturnStatement>(&stmt)) {
      return mlirGen(*ret);
    }
    if (auto *assign = dyn_cast<Assignment>(&stmt)) {
      return mlirGen(*assign);
    }

    return mlir::failure();
  }

  /// Emit a return statement
  mlir::LogicalResult mlirGen(ReturnStatement &ret) {
    auto value = mlirGen(*ret.value);
    if (!value)
      return mlir::failure();

    builder.create<mlir::nn::ReturnOp>(builder.getUnknownLoc(), value);
    return mlir::success();
  }

  /// Emit an assignment statement
  mlir::LogicalResult mlirGen(Assignment &assign) {
    auto value = mlirGen(*assign.value);
    if (!value)
      return mlir::failure();

    // Register the value in the symbol table
    symbolTable.insert(assign.variable, value);
    return mlir::success();
  }

  /// Emit an expression, returns the computed value.
  mlir::Value mlirGen(Expression &expr) {
    if (auto *ident = dyn_cast<Identifier>(&expr)) {
      return mlirGen(*ident);
    }
    if (auto *funcCall = dyn_cast<FunctionCall>(&expr)) {
      return mlirGen(*funcCall);
    }
    if (auto *numLit = dyn_cast<NumberLiteral>(&expr)) {
      return mlirGen(*numLit);
    }

    return nullptr;
  }

  /// Emit an identifier expression
  mlir::Value mlirGen(Identifier &ident) {
    if (auto variable = symbolTable.lookup(ident.name))
      return variable;

    return nullptr;
  }

  /// Emit a function call expression
  mlir::Value mlirGen(FunctionCall &call) {
    auto loc = builder.getUnknownLoc();

    // Check if this is a layer call
    if (layerMap.find(call.functionName) != layerMap.end()) {
      LayerDefinition *layerDef = layerMap[call.functionName];

      if (layerDef->layerType == "dense") {
        // Get the input argument
        if (call.arguments.empty())
          return nullptr;

        auto input = mlirGen(*call.arguments[0]);
        if (!input)
          return nullptr;

        // Get layer parameters from symbol table
        auto weight = symbolTable.lookup(call.functionName + "_param0");
        auto bias = symbolTable.lookup(call.functionName + "_param1");

        if (!weight || !bias)
          return nullptr;

        // Create dense operation using our NN dialect
        auto denseOp = builder.create<mlir::nn::DenseOp>(loc, input.getType(),
                                                         input, weight, bias);
        return denseOp.getResult();
      }
    }

    // TODO: Handle convolution layers.

    // Handle builtin activation functions
    if (call.isBuiltin) {
      if (call.arguments.empty())
        return nullptr;

      auto input = mlirGen(*call.arguments[0]);
      if (!input)
        return nullptr;

      if (call.functionName == "relu") {
        auto reluOp =
            builder.create<mlir::nn::ReluOp>(loc, input.getType(), input);
        return reluOp.getResult();
      } else if (call.functionName == "sigmoid") {
        auto sigmoidOp =
            builder.create<mlir::nn::SigmoidOp>(loc, input.getType(), input);
        return sigmoidOp.getResult();
      } else if (call.functionName == "tanh") {
        auto tanhOp =
            builder.create<mlir::nn::TanhOp>(loc, input.getType(), input);
        return tanhOp.getResult();
      } else if (call.functionName == "softmax") {
        auto softmaxOp =
            builder.create<mlir::nn::SoftmaxOp>(loc, input.getType(), input);
        return softmaxOp.getResult();
      } else if (call.functionName == "add") {
        if (call.arguments.size() < 2)
          return nullptr;

        auto lhs = mlirGen(*call.arguments[0]);
        auto rhs = mlirGen(*call.arguments[1]);
        if (!lhs || !rhs)
          return nullptr;

        auto addOp =
            builder.create<mlir::nn::AddOp>(loc, lhs.getType(), lhs, rhs);
        return addOp.getResult();
      } else if (call.functionName == "matmul") {
        if (call.arguments.size() < 2)
          return nullptr;

        auto lhs = mlirGen(*call.arguments[0]);
        auto rhs = mlirGen(*call.arguments[1]);
        if (!lhs || !rhs)
          return nullptr;

        auto matmulOp =
            builder.create<mlir::nn::MatmulOp>(loc, lhs.getType(), lhs, rhs);
        return matmulOp.getResult();
      } else if (call.functionName == "transpose") {
        if (call.arguments.empty())
          return nullptr;
        // Create transpose operation
        auto transposeOp =
            builder.create<mlir::nn::TransposeOp>(loc, input.getType(), input);
        return transposeOp.getResult();
      }
    }

    return nullptr;
  }

  /// Emit a number literal
  mlir::Value mlirGen(NumberLiteral &num) {
    if (num.isFloat) {
      auto value = builder.getF32FloatAttr(std::stof(num.value));
      return builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),
                                                     value);
    } else {
      auto value = builder.getI32IntegerAttr(std::stoi(num.value));
      return builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),
                                                     value);
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace nn {

// The public API for codegen.
OwningOpRef<ModuleOp> mlirGen(MLIRContext &context, Program &program) {
  return MLIRGenImpl(context).mlirGen(program);
}

} // namespace nn
} // namespace mlir
