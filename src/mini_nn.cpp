#include <cstdio>
#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "parser/parser.h"
#include "dialect/MLIRGen.h"
#include "dialect/NNDialect.h"
#include "passes/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

int main(int argc, char** argv) {
  if (argc == 2) {
    pFile = fopen(argv[1], "r");
    if (pFile == NULL)
      perror("Error opening file");
  } else {
    std::cout << "Usage: ./mini_nn InputFile\n";
    return 1;
  }

  // Parse the input program
  Parser parser;
  auto program = parser.parseProgram();
  fclose(pFile);
  
  if (!program) {
    std::cerr << "Failed to parse program" << std::endl;
    return 1;
  }
  
  std::cout << "=== AST ===\n";
  std::cout << program->toString() << std::endl;
  
  // Create MLIR context and register our dialect
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::nn::NNDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  
  // Generate MLIR from AST
  auto module = mlir::nn::mlirGen(context, *program);
  if (!module) {
    std::cerr << "Failed to generate MLIR" << std::endl;
    return 1;
  }
  
  std::cout << "\n=== MLIR (Before Shape Inference) ===\n";
  module->print(llvm::outs());
  std::cout << std::endl;
  
  // Create a pass manager and add the shape inference pass
  mlir::PassManager pm(&context);
  
  // Since ShapeInferencePass operates on FuncOp, we need to use a nested pass manager
  pm.addNestedPass<mlir::nn::FuncOp>(mlir::nn::createShapeInferencePass());
  
  // Run the shape inference pass
  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Failed to run shape inference pass" << std::endl;
    return 1;
  }
  
  std::cout << "\n=== MLIR (After Shape Inference) ===\n";
  module->print(llvm::outs());
  std::cout << std::endl;
  
  return 0;
}
