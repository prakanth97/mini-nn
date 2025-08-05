#include <cstdio>
#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "parser/parser.h"
#include "dialect/MLIRGen.h"
#include "dialect/NNDialect.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"

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
  
  std::cout << "\n=== MLIR ===\n";
  module->print(llvm::outs());
  std::cout << std::endl;
  
  return 0;
}
