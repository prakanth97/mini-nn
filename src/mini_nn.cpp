#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "dialect/MLIRGen.h"
#include "dialect/NNDialect.h"
#include "parser/parser.h"
#include "passes/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

int main(int argc, char **argv) {
  if (argc == 2) {
    pFile = fopen(argv[1], "r");
    if (pFile == NULL)
      perror("Error opening file");
  } else {
    std::cout << "Usage: ./mini_nn InputFile\n";
    return 1;
  }

  // Extract directory path from input file for output files
  std::string input_file_path(argv[1]);
  std::string location_prefix = "";
  size_t last_slash = input_file_path.find_last_of('/');
  if (last_slash != std::string::npos) {
    location_prefix =
        input_file_path.substr(0, last_slash + 1); // Include the trailing slash
  }

  // Extract base filename without extension for naming output files
  std::string base_filename = input_file_path;
  if (last_slash != std::string::npos) {
    base_filename = input_file_path.substr(last_slash + 1);
  }
  size_t dot_pos = base_filename.find_last_of('.');
  if (dot_pos != std::string::npos) {
    base_filename = base_filename.substr(0, dot_pos);
  }

  std::cout << "Input file: " << input_file_path << std::endl;
  std::cout << "Output directory: "
            << (location_prefix.empty() ? "./" : location_prefix) << std::endl;
  std::cout << "Base filename: " << base_filename << std::endl;

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

  // Create MLIR context and register all necessary dialects
  mlir::MLIRContext context;

  // Load specific dialects we need, without loadAllAvailableDialects() which
  // might cause issues
  context.getOrLoadDialect<mlir::nn::NNDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  // Generate MLIR from AST
  auto module = mlir::nn::mlirGen(context, *program);
  if (!module) {
    std::cerr << "Failed to generate MLIR" << std::endl;
    return 1;
  }

  llvm::outs() << "\n=== MLIR (Before Shape Inference) ===\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  // Create a pass manager and add the shape inference pass
  mlir::PassManager pm(&context);

  // Since ShapeInferencePass operates on FuncOp, we need to use a nested pass
  // manager
  pm.addNestedPass<mlir::nn::FuncOp>(mlir::nn::createShapeInferencePass());

  // Run the shape inference pass
  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Failed to run shape inference pass" << std::endl;
    return 1;
  }

  llvm::outs() << "\n=== MLIR (After Shape Inference) ===\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  // Add lowering to Linalg pass
  mlir::PassManager loweringPM(&context);
  loweringPM.addPass(mlir::nn::createLowerToLinalgPass());

  // Run the lowering pass
  if (mlir::failed(loweringPM.run(*module))) {
    std::cerr << "Failed to run lowering pass" << std::endl;
    return 1;
  }

  llvm::outs() << "\n=== MLIR (After Lowering to Linalg) ===\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  // Save the Linalg IR to a file for external processing
  std::string linalg_file = location_prefix + base_filename + "_linalg.mlir";
  std::error_code EC;
  llvm::raw_fd_ostream linalg_output(linalg_file, EC);
  if (EC) {
    std::cerr << "Failed to open file for writing: " << linalg_file
              << std::endl;
    return 1;
  }
  module->print(linalg_output);
  linalg_output.close();
  std::cout << "Saved Linalg IR to: " << linalg_file << std::endl;

  mlir::PassManager llvmPM(&context);
  llvmPM.addPass(mlir::nn::createLowerToLLVMPass());

  // Run the LLVM lowering pass
  if (mlir::failed(llvmPM.run(*module))) {
    std::cerr << "Failed to run LLVM lowering pass" << std::endl;
    return 1;
  }

  std::string llvm_file = location_prefix + base_filename + "_llvm.mlir";
  llvm::raw_fd_ostream llvm_output(llvm_file, EC);
  if (EC) {
    std::cerr << "Failed to open file for writing: " << llvm_file
              << std::endl;
    return 1;
  }

  module->print(llvm_output);
  llvm_output.close();
  std::cout << "Saved LLVM IR to: " << llvm_file << std::endl;

  llvm::outs() << "\n=== MLIR (After Lowering to LLVM) ===\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  // auto clonedModule = module->clone();
  // llvm::LLVMContext llvmContext;
  // std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(clonedModule, llvmContext);

  // if (!llvmModule) {
  //   std::cerr << "Failed to translate MLIR module to LLVM IR" << std::endl;
  //   return 1;
  // }

  // std::string llc_file = location_prefix + base_filename + ".ll";
  // llvm::raw_fd_ostream llc_output(llc_file, EC);
  // if (EC) {
  //   std::cerr << "Failed to open file for writing: " << llc_file 
  //             << std::endl;
  //   return 1;
  // }

  // llvmModule->print(llc_output, nullptr);
  // llc_output.close();

  return 0;
}
