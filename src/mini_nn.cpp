#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
// #include "mlir/Dialect/Bufferization/IR/Bufferization.h"
// #include "mlir/IR/BuiltinAttributes.h"
#include "dialect/MLIRGen.h"
#include "dialect/NNDialect.h"
#include "parser/parser.h"
#include "passes/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
// #include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
// #include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// #include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
// #include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
// #include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
// #include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
// #include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include
// "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
// #include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
// #include "mlir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

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
  // context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  // context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  // context.getOrLoadDialect<mlir::scf::SCFDialect>();
  // context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  // context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();

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

  // Since ShapeInferencePass operates on FuncOp, we need to use a nested pass
  // manager
  pm.addNestedPass<mlir::nn::FuncOp>(mlir::nn::createShapeInferencePass());

  // Run the shape inference pass
  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Failed to run shape inference pass" << std::endl;
    return 1;
  }

  std::cout << "\n=== MLIR (After Shape Inference) ===\n";
  module->print(llvm::outs());
  std::cout << std::endl;

  // Add lowering to Linalg pass
  mlir::PassManager loweringPM(&context);
  loweringPM.addPass(mlir::nn::createLowerToLinalgPass());

  // Run the lowering pass
  if (mlir::failed(loweringPM.run(*module))) {
    std::cerr << "Failed to run lowering pass" << std::endl;
    return 1;
  }

  std::cout << "\n=== MLIR (After Lowering to Linalg) ===\n";
  module->print(llvm::outs());
  std::cout << std::endl;

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

  // Define MLIR-opt path - update this path to match your LLVM build
  std::string mlir_opt_path =
      "/Users/u5624836/Desktop/PhD/repos/llvm-project/build/bin/mlir-opt";

  // Lower to SCF dialect
  std::string bufferized_file =
      location_prefix + base_filename + "_bufferized.mlir";
  std::string mlir_passes_cmd =
      mlir_opt_path + " " + linalg_file +
      " --one-shot-bufferize=\"bufferize-function-boundaries\"" +
      " -buffer-hoisting" + " -buffer-loop-hoisting" +
      " -buffer-results-to-out-params" + " -drop-equivalent-buffer-results" +
      " -promote-buffers-to-stack" + " -buffer-deallocation-pipeline" +
      " -convert-linalg-to-parallel-loops" + " -canonicalize" + " -o " +
      bufferized_file;

  std::cout << "\nRunning MLIR bufferization passes..." << std::endl;
  std::cout << "Command: " << mlir_passes_cmd << std::endl;
  int result1 = std::system(mlir_passes_cmd.c_str());
  if (result1 != 0) {
    std::cerr << "Failed to run MLIR bufferization passes" << std::endl;
    return 1;
  }
  std::cout << "Successfully generated: " << bufferized_file << std::endl;

  // Lower to LLVM IR through OMP dialect
  std::string llvm_file = location_prefix + base_filename + "_llvm.mlir";
  std::string lower_passes_cmd =
      mlir_opt_path + " " + bufferized_file +
      " -convert-bufferization-to-memref" + " -convert-scf-to-openmp" +
      " -canonicalize" + " -cse" + " -convert-openmp-to-llvm" +
      " -canonicalize" + " -lower-affine" + " -expand-strided-metadata" +
      " -finalize-memref-to-llvm" + " -convert-scf-to-cf" +
      " -convert-cf-to-llvm" + " -convert-to-llvm" + " -lower-affine" +
      " -convert-arith-to-llvm" + " -reconcile-unrealized-casts" + " -o " +
      llvm_file;

  std::cout << "\nRunning MLIR lowering passes..." << std::endl;
  std::cout << "Command: " << lower_passes_cmd << std::endl;
  int result2 = std::system(lower_passes_cmd.c_str());
  if (result2 != 0) {
    std::cerr << "Failed to run MLIR lowering passes" << std::endl;
    return 1;
  }
  std::cout << "Successfully generated: " << llvm_file << std::endl;

  // Optionally read and display the final LLVM IR
  std::ifstream llvm_ir_file(llvm_file);
  if (llvm_ir_file.is_open()) {
    std::cout << "\n=== MLIR (After Lowering to LLVM) ===\n";
    std::string line;
    while (std::getline(llvm_ir_file, line)) {
      std::cout << line << std::endl;
    }
    llvm_ir_file.close();
  }
  return 0;
}
