//===- LowerToLinalgDialect.cpp - Lower NN Dialect to Linalg -----------===//
//
// This file implements the conversion from NN dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

using namespace mlir;
using namespace mlir::nn;

struct LoopsToGPU 
  : public PassWrapper<LoopsToGPU, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopsToGPU)
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect,
                    memref::MemRefDialect,
                    affine::AffineDialect,
                    scf::SCFDialect,
                    omp::OpenMPDialect,
                    cf::ControlFlowDialect,
                    LLVM::LLVMDialect>();
    
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    memref::registerAllocationOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Build a sub-pipeline on ModuleOp and run it.
    OpPassManager pm(ModuleOp::getOperationName());

    /////////// Lower to GPU ///////////////
    pm.addPass(createBufferizationToMemRefPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // ////////// GPU to LLVM ////////////////
    // pm.addPass(createCanonicalizerPass());
    // pm.nest<func::FuncOp>().addPass(createLowerAffinePass());
    // pm.addPass(memref::createExpandStridedMetadataPass());
    // pm.addPass(createFinalizeMemRefToLLVMConversionPass());

    // pm.addPass(createConvertSCFToCFPass());
    // pm.addPass(createConvertControlFlowToLLVMPass());
    // pm.addPass(createLowerAffinePass());
    // pm.addPass(createArithToLLVMConversionPass());

    // pm.addPass(createConvertMathToLLVMPass());
    // pm.addPass(createConvertFuncToLLVMPass());
    // pm.addPass(createReconcileUnrealizedCastsPass());
    
    if (failed(runPipeline(pm, module)))
      signalPassFailure();
    
 }
  
  llvm::StringRef getArgument() const override {
    return "lower-from-affine-to-llvm";
  }

  llvm::StringRef getDescription() const override {
    return "Lower Affine dialect to llvm";
  }
};

namespace mlir {
namespace nn {

std::unique_ptr<mlir::Pass> createLowerLoopsToGPUPass() {
  return std::make_unique<LoopsToGPU>();
}

}
}
