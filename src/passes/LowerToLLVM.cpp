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

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
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

struct LinalgToLLVMPass 
  : public PassWrapper<LinalgToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToLLVMPass)
  
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

    //////////// Bufferization Phase ////////////

    // Build a sub-pipeline on ModuleOp and run it.
    OpPassManager pm(ModuleOp::getOperationName());

    // 1) One-Shot Bufferize (module-level)
    bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    pm.addPass(bufferization::createOneShotBufferizePass(options));

    // 2) Optional cleanup before deallocation (module-level)
    pm.addPass(bufferization::createDropEquivalentBufferResultsPass());

    // 3) Insert deallocations (function-level)
    pm.nest<func::FuncOp>().addPass(bufferization::createBufferDeallocationPass());

    // 4) Optional placement/cleanup (function-level)
    pm.nest<func::FuncOp>().addPass(bufferization::createPromoteBuffersToStackPass());
    pm.nest<func::FuncOp>().addPass(bufferization::createBufferHoistingPass());
    pm.nest<func::FuncOp>().addPass(bufferization::createBufferLoopHoistingPass());

    /////////// Affine Loops /////////////////
    pm.nest<func::FuncOp>().addPass(createConvertLinalgToAffineLoopsPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.nest<func::FuncOp>().addPass(affine::createAffineParallelizePass());
    pm.nest<func::FuncOp>().addPass(createLowerAffinePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    /////////// Lower to OpenMP ///////////////
    pm.addPass(createBufferizationToMemRefPass());
    pm.addPass(createConvertSCFToOpenMPPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    ////////// OpenMp to LLVM ////////////////
    pm.addPass(createConvertOpenMPToLLVMPass());
    pm.addPass(createCanonicalizerPass());
    pm.nest<func::FuncOp>().addPass(createLowerAffinePass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());

    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createArithToLLVMConversionPass());

    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    if (failed(runPipeline(pm, module)))
      signalPassFailure();
    
 }
  
  llvm::StringRef getArgument() const override {
    return "lower-to-llvm";
  }

  llvm::StringRef getDescription() const override {
    return "Lower Linalg dialect to llvm";
  }
};

namespace mlir {
namespace nn {

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LinalgToLLVMPass>();
}

}
}
