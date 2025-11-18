//===- LowerToLinalgDialect.cpp - Lower NN Dialect to Linalg -----------===//
//
// This file implements the conversion from NN dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

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
                    gpu::GPUDialect,
                    NVVM::NVVMDialect,
                    cf::ControlFlowDialect,
                    LLVM::LLVMDialect,
                    ub::UBDialect,
                    omp::OpenMPDialect>();
    
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    memref::registerAllocationOpInterfaceExternalModels(registry);

    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
    mlir::registerGPUDialectTranslation(registry);
    NVVM::registerNVVMTargetInterfaceExternalModels(registry);
    
    // Register ConvertToLLVM interfaces for all dialects used in lowering
    // NOTE: some trees may not provide all of these registration helpers; remove if needed.
    ub::registerConvertUBToLLVMInterface(registry);
    registerConvertOpenMPToLLVMInterface(registry);
    registerConvertNVVMToLLVMInterface(registry);
    registerConvertMemRefToLLVMInterface(registry);
    registerConvertFuncToLLVMInterface(registry);
    arith::registerConvertArithToLLVMInterface(registry);
    cf::registerConvertControlFlowToLLVMInterface(registry);
    registerConvertComplexToLLVMInterface(registry);
    index::registerConvertIndexToLLVMInterface(registry);
    registerConvertMathToLLVMInterface(registry);
    gpu::registerConvertGpuToLLVMInterface(registry);
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpPassManager pm(ModuleOp::getOperationName());

    pm.addPass(createBufferizationToMemRefPass()); // bufferize -> memref
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.nest<func::FuncOp>().addPass(createGpuMapParallelLoopsPass()); // map parallel loops
    pm.addPass(createParallelLoopToGpuPass());
    pm.addPass(createGpuKernelOutliningPass());

    pm.nest<gpu::GPUModuleOp>().addPass(createConvertGpuOpsToNVVMOps());
    GpuNVVMAttachTargetOptions gputargetOptions;
    gputargetOptions.chip = "sm_90";
    gputargetOptions.triple = "nvptx64-nvidia-cuda";
    pm.addPass(createGpuNVVMAttachTarget(gputargetOptions));

    // --- Lower control-flow/affine and convert scalar/index/arithmetic types ---
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createLowerAffinePass()); // Removed the unrealized casts in gpu kernels
    pm.addPass(createArithToLLVMConversionPass());
    ConvertIndexToLLVMPassOptions indexToLLVMOptions;
    indexToLLVMOptions.indexBitwidth = 32;
    pm.addPass(createConvertIndexToLLVMPass(indexToLLVMOptions));
    pm.addPass(createUBToLLVMConversionPass());

    // MemRef/struct handling: expand metadata, finalize memref->llvm
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());

    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createConvertFuncToLLVMPass());

    // Convert device NVVM intrinsics to LLVM IR (device side)
    pm.addPass(createConvertNVVMToLLVMPass());
    pm.addPass(createConvertToLLVMPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());


    // --- Serialize GPU module to binary (PTX/ELF blob) ---
    pm.addPass(createGpuModuleToBinaryPass());

    pm.addPass(createGpuToLLVMConversionPass());
    
    // // Convert control-flow to LLVM (host CF)
    pm.addPass(createConvertControlFlowToLLVMPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Run the pipeline now.
    if (failed(runPipeline(pm, module)))
      signalPassFailure();
  }
  
  llvm::StringRef getArgument() const override {
    return "lower-loops-to-gpu-and-llvm";
  }

  llvm::StringRef getDescription() const override {
    return "Run GPU/NVVM and LLVM lowering passes (maps mlir-opt flags to C++ API).";
  }
};

namespace mlir {
namespace nn {

std::unique_ptr<mlir::Pass> createLowerLoopsToGPUPass() {
  return std::make_unique<LoopsToGPU>();
}

}
}
