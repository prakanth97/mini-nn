#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h" 

#include "Passes.h" 

/*
Solution from https://discourse.llvm.org/t/making-linalg-matmul-to-gpu-runnable-code/3910 
*/

using namespace mlir;

class HostRegister : public PassWrapper<HostRegister, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HostRegister)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    
    llvm::DenseSet<Operation *> processedAllocas;

    module.walk([&](gpu::LaunchFuncOp launchOp) {
      for (Value operand : launchOp.getKernelOperands()) {
        Operation *defOp = operand.getDefiningOp();
        
        // Check for AllocOp (Heap)
        if (defOp && (isa<memref::AllocOp>(defOp))) {
          
          if (processedAllocas.contains(defOp)) {
            continue;
          }

          builder.setInsertionPointAfter(defOp);

          auto rankedType = mlir::cast<MemRefType>(operand.getType());
          
          auto unrankedType = UnrankedMemRefType::get(
              rankedType.getElementType(), 
              rankedType.getMemorySpace());

          Value unrankedMemRef = builder.create<memref::CastOp>(
              defOp->getLoc(), 
              unrankedType, 
              operand
          );

          builder.create<gpu::HostRegisterOp>(defOp->getLoc(), unrankedMemRef);

          processedAllocas.insert(defOp);
        }
      }
    });
  }
};

namespace mlir {
namespace nn {
std::unique_ptr<mlir::Pass> createGPUHostRegister() {
  return std::make_unique<HostRegister>();
}
}
}