#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

#include "Passes.h"

using namespace mlir;

class GPUMemoryTransfer : public PassWrapper<GPUMemoryTransfer, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUMemoryTransfer)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    OpBuilder builder(module.getContext());

    llvm::DenseMap<Value, Value> hostToDeviceMap;
    llvm::DenseSet<Operation*> updatedKernels;

    module.walk([&](gpu::LaunchFuncOp launchOp) {
      builder.setInsertionPoint(launchOp);

      unsigned totalOperands = launchOp->getNumOperands();
      unsigned numKernelOperands = launchOp.getNumKernelOperands();
      unsigned kernelArgsStart = totalOperands - numKernelOperands;

      // Track changed indices to update kernel signature. 
      llvm::SmallVector<unsigned, 4> changedIndices;

      for (unsigned i = 0; i < numKernelOperands; ++i) {
        unsigned absoluteIndex = kernelArgsStart + i;
        Value operand = launchOp->getOperand(absoluteIndex);
        
        auto hostMemRefType = operand.getType().dyn_cast<MemRefType>();

        // Skip non-memrefs or memrefs that are already on device (space != 0)
        if (!hostMemRefType || hostMemRefType.getMemorySpaceAsInt() != 0) 
            continue;

        Value finalOperand;

        if (hostToDeviceMap.count(operand)) {
          finalOperand = hostToDeviceMap[operand];
        } else {
          // Dynamic Sizes
          llvm::SmallVector<Value, 4> dynamicSizes;
          for (unsigned dim = 0; dim < hostMemRefType.getRank(); ++dim) {
            if (hostMemRefType.isDynamicDim(dim)) {
              auto dimOp = builder.create<memref::DimOp>(launchOp.getLoc(), operand, dim);
              dynamicSizes.push_back(dimOp);
            }
          }

          // Dense Alloc Type (Space 1)
          auto identityMap = builder.getMultiDimIdentityMap(hostMemRefType.getRank());
          auto deviceAllocType = MemRefType::get(
              hostMemRefType.getShape(),
              hostMemRefType.getElementType(),
              AffineMapAttr::get(identityMap), 
              builder.getI64IntegerAttr(1)
          );

          // Alloc
          auto deviceAlloc = builder.create<gpu::AllocOp>(
              launchOp.getLoc(), deviceAllocType, nullptr, ValueRange{}, 
              dynamicSizes, ValueRange{}, false);

          // Memcpy
          finalOperand = deviceAlloc.getResult(0);
          builder.create<gpu::MemcpyOp>(
              launchOp.getLoc(), std::nullopt, ValueRange{}, finalOperand, operand);

          // Cast to Strided (Space 1)
          // auto targetKernelType = MemRefType::get(
          //     hostMemRefType.getShape(),
          //     hostMemRefType.getElementType(),
          //     hostMemRefType.getLayout(), 
          //     builder.getI64IntegerAttr(1));

          // if (deviceAllocType != targetKernelType) {
          //   finalOperand = builder.create<memref::CastOp>(
          //       launchOp.getLoc(), targetKernelType, devicePtr);
          // }

          hostToDeviceMap[operand] = finalOperand;
        }

        // Update Launch Op
        launchOp->setOperand(absoluteIndex, finalOperand);
        
        // Mark this index as changed so we can update the kernel function
        changedIndices.push_back(i);
      }

      auto kernelFunc = symbolTable.lookupNearestSymbolFrom<gpu::GPUFuncOp>(
          launchOp, launchOp.getKernel());

      if (kernelFunc && !updatedKernels.count(kernelFunc)) {
        updatedKernels.insert(kernelFunc);
        
        FunctionType funcType = kernelFunc.getFunctionType();
        SmallVector<Type, 4> newInputTypes(funcType.getInputs().begin(), funcType.getInputs().end());
        Block &entryBlock = kernelFunc.front();

        for (unsigned idx : changedIndices) {
          // Get the current type (which is Space 0)
          auto oldType = newInputTypes[idx].cast<MemRefType>();

          auto identityMap = builder.getMultiDimIdentityMap(oldType.getRank());
          
          // Create the new type (Space 1, same layout)
          auto newType = MemRefType::get(
              oldType.getShape(),
              oldType.getElementType(),
              AffineMapAttr::get(identityMap),
              builder.getI64IntegerAttr(1));

          newInputTypes[idx] = newType;

          entryBlock.getArgument(idx).setType(newType);
        }

        auto newFuncType = FunctionType::get(
            kernelFunc.getContext(), newInputTypes, funcType.getResults());
        kernelFunc.setFunctionType(newFuncType);
      }
    });

    // func->walk([&] (func::ReturnOp returnOp) {
    //   builder.setInsertionPoint(returnOp);

    //   for (auto [hostVal, deviceVal] : hostToDeviceMap) {
    //     builder.create<gpu::MemcpyOp>(
    //       returnOp->getLoc(),
    //       /*asyncToken=*/std::nullopt, // TODO: Is this correct?
    //       /*asyncDependencies=*/ValueRange{},
    //       /*dst=*/hostVal,
    //       /*src=*/deviceVal
    //     );

    //     // builder.create<gpu::DeallocOp>(
    //     //   returnOp->getLoc(),
    //     //   nullptr,
    //     //   deviceVal
    //     // );
    //   }
    // });
  }
};

namespace mlir {
namespace nn {
std::unique_ptr<mlir::Pass> createGPUMemoryTransferPass() {
  return std::make_unique<GPUMemoryTransfer>();
}
}
}