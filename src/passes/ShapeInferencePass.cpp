#include "../dialect/NNDialect.h"
#include "Passes.h"
#include "ShapeInferenceInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::nn;

namespace {

class ShapeInferencePass : public mlir::PassWrapper<ShapeInferencePass, mlir::OperationPass<mlir::nn::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  
  void runOnOperation() override {
    auto function = getOperation();
    
    // Walk through all operations in the function
    function.walk([&](Operation *op) {
      // Check if the operation implements the ShapeInference interface
      if (auto shapeInferenceOp = dyn_cast<ShapeInference>(op)) {
        // Call the inferShapes method to update the operation's result types
        shapeInferenceOp.inferShapes();
        
        // Print debug information
        llvm::outs() << "Shape inference applied to: " << op->getName() << "\n";
        for (auto result : op->getResults()) {
          llvm::outs() << "  Result type: " << result.getType() << "\n";
        }
      }
    });
  }
  
  StringRef getArgument() const final { return "shape-inference"; }
  StringRef getDescription() const final {
    return "Infer shapes for operations in the NN dialect";
  }
};

} // anonymous namespace

// Function to create and return the pass
std::unique_ptr<mlir::Pass> mlir::nn::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
