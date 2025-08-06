//===- LowerToLinalgDialect.cpp - Lower NN Dialect to Linalg -----------===//
//
// This file implements the conversion from NN dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "../dialect/NNDialect.h"
#include "Passes.h"

using namespace mlir;
using namespace mlir::nn;

namespace {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Create a linalg.generic operation with the given indexing maps and iterator types.
linalg::GenericOp createGenericOp(OpBuilder &builder, Location loc,
                                  ArrayRef<Value> inputs,
                                  ArrayRef<Value> outputs,
                                  ArrayRef<AffineMap> indexingMaps,
                                  ArrayRef<utils::IteratorType> iteratorTypes,
                                  function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  return builder.create<linalg::GenericOp>(
      loc, outputs[0].getType(), inputs, outputs, indexingMaps, iteratorTypes,
      bodyBuilder);
}

//===----------------------------------------------------------------------===//
// Pattern Rewriters
//===----------------------------------------------------------------------===//

/// Lower nn.dense to linalg operations
struct DenseToLinalgLowering : public OpRewritePattern<DenseOp> {
  using OpRewritePattern<DenseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DenseOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    
    auto resultType = mlir::cast<TensorType>(op.getResult().getType());
    
    // Create output tensor initialized to zero
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    Value filledOutput = rewriter.create<linalg::FillOp>(loc, zero, output).getResult(0);
    
    // Perform matrix multiplication: input * weight
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, filledOutput.getType(), ValueRange{input, weight}, filledOutput);
    Value matmulResult = matmulOp.getResult(0);
    
    // Add bias if present
    if (bias) {
      // Create indexing maps for bias addition
      AffineMap inputMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
      AffineMap biasMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)}, rewriter.getContext());
      AffineMap outputMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
      
      SmallVector<AffineMap> indexingMaps = {inputMap, biasMap, outputMap};
      SmallVector<utils::IteratorType> iteratorTypes = {
          utils::IteratorType::parallel, utils::IteratorType::parallel};
      
      auto biasOp = createGenericOp(
          rewriter, loc, {matmulResult, bias}, {filledOutput}, indexingMaps, iteratorTypes,
          [](OpBuilder &b, Location loc, ValueRange args) {
            Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
            b.create<linalg::YieldOp>(loc, sum);
          });
      
      rewriter.replaceOp(op, biasOp.getResult(0));
    } else {
      rewriter.replaceOp(op, matmulResult);
    }
    
    return success();
  }
};

/// TODO: Lower nn.conv2d to linalg operations
// struct Conv2DToLinalgLowering : public OpRewritePattern<Conv2DOp> {
// 	using OpRewritePattern<Conv2DOp>::OpRewritePattern;	

// };

/// TODO: Lower nn.conv1d to linalg operations
// struct Conv1DToLinalgLowering : public OpRewritePattern<Conv1DOp> {
// 	using OpRewritePattern<Conv1DOp>::OpRewritePattern;
// };

/// Lower nn.relu to linalg.generic
struct ReluOpLinalgLowering : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto resultType = mlir::cast<TensorType>(op.getResult().getType());
    
    // Create output tensor
    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    // Create zero constant
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    
    // Create indexing maps (identity for both input and output)
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(
        resultType.getRank(), rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
    
    // Create iterator types (all parallel)
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);
    
    auto genericOp = createGenericOp(
        rewriter, loc, {input}, {output}, indexingMaps, iteratorTypes,
        [zero](OpBuilder &b, Location loc, ValueRange args) {
          Value max = b.create<arith::MaximumFOp>(loc, args[0], zero);
          b.create<linalg::YieldOp>(loc, max);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

/// Lower nn.sigmoid to linalg.generic  
struct SigmoidOpLinalgLowering : public OpRewritePattern<SigmoidOp> {
  using OpRewritePattern<SigmoidOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SigmoidOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto resultType = mlir::cast<TensorType>(op.getResult().getType());
    
    // Create output tensor
    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    // Create constants
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(resultType.getElementType(), 1.0));
    
    // Create indexing maps (identity for both input and output)
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(
        resultType.getRank(), rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
    
    // Create iterator types (all parallel)
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);
    
    auto genericOp = createGenericOp(
        rewriter, loc, {input}, {output}, indexingMaps, iteratorTypes,
        [one](OpBuilder &b, Location loc, ValueRange args) {
          // sigmoid(x) = 1 / (1 + exp(-x))
          Value negInput = b.create<arith::NegFOp>(loc, args[0]);
          Value exp = b.create<math::ExpOp>(loc, negInput);
          Value onePlusExp = b.create<arith::AddFOp>(loc, one, exp);
          Value sigmoid = b.create<arith::DivFOp>(loc, one, onePlusExp);
          b.create<linalg::YieldOp>(loc, sigmoid);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

/// Lower nn.tanh to math.tanh
struct TanhOpLinalgLowering : public OpRewritePattern<TanhOp> {
	using OpRewritePattern<TanhOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(TanhOp op, PatternRewriter &rewriter) const override {
		Location loc = op.getLoc();
		Value input = op.getInput();
		auto resultType = mlir::cast<TensorType>(op.getResult().getType());

		math::TanhOp tanhOp = rewriter.create<math::TanhOp>(
			loc, resultType, input);
		rewriter.replaceOp(op, tanhOp.getResult());
		return success();
	}
};

/// Lower nn.softmax to linalg.generic
struct SoftmaxOpLinalgLowering : public OpRewritePattern<SoftmaxOp> {
	using OpRewritePattern<SoftmaxOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(SoftmaxOp op, PatternRewriter &rewriter) const override {
		Location loc = op.getLoc();
		Value input = op.getInput();
		auto resultType = mlir::cast<TensorType>(op.getResult().getType());
		
		// Step 1: Apply exp element-wise
		Value expOutput = rewriter.create<tensor::EmptyOp>(
			loc, resultType.getShape(), resultType.getElementType());
		
		AffineMap identityMap = AffineMap::getMultiDimIdentityMap(
			resultType.getRank(), rewriter.getContext());
		SmallVector<AffineMap> expIndexingMaps = {identityMap, identityMap};
		SmallVector<utils::IteratorType> expIteratorTypes(
			resultType.getRank(), utils::IteratorType::parallel);
		
		auto expOp = createGenericOp(
			rewriter, loc, {input}, {expOutput}, expIndexingMaps, expIteratorTypes,
			[](OpBuilder &b, Location loc, ValueRange args) {
				Value exp = b.create<math::ExpOp>(loc, args[0]);
				b.create<linalg::YieldOp>(loc, exp);
			});
		
		// Step 2: Sum along the last dimension (assuming 2D input for simplicity)
		// For a proper implementation, you'd need to handle arbitrary dimensions
		auto inputShape = resultType.getShape();
		SmallVector<int64_t> sumShape(inputShape.begin(), inputShape.end()-1);
		if (sumShape.empty()) sumShape.push_back(1); // Handle 1D case
		
		Value sumInit = rewriter.create<tensor::EmptyOp>(loc, sumShape, resultType.getElementType());
		Value zero = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getZeroAttr(resultType.getElementType()));
		Value sumFilled = rewriter.create<linalg::FillOp>(loc, zero, sumInit).getResult(0);
		
		// Create reduction along last dimension
		SmallVector<int64_t> reductionDims = {static_cast<int64_t>(resultType.getRank() - 1)};
		auto sumOp = rewriter.create<linalg::ReduceOp>(
			loc, expOp.getResult(0), sumFilled, reductionDims,
			[](OpBuilder &b, Location loc, ValueRange args) {
				Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
				b.create<linalg::YieldOp>(loc, sum);
			});
		
		// Step 3: Divide exp values by sum
		Value output = rewriter.create<tensor::EmptyOp>(
			loc, resultType.getShape(), resultType.getElementType());
		
		// Create indexing maps for broadcasting division
		AffineMap inputMap = AffineMap::getMultiDimIdentityMap(resultType.getRank(), rewriter.getContext());
		AffineMap sumMap = AffineMap::get(resultType.getRank(), 0, 
			{rewriter.getAffineDimExpr(0)}, rewriter.getContext()); // Broadcast sum
		AffineMap outputMap = AffineMap::getMultiDimIdentityMap(resultType.getRank(), rewriter.getContext());
		
		SmallVector<AffineMap> divIndexingMaps = {inputMap, sumMap, outputMap};
		SmallVector<utils::IteratorType> divIteratorTypes(
			resultType.getRank(), utils::IteratorType::parallel);
		
		auto divOp = createGenericOp(
			rewriter, loc, {expOp.getResult(0), sumOp.getResult(0)}, {output}, 
			divIndexingMaps, divIteratorTypes,
			[](OpBuilder &b, Location loc, ValueRange args) {
				Value div = b.create<arith::DivFOp>(loc, args[0], args[1]);
				b.create<linalg::YieldOp>(loc, div);
			});
		
		rewriter.replaceOp(op, divOp.getResult(0));
		return success();
	}
};

/// Lower nn.transpose to linalg.generic
struct TransposeOpLinalgLowering : public OpRewritePattern<TransposeOp> {
	using OpRewritePattern<TransposeOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const override {
		Location loc = op.getLoc();
		Value input = op.getInput();
		auto inputType = mlir::cast<TensorType>(input.getType());
		auto resultType = mlir::cast<TensorType>(op.getResult().getType());

		// Create output tensor
		Value output = rewriter.create<tensor::EmptyOp>(
				loc, resultType.getShape(), resultType.getElementType());

		// For transpose, we need to swap the last two dimensions
		// Input indexing map: (d0, d1) -> (d0, d1) 
		// Output indexing map: (d0, d1) -> (d1, d0) - this swaps the dimensions
		MLIRContext *context = rewriter.getContext();
		
		if (inputType.getRank() != 2) {
			return rewriter.notifyMatchFailure(op, "transpose only supports 2D tensors currently");
		}
		
		AffineMap inputMap = AffineMap::get(2, 0, 
			{rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)}, context);
		AffineMap outputMap = AffineMap::get(2, 0, 
			{rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(0)}, context);
			
		SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};

		// Create iterator types (all parallel)
		SmallVector<utils::IteratorType> iteratorTypes = {
			utils::IteratorType::parallel, utils::IteratorType::parallel};

		auto genericOp = createGenericOp(
				rewriter, loc, {input}, {output}, indexingMaps, iteratorTypes,
				[](OpBuilder &b, Location loc, ValueRange args) {
					b.create<linalg::YieldOp>(loc, args[0]);
				});

		rewriter.replaceOp(op, genericOp.getResult(0));
		return success();
	}
};

/// Lower nn.matmul to linalg.matmul
struct MatmulOpLinalgLowering : public OpRewritePattern<MatmulOp> {
	using OpRewritePattern<MatmulOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
		Location loc = op.getLoc();
		Value lhs = op.getLhs();
		Value rhs = op.getRhs();
		auto resultType = mlir::cast<TensorType>(op.getResult().getType());

		// Create output tensor initialized to zero
		Value zero = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getZeroAttr(resultType.getElementType()));
		Value output = rewriter.create<tensor::EmptyOp>(
			loc, resultType.getShape(), resultType.getElementType());
		Value filledOutput = rewriter.create<linalg::FillOp>(loc, zero, output).getResult(0);

		// Create matmul operation
		auto matmulOp = rewriter.create<linalg::MatmulOp>(
			loc, filledOutput.getType(), ValueRange{lhs, rhs}, filledOutput);

		rewriter.replaceOp(op, matmulOp.getResult(0));
		return success();
	}
};

/// Lower nn.add to linalg.generic (element-wise addition with broadcasting support)
struct AddOpLinalgLowering : public OpRewritePattern<AddOp> {
	using OpRewritePattern<AddOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
		Location loc = op.getLoc();
		Value lhs = op.getLhs();
		Value rhs = op.getRhs();
		auto lhsType = mlir::cast<TensorType>(lhs.getType());
		auto rhsType = mlir::cast<TensorType>(rhs.getType());
		auto resultType = mlir::cast<TensorType>(op.getResult().getType());

		// Create output tensor
		Value output = rewriter.create<tensor::EmptyOp>(
			loc, resultType.getShape(), resultType.getElementType());

		MLIRContext *context = rewriter.getContext();
		
		// TODO : Handle this more generically, or have a sparate pass for broadcasting.
		// Handle broadcasting case: [1x5] + [5] -> [1x5]
		if (lhsType.getRank() == 2 && rhsType.getRank() == 1) {
			// LHS: (d0, d1) -> (d0, d1)
			// RHS: (d0, d1) -> (d1)  - broadcast along first dimension
			// Result: (d0, d1) -> (d0, d1)
			AffineMap lhsMap = AffineMap::getMultiDimIdentityMap(2, context);
			AffineMap rhsMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)}, context);
			AffineMap resultMap = AffineMap::getMultiDimIdentityMap(2, context);
			
			SmallVector<AffineMap> indexingMaps = {lhsMap, rhsMap, resultMap};
			SmallVector<utils::IteratorType> iteratorTypes = {
				utils::IteratorType::parallel, utils::IteratorType::parallel};

			auto genericOp = createGenericOp(
				rewriter, loc, {lhs, rhs}, {output}, indexingMaps, iteratorTypes,
				[](OpBuilder &b, Location loc, ValueRange args) {
					Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
					b.create<linalg::YieldOp>(loc, sum);
				});

			rewriter.replaceOp(op, genericOp.getResult(0));
		}
		else if (lhsType.getRank() == rhsType.getRank()) {
			// Same rank - element-wise addition
			AffineMap identityMap = AffineMap::getMultiDimIdentityMap(
				resultType.getRank(), context);
			SmallVector<AffineMap> indexingMaps = {identityMap, identityMap, identityMap};

			SmallVector<utils::IteratorType> iteratorTypes(
				resultType.getRank(), utils::IteratorType::parallel);

			auto genericOp = createGenericOp(
				rewriter, loc, {lhs, rhs}, {output}, indexingMaps, iteratorTypes,
				[](OpBuilder &b, Location loc, ValueRange args) {
					Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
					b.create<linalg::YieldOp>(loc, sum);
				});

			rewriter.replaceOp(op, genericOp.getResult(0));
		}
		else {
			return rewriter.notifyMatchFailure(op, "unsupported broadcasting pattern");
		}

		return success();
	}
};

/// Lower nn.func to func.func
struct FuncOpLowering : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op, PatternRewriter &rewriter) const override {
    // Convert nn.func to func.func
    auto funcType = op.getFunctionType();
    auto newFuncOp = rewriter.create<func::FuncOp>(
        op.getLoc(), op.getSymName(), funcType);
    
    // Move the body
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.end());
    
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower nn.return to func.return
struct ReturnOpLowering : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern<ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReturnOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LowerToLinalgPass : public PassWrapper<LowerToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, 
                   arith::ArithDialect, math::MathDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                          arith::ArithDialect, math::MathDialect, func::FuncDialect>();
    target.addIllegalDialect<NNDialect>();

    RewritePatternSet patterns(context);
    patterns.add<DenseToLinalgLowering, ReluOpLinalgLowering, SigmoidOpLinalgLowering, 
                 TanhOpLinalgLowering, SoftmaxOpLinalgLowering, TransposeOpLinalgLowering,
                 MatmulOpLinalgLowering, AddOpLinalgLowering,
                 FuncOpLowering, ReturnOpLowering>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "lower-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower NN dialect to Linalg dialect";
  }
};

} // namespace

namespace mlir {
namespace nn {

std::unique_ptr<mlir::Pass> createLowerToLinalgPass() {
  return std::make_unique<LowerToLinalgPass>();
}

} // namespace nn
} // namespace mlir
