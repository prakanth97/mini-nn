//===- ops.td - NN Dialect Operations ----------------------------*- tablegen -*-===//
//
// This file defines the operations for the NN dialect.
//===----------------------------------------------------------------------===//

#ifndef NN_OPS
#define NN_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// NN Dialect Definition
//===----------------------------------------------------------------------===//

def NN_Dialect : Dialect {
    let name = "nn";
    let cppNamespace = "::mlir::nn";
    let summary = "High Level Dialect for Neural Networks";
    let description = [{
        This dialect provides operations for defining and manipulating neural network models.
        It includes operations for layers, activations, and other common neural network constructs.
    }];
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base NN Operation
//===----------------------------------------------------------------------===//

class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Layer Operations
//===----------------------------------------------------------------------===//

def NN_DenseOp : NN_Op<"dense", [Pure]> {
    let summary = "Dense/Fully connected layer operation";
    let description = [{
        Dense layer operation that performs matrix multiplication followed by bias addition:
        output = input * weight + bias
        
        Example:
        ```mlir
        %result = nn.dense(%input, %weight, %bias) : (tensor<?x784xf32>, tensor<784x128xf32>, tensor<128xf32>) -> tensor<?x128xf32>
        ```
    }];
    
    let arguments = (ins 
        AnyTensor:$input,
        AnyTensor:$weight,
        Optional<AnyTensor>:$bias
    );
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $input `,` $weight (`,` $bias^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def NN_Conv1DOp : NN_Op<"conv1d", [Pure]> {
    let summary = "1D Convolution layer operation";
    let description = [{
        1D convolution operation for processing sequential data.
        
        Example:
        ```mlir
        %result = nn.conv1d(%input, %kernel, %bias) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
        ```
    }];
    
    let arguments = (ins 
        AnyTensor:$input,
        AnyTensor:$kernel,
        Optional<AnyTensor>:$bias,
        I64Attr:$stride,
        I64Attr:$padding
    );
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $input `,` $kernel (`,` $bias^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def NN_Conv2DOp : NN_Op<"conv2d", [Pure]> {
    let summary = "2D Convolution layer operation";
    let description = [{
        2D convolution operation for processing image data.
        
        Example:
        ```mlir
        %result = nn.conv2d(%input, %kernel, %bias) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
        ```
    }];
    
    let arguments = (ins 
        AnyTensor:$input,
        AnyTensor:$kernel,
        Optional<AnyTensor>:$bias,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padding
    );
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $input `,` $kernel (`,` $bias^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// Activation Functions
//===----------------------------------------------------------------------===//

def NN_ReluOp : NN_Op<"relu", [Pure, SameOperandsAndResultShape]> {
    let summary = "ReLU activation function";
    let description = [{
        Applies the ReLU (Rectified Linear Unit) activation function element-wise:
        relu(x) = max(0, x)
        
        Example:
        ```mlir
        %result = nn.relu(%input) : tensor<?x?xf32> -> tensor<?x?xf32>
        ```
    }];
    
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `->` type($result)
    }];
}

def NN_SigmoidOp : NN_Op<"sigmoid", [Pure, SameOperandsAndResultShape]> {
    let summary = "Sigmoid activation function";
    let description = [{
        Applies the sigmoid activation function element-wise:
        sigmoid(x) = 1 / (1 + exp(-x))
        
        Example:
        ```mlir
        %result = nn.sigmoid(%input) : tensor<?x?xf32> -> tensor<?x?xf32>
        ```
    }];
    
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `->` type($result)
    }];
}

def NN_TanhOp : NN_Op<"tanh", [Pure, SameOperandsAndResultShape]> {
    let summary = "Tanh activation function";
    let description = [{
        Applies the hyperbolic tangent activation function element-wise:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        
        Example:
        ```mlir
        %result = nn.tanh(%input) : tensor<?x?xf32> -> tensor<?x?xf32>
        ```
    }];
    
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `->` type($result)
    }];
}

def NN_SoftmaxOp : NN_Op<"softmax", [Pure, SameOperandsAndResultShape]> {
    let summary = "Softmax activation function";
    let description = [{
        Applies the softmax activation function along the last dimension:
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        
        Example:
        ```mlir
        %result = nn.softmax(%input) : tensor<?x?xf32> -> tensor<?x?xf32>
        ```
    }];
    
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `->` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def NN_TransposeOp : NN_Op<"transpose", [Pure]> {
    let summary = "Matrix transpose operation";
    let description = [{
        Transposes the last two dimensions of a tensor.
        
        Example:
        ```mlir
        %result = nn.transpose(%input) : tensor<?x?x?xf32> -> tensor<?x?x?xf32>
        ```
    }];
    
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` functional-type(operands, results)
    }];
}

def NN_MatmulOp : NN_Op<"matmul", [Pure]> {
    let summary = "Matrix multiplication operation";
    let description = [{
        Performs matrix multiplication between two tensors.
        
        Example:
        ```mlir
        %result = nn.matmul(%lhs, %rhs) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
        ```
    }];
    
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
    
    let assemblyFormat = [{
        `(` $lhs `,` $rhs `)` attr-dict `:` functional-type(operands, results)
    }];
}

def NN_AddOp : NN_Op<"add", [Pure, Commutative, SameOperandsAndResultShape]> {
  let summary = "Elementwise addition operation";
  let description = [{
    Performs elementwise addition of two tensors.
    
    Example:
    ```mlir
    %2 = nn.add %0, %1 : tensor<4x4xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

def NN_FuncOp : NN_Op<"func", [
	CallableOpInterface,
    FunctionOpInterface,
    IsolatedFromAbove
]> {
    let summary = "Neural network function definition";
    let description = [{
        Defines a function in the neural network dialect that can contain
        layer operations and other neural network computations.
        
        Example:
        ```mlir
        nn.func @forward(%input: tensor<?x784xf32>) -> tensor<?x10xf32> {
            %dense1 = nn.dense(%input, %w1, %b1) : (tensor<?x784xf32>, tensor<784x128xf32>, tensor<128xf32>) -> tensor<?x128xf32>
            %relu1 = nn.relu(%dense1) : tensor<?x128xf32>
            %output = nn.dense(%relu1, %w2, %b2) : (tensor<?x128xf32>, tensor<128x10xf32>, tensor<10xf32>) -> tensor<?x10xf32>
            nn.return %output : tensor<?x10xf32>
        }
        ```
    }];
    
    let arguments = (ins 
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type
    );
    let regions = (region AnyRegion:$body);
    
    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    ];
    
    let extraClassDeclaration = [{
        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
        
        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
        
        /// Returns the region on the current operation that is callable.
        Region *getCallableRegion() { return &getBody(); }
        
        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }
    }];
    
    let assemblyFormat = [{
        $sym_name `(` `)` attr-dict `:` $function_type $body
    }];
}

def NN_ReturnOp : NN_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let summary = "Neural network function return operation";
    let description = [{
        Returns values from a neural network function.
        
        Example:
        ```mlir
        nn.return %result : tensor<?x10xf32>
        ```
    }];
    
    let arguments = (ins Variadic<AnyType>:$operands);
    
    let assemblyFormat = [{
        ($operands^ `:` type($operands))? attr-dict
    }];
}

#endif // NN_OPS
