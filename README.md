# mini-nn
MLIR Compiler for neural network.

Lowering Samples. 

Dense Layer lower to Linalg dialect.

```
// Before:
%0 = nn.dense(%arg0, %arg1, %arg2) : (tensor<1x10xf32>, tensor<10x5xf32>, tensor<5xf32>) -> tensor<1x5xf32>

// After:
%0 = tensor.empty() : tensor<1x5xf32>
%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x5xf32>) -> tensor<1x5xf32>
%2 = linalg.matmul ins(%arg0, %arg1 : tensor<1x10xf32>, tensor<10x5xf32>) outs(%1 : tensor<1x5xf32>) -> tensor<1x5xf32>
%3 = linalg.generic {...} ins(%2, %arg2 : tensor<1x5xf32>, tensor<5xf32>) outs(%1 : tensor<1x5xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %12 = arith.addf %in, %in_3 : f32
    linalg.yield %12 : f32
} -> tensor<1x5xf32>
```

Relu lowered to Linalg dialect.

```
// Before:
%1 = nn.relu(%0) : tensor<1x5xf32> -> tensor<1x5xf32>

// After:
%5 = linalg.generic {...} ins(%3 : tensor<1x5xf32>) outs(%4 : tensor<1x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %12 = arith.maximumf %in, %cst_0 : f32
    linalg.yield %12 : f32
} -> tensor<1x5xf32>
```

## Set-up guide

### Build MLIR repository

```
mkdir build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

cmake --build . -j --target check-mlir
```

Set the LLVM_ROOT
```
export LLVM_ROOT=....../llvm-project/build
```

### Build Mini-NN

```
mkdir build

cmake -DLLVM_ENABLE_RTTI=ON .. 

make
```

Execute -> `./build/mini_nn tests/dense_layer/dense.nn`
