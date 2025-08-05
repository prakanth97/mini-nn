# mini-nn
MLIR Compiler for neural network.

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
