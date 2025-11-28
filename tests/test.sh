#!/bin/bash
set -e  # Exit on any error

export MLIR_SRC_ROOT="/home/pt/warwick/cs325_demo/llvm-project/llvm-project/mlir"
export DYLD_LIBRARY_PATH="/home/pt/warwick/cs325_demo/llvm-project/openmp/build/runtime/src":$DYLD_LIBRARY_PATH

MLIR_TRANSLATE=/home/pt/warwick/cs325_demo/llvm-project/build/bin/mlir-translate
LLC_PATH=/home/pt/warwick/cs325_demo/llvm-project/build/bin/llc
OPENMP_RUNTIME_PATH=/home/pt/warwick/cs325_demo/llvm-project/openmp/build/runtime/src

DIR="$(pwd)"

### Build mini_nn compiler
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake -G Ninja ..
ninja
cd ..

### Build tests
COMP=$DIR/build/mini_nn
echo "Using compiler: $COMP"

cd tests/dense_layer

$COMP dense.nn

echo "Translating MLIR to LLVM IR..."
$MLIR_TRANSLATE dense_llvm.mlir --mlir-to-llvmir > out.ll

echo "Compiling LLVM IR to assembly..."
$LLC_PATH out.ll -o out.s --relocation-model=pic

echo "Assembling object file..."
clang -c -fPIC out.s -o out.o

echo "Compiling MLIR runtime..."
clang -c -fPIC ../mlir_runtime.c -o mlir_runtime.o

echo "Linking final executable with OpenMP..."
clang++ -o dense_exe driver.cpp out.o mlir_runtime.o \
  -I$OPENMP_RUNTIME_PATH \
  -L$OPENMP_RUNTIME_PATH \
  -lomp -lm \
  -Wl,-rpath,$OPENMP_RUNTIME_PATH

echo "Build completed successfully!"
echo "Run ./dense_exe to execute the test"
