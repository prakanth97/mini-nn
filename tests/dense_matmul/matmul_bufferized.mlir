module {
  func.func @model(%arg0: memref<1x10xf32, strided<[?, ?], offset: ?>>, %arg1: memref<5x10xf32, strided<[?, ?], offset: ?>>, %arg2: memref<5xf32, strided<[?], offset: ?>>, %arg3: memref<1x5xf32>) {
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<10x5xf32>
    scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c5, %c10) step (%c1, %c1) {
      %0 = memref.load %arg1[%arg4, %arg5] : memref<5x10xf32, strided<[?, ?], offset: ?>>
      memref.store %0, %alloca[%arg5, %arg4] : memref<10x5xf32>
      scf.reduce 
    }
    %alloca_0 = memref.alloca() {alignment = 64 : i64} : memref<1x5xf32>
    scf.parallel (%arg4) = (%c0) to (%c5) step (%c1) {
      memref.store %cst, %alloca_0[%c0, %arg4] : memref<1x5xf32>
      scf.reduce 
    }
    scf.parallel (%arg4) = (%c0) to (%c5) step (%c1) {
      scf.for %arg5 = %c0 to %c10 step %c1 {
        %0 = memref.load %arg0[%c0, %arg5] : memref<1x10xf32, strided<[?, ?], offset: ?>>
        %1 = memref.load %alloca[%arg5, %arg4] : memref<10x5xf32>
        %2 = memref.load %alloca_0[%c0, %arg4] : memref<1x5xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        memref.store %4, %alloca_0[%c0, %arg4] : memref<1x5xf32>
      }
      scf.reduce 
    }
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<1x5xf32>
    scf.parallel (%arg4) = (%c0) to (%c5) step (%c1) {
      %0 = memref.load %alloca_0[%c0, %arg4] : memref<1x5xf32>
      %1 = memref.load %arg2[%arg4] : memref<5xf32, strided<[?], offset: ?>>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloca_1[%c0, %arg4] : memref<1x5xf32>
      scf.reduce 
    }
    memref.copy %alloca_1, %arg3 : memref<1x5xf32> to memref<1x5xf32>
    return
  }
}

