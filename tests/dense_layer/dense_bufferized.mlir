module {
  func.func @model(%arg0: memref<1x10xf32, strided<[?, ?], offset: ?>>, %arg1: memref<10x5xf32, strided<[?, ?], offset: ?>>, %arg2: memref<5xf32, strided<[?], offset: ?>>, %arg3: memref<5x1xf32, strided<[?, ?], offset: ?>>, %arg4: memref<1xf32, strided<[?], offset: ?>>, %arg5: memref<1x1xf32>) {
    %c10 = arith.constant 10 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x5xf32>
    scf.parallel (%arg6) = (%c0) to (%c5) step (%c1) {
      memref.store %cst_0, %alloca[%c0, %arg6] : memref<1x5xf32>
      scf.reduce 
    }
    scf.parallel (%arg6) = (%c0) to (%c5) step (%c1) {
      scf.for %arg7 = %c0 to %c10 step %c1 {
        %8 = memref.load %arg0[%c0, %arg7] : memref<1x10xf32, strided<[?, ?], offset: ?>>
        %9 = memref.load %arg1[%arg7, %arg6] : memref<10x5xf32, strided<[?, ?], offset: ?>>
        %10 = memref.load %alloca[%c0, %arg6] : memref<1x5xf32>
        %11 = arith.mulf %8, %9 : f32
        %12 = arith.addf %10, %11 : f32
        memref.store %12, %alloca[%c0, %arg6] : memref<1x5xf32>
      }
      scf.reduce 
    }
    scf.parallel (%arg6) = (%c0) to (%c5) step (%c1) {
      %8 = memref.load %alloca[%c0, %arg6] : memref<1x5xf32>
      %9 = memref.load %arg2[%arg6] : memref<5xf32, strided<[?], offset: ?>>
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %alloca[%c0, %arg6] : memref<1x5xf32>
      scf.reduce 
    }
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<1x5xf32>
    scf.parallel (%arg6) = (%c0) to (%c5) step (%c1) {
      %8 = memref.load %alloca[%c0, %arg6] : memref<1x5xf32>
      %9 = arith.maximumf %8, %cst_0 : f32
      memref.store %9, %alloca_1[%c0, %arg6] : memref<1x5xf32>
      scf.reduce 
    }
    %alloca_2 = memref.alloca() {alignment = 64 : i64} : memref<1x1xf32>
    memref.store %cst_0, %alloca_2[%c0, %c0] : memref<1x1xf32>
    scf.for %arg6 = %c0 to %c5 step %c1 {
      %8 = memref.load %alloca_1[%c0, %arg6] : memref<1x5xf32>
      %9 = memref.load %arg3[%arg6, %c0] : memref<5x1xf32, strided<[?, ?], offset: ?>>
      %10 = memref.load %alloca_2[%c0, %c0] : memref<1x1xf32>
      %11 = arith.mulf %8, %9 : f32
      %12 = arith.addf %10, %11 : f32
      memref.store %12, %alloca_2[%c0, %c0] : memref<1x1xf32>
    }
    %0 = memref.load %alloca_2[%c0, %c0] : memref<1x1xf32>
    %1 = memref.load %arg4[%c0] : memref<1xf32, strided<[?], offset: ?>>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %alloca_2[%c0, %c0] : memref<1x1xf32>
    %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<1x1xf32>
    %3 = memref.load %alloca_2[%c0, %c0] : memref<1x1xf32>
    %4 = arith.negf %3 : f32
    %5 = math.exp %4 : f32
    %6 = arith.addf %5, %cst : f32
    %7 = arith.divf %cst, %6 : f32
    memref.store %7, %alloca_3[%c0, %c0] : memref<1x1xf32>
    memref.copy %alloca_3, %arg5 : memref<1x1xf32> to memref<1x1xf32>
    return
  }
}

