module {
  llvm.func @model(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64) {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(50 : index) : i64
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(5 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(10 : index) : i64
    %7 = llvm.alloca %1 x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg26, %arg27) : i64 = (%3, %3) to (%4, %6) step (%5, %5) {
          %16 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %17 = llvm.mul %arg26, %arg12 : i64
          %18 = llvm.mul %arg27, %arg13 : i64
          %19 = llvm.add %17, %18 : i64
          %20 = llvm.getelementptr %16[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %21 = llvm.load %20 : !llvm.ptr -> f32
          %22 = llvm.mul %arg27, %4 : i64
          %23 = llvm.add %22, %arg26 : i64
          %24 = llvm.getelementptr %7[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %21, %24 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    %8 = llvm.alloca %4 x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg26) : i64 = (%3) to (%4) step (%5) {
          %16 = llvm.mul %3, %4 : i64
          %17 = llvm.add %16, %arg26 : i64
          %18 = llvm.getelementptr %8[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %2, %18 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg26) : i64 = (%3) to (%4) step (%5) {
          llvm.br ^bb1(%3 : i64)
        ^bb1(%16: i64):  // 2 preds: ^bb0, ^bb2
          %17 = llvm.icmp "slt" %16, %6 : i64
          llvm.cond_br %17, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %18 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %19 = llvm.mul %arg5, %3 : i64
          %20 = llvm.mul %16, %arg6 : i64
          %21 = llvm.add %19, %20 : i64
          %22 = llvm.getelementptr %18[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %23 = llvm.load %22 : !llvm.ptr -> f32
          %24 = llvm.mul %16, %4 : i64
          %25 = llvm.add %24, %arg26 : i64
          %26 = llvm.getelementptr %7[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %27 = llvm.load %26 : !llvm.ptr -> f32
          %28 = llvm.mul %3, %4 : i64
          %29 = llvm.add %28, %arg26 : i64
          %30 = llvm.getelementptr %8[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %31 = llvm.load %30 : !llvm.ptr -> f32
          %32 = llvm.fmul %23, %27 : f32
          %33 = llvm.fadd %31, %32 : f32
          %34 = llvm.mul %3, %4 : i64
          %35 = llvm.add %34, %arg26 : i64
          %36 = llvm.getelementptr %8[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %33, %36 : f32, !llvm.ptr
          %37 = llvm.add %16, %5 : i64
          llvm.br ^bb1(%37 : i64)
        ^bb3:  // pred: ^bb1
          omp.yield
        }
      }
      omp.terminator
    }
    %9 = llvm.alloca %4 x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg26) : i64 = (%3) to (%4) step (%5) {
          %16 = llvm.mul %3, %4 : i64
          %17 = llvm.add %16, %arg26 : i64
          %18 = llvm.getelementptr %8[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %19 = llvm.load %18 : !llvm.ptr -> f32
          %20 = llvm.getelementptr %arg15[%arg16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %21 = llvm.mul %arg26, %arg18 : i64
          %22 = llvm.getelementptr %20[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %23 = llvm.load %22 : !llvm.ptr -> f32
          %24 = llvm.fadd %19, %23 : f32
          %25 = llvm.mul %3, %4 : i64
          %26 = llvm.add %25, %arg26 : i64
          %27 = llvm.getelementptr %9[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %24, %27 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    %10 = llvm.mul %5, %5 : i64
    %11 = llvm.mul %10, %4 : i64
    %12 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.mul %11, %13 : i64
    %15 = llvm.getelementptr %arg20[%arg21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%15, %9, %14) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

