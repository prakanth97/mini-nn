module {
  llvm.func @model(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: !llvm.ptr, %arg32: !llvm.ptr, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: i64) {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(5 : index) : i64
    %6 = llvm.mlir.constant(10 : index) : i64
    %7 = llvm.alloca %5 x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg38) : i64 = (%3) to (%5) step (%4) {
          %58 = llvm.mul %3, %5 : i64
          %59 = llvm.add %58, %arg38 : i64
          %60 = llvm.getelementptr %7[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %1, %60 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg38) : i64 = (%3) to (%5) step (%4) {
          llvm.br ^bb1(%3 : i64)
        ^bb1(%58: i64):  // 2 preds: ^bb0, ^bb2
          %59 = llvm.icmp "slt" %58, %6 : i64
          llvm.cond_br %59, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %60 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %61 = llvm.mul %arg5, %3 : i64
          %62 = llvm.mul %58, %arg6 : i64
          %63 = llvm.add %61, %62 : i64
          %64 = llvm.getelementptr %60[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %65 = llvm.load %64 : !llvm.ptr -> f32
          %66 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %67 = llvm.mul %58, %arg12 : i64
          %68 = llvm.mul %arg38, %arg13 : i64
          %69 = llvm.add %67, %68 : i64
          %70 = llvm.getelementptr %66[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %71 = llvm.load %70 : !llvm.ptr -> f32
          %72 = llvm.mul %3, %5 : i64
          %73 = llvm.add %72, %arg38 : i64
          %74 = llvm.getelementptr %7[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %75 = llvm.load %74 : !llvm.ptr -> f32
          %76 = llvm.fmul %65, %71 : f32
          %77 = llvm.fadd %75, %76 : f32
          %78 = llvm.mul %3, %5 : i64
          %79 = llvm.add %78, %arg38 : i64
          %80 = llvm.getelementptr %7[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %77, %80 : f32, !llvm.ptr
          %81 = llvm.add %58, %4 : i64
          llvm.br ^bb1(%81 : i64)
        ^bb3:  // pred: ^bb1
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg38) : i64 = (%3) to (%5) step (%4) {
          %58 = llvm.mul %3, %5 : i64
          %59 = llvm.add %58, %arg38 : i64
          %60 = llvm.getelementptr %7[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %61 = llvm.load %60 : !llvm.ptr -> f32
          %62 = llvm.getelementptr %arg15[%arg16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %63 = llvm.mul %arg38, %arg18 : i64
          %64 = llvm.getelementptr %62[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %65 = llvm.load %64 : !llvm.ptr -> f32
          %66 = llvm.fadd %61, %65 : f32
          %67 = llvm.mul %3, %5 : i64
          %68 = llvm.add %67, %arg38 : i64
          %69 = llvm.getelementptr %7[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %66, %69 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    %8 = llvm.alloca %5 x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg38) : i64 = (%3) to (%5) step (%4) {
          %58 = llvm.mul %3, %5 : i64
          %59 = llvm.add %58, %arg38 : i64
          %60 = llvm.getelementptr %7[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %61 = llvm.load %60 : !llvm.ptr -> f32
          %62 = llvm.intr.maximum(%61, %1) : (f32, f32) -> f32
          %63 = llvm.mul %3, %5 : i64
          %64 = llvm.add %63, %arg38 : i64
          %65 = llvm.getelementptr %8[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %62, %65 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    %9 = llvm.alloca %4 x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.add %3, %3 : i64
    %11 = llvm.getelementptr %9[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1, %11 : f32, !llvm.ptr
    llvm.br ^bb1(%3 : i64)
  ^bb1(%12: i64):  // 2 preds: ^bb0, ^bb2
    %13 = llvm.icmp "slt" %12, %5 : i64
    llvm.cond_br %13, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %14 = llvm.mul %3, %5 : i64
    %15 = llvm.add %14, %12 : i64
    %16 = llvm.getelementptr %8[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %17 = llvm.load %16 : !llvm.ptr -> f32
    %18 = llvm.getelementptr %arg20[%arg21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.mul %12, %arg24 : i64
    %20 = llvm.mul %arg25, %3 : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.getelementptr %18[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %23 = llvm.load %22 : !llvm.ptr -> f32
    %24 = llvm.add %3, %3 : i64
    %25 = llvm.getelementptr %9[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.load %25 : !llvm.ptr -> f32
    %27 = llvm.fmul %17, %23 : f32
    %28 = llvm.fadd %26, %27 : f32
    %29 = llvm.add %3, %3 : i64
    %30 = llvm.getelementptr %9[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %28, %30 : f32, !llvm.ptr
    %31 = llvm.add %12, %4 : i64
    llvm.br ^bb1(%31 : i64)
  ^bb3:  // pred: ^bb1
    %32 = llvm.add %3, %3 : i64
    %33 = llvm.getelementptr %9[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %34 = llvm.load %33 : !llvm.ptr -> f32
    %35 = llvm.getelementptr %arg27[%arg28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %36 = llvm.mul %arg30, %3 : i64
    %37 = llvm.getelementptr %35[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.fadd %34, %38 : f32
    %40 = llvm.add %3, %3 : i64
    %41 = llvm.getelementptr %9[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %39, %41 : f32, !llvm.ptr
    %42 = llvm.alloca %4 x f32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
    %43 = llvm.add %3, %3 : i64
    %44 = llvm.getelementptr %9[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %45 = llvm.load %44 : !llvm.ptr -> f32
    %46 = llvm.fneg %45 : f32
    %47 = llvm.intr.exp(%46) : (f32) -> f32
    %48 = llvm.fadd %47, %2 : f32
    %49 = llvm.fdiv %2, %48 : f32
    %50 = llvm.add %3, %3 : i64
    %51 = llvm.getelementptr %42[%50] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %49, %51 : f32, !llvm.ptr
    %52 = llvm.mul %4, %4 : i64
    %53 = llvm.mul %52, %4 : i64
    %54 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.mul %53, %55 : i64
    %57 = llvm.getelementptr %arg32[%arg33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%57, %42, %56) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}

