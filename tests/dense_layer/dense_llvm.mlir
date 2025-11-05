module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @model(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: !llvm.ptr, %arg32: !llvm.ptr, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: i64) {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(5 : index) : i64
    %7 = llvm.mlir.constant(10 : index) : i64
    %8 = llvm.getelementptr %1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    %10 = llvm.add %9, %0 : i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.sub %0, %5 : i64
    %14 = llvm.add %12, %13 : i64
    %15 = llvm.urem %14, %0 : i64
    %16 = llvm.sub %14, %15 : i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg38) : i64 = (%4) to (%6) step (%5) {
          %95 = llvm.mul %4, %6 : i64
          %96 = llvm.add %95, %arg38 : i64
          %97 = llvm.getelementptr %17[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %2, %97 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg38) : i64 = (%4) to (%6) step (%5) {
          llvm.br ^bb1(%4 : i64)
        ^bb1(%95: i64):  // 2 preds: ^bb0, ^bb2
          %96 = llvm.icmp "slt" %95, %7 : i64
          llvm.cond_br %96, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %97 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %98 = llvm.mul %arg5, %4 : i64
          %99 = llvm.mul %95, %arg6 : i64
          %100 = llvm.add %98, %99 : i64
          %101 = llvm.getelementptr %97[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %102 = llvm.load %101 : !llvm.ptr -> f32
          %103 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %104 = llvm.mul %95, %arg12 : i64
          %105 = llvm.mul %arg38, %arg13 : i64
          %106 = llvm.add %104, %105 : i64
          %107 = llvm.getelementptr %103[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %108 = llvm.load %107 : !llvm.ptr -> f32
          %109 = llvm.mul %4, %6 : i64
          %110 = llvm.add %109, %arg38 : i64
          %111 = llvm.getelementptr %17[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %112 = llvm.load %111 : !llvm.ptr -> f32
          %113 = llvm.fmul %102, %108 : f32
          %114 = llvm.fadd %112, %113 : f32
          %115 = llvm.mul %4, %6 : i64
          %116 = llvm.add %115, %arg38 : i64
          %117 = llvm.getelementptr %17[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %114, %117 : f32, !llvm.ptr
          %118 = llvm.add %95, %5 : i64
          llvm.br ^bb1(%118 : i64)
        ^bb3:  // pred: ^bb1
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg38) : i64 = (%4) to (%6) step (%5) {
          %95 = llvm.mul %4, %6 : i64
          %96 = llvm.add %95, %arg38 : i64
          %97 = llvm.getelementptr %17[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %98 = llvm.load %97 : !llvm.ptr -> f32
          %99 = llvm.getelementptr %arg15[%arg16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %100 = llvm.mul %arg38, %arg18 : i64
          %101 = llvm.getelementptr %99[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %102 = llvm.load %101 : !llvm.ptr -> f32
          %103 = llvm.fadd %98, %102 : f32
          %104 = llvm.mul %4, %6 : i64
          %105 = llvm.add %104, %arg38 : i64
          %106 = llvm.getelementptr %17[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %103, %106 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    %18 = llvm.getelementptr %1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.add %19, %0 : i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.sub %0, %5 : i64
    %24 = llvm.add %22, %23 : i64
    %25 = llvm.urem %24, %0 : i64
    %26 = llvm.sub %24, %25 : i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg38) : i64 = (%4) to (%6) step (%5) {
          %95 = llvm.mul %4, %6 : i64
          %96 = llvm.add %95, %arg38 : i64
          %97 = llvm.getelementptr %17[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %98 = llvm.load %97 : !llvm.ptr -> f32
          %99 = llvm.intr.maximum(%98, %2) : (f32, f32) -> f32
          %100 = llvm.mul %4, %6 : i64
          %101 = llvm.add %100, %arg38 : i64
          %102 = llvm.getelementptr %27[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %99, %102 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.call @free(%11) : (!llvm.ptr) -> ()
    %28 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.add %29, %0 : i64
    %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.sub %0, %5 : i64
    %34 = llvm.add %32, %33 : i64
    %35 = llvm.urem %34, %0 : i64
    %36 = llvm.sub %34, %35 : i64
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr
    %38 = llvm.add %4, %4 : i64
    %39 = llvm.getelementptr %37[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %39 : f32, !llvm.ptr
    llvm.br ^bb1(%4 : i64)
  ^bb1(%40: i64):  // 2 preds: ^bb0, ^bb2
    %41 = llvm.icmp "slt" %40, %6 : i64
    llvm.cond_br %41, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %42 = llvm.mul %4, %6 : i64
    %43 = llvm.add %42, %40 : i64
    %44 = llvm.getelementptr %27[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %45 = llvm.load %44 : !llvm.ptr -> f32
    %46 = llvm.getelementptr %arg20[%arg21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %47 = llvm.mul %40, %arg24 : i64
    %48 = llvm.mul %arg25, %4 : i64
    %49 = llvm.add %47, %48 : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %51 = llvm.load %50 : !llvm.ptr -> f32
    %52 = llvm.add %4, %4 : i64
    %53 = llvm.getelementptr %37[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.load %53 : !llvm.ptr -> f32
    %55 = llvm.fmul %45, %51 : f32
    %56 = llvm.fadd %54, %55 : f32
    %57 = llvm.add %4, %4 : i64
    %58 = llvm.getelementptr %37[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %58 : f32, !llvm.ptr
    %59 = llvm.add %40, %5 : i64
    llvm.br ^bb1(%59 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @free(%21) : (!llvm.ptr) -> ()
    %60 = llvm.add %4, %4 : i64
    %61 = llvm.getelementptr %37[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %62 = llvm.load %61 : !llvm.ptr -> f32
    %63 = llvm.getelementptr %arg27[%arg28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.mul %arg30, %4 : i64
    %65 = llvm.getelementptr %63[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %66 = llvm.load %65 : !llvm.ptr -> f32
    %67 = llvm.fadd %62, %66 : f32
    %68 = llvm.add %4, %4 : i64
    %69 = llvm.getelementptr %37[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %67, %69 : f32, !llvm.ptr
    %70 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %71 = llvm.ptrtoint %70 : !llvm.ptr to i64
    %72 = llvm.add %71, %0 : i64
    %73 = llvm.call @malloc(%72) : (i64) -> !llvm.ptr
    %74 = llvm.ptrtoint %73 : !llvm.ptr to i64
    %75 = llvm.sub %0, %5 : i64
    %76 = llvm.add %74, %75 : i64
    %77 = llvm.urem %76, %0 : i64
    %78 = llvm.sub %76, %77 : i64
    %79 = llvm.inttoptr %78 : i64 to !llvm.ptr
    %80 = llvm.add %4, %4 : i64
    %81 = llvm.getelementptr %37[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %82 = llvm.load %81 : !llvm.ptr -> f32
    %83 = llvm.fneg %82 : f32
    %84 = llvm.intr.exp(%83) : (f32) -> f32
    %85 = llvm.fadd %84, %3 : f32
    %86 = llvm.fdiv %3, %85 : f32
    %87 = llvm.add %4, %4 : i64
    %88 = llvm.getelementptr %79[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %86, %88 : f32, !llvm.ptr
    llvm.call @free(%31) : (!llvm.ptr) -> ()
    %89 = llvm.mul %5, %5 : i64
    %90 = llvm.mul %89, %5 : i64
    %91 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.mul %90, %92 : i64
    %94 = llvm.getelementptr %arg32[%arg33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%94, %79, %93) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }
}
