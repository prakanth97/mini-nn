module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @model(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64, %arg29: i64, %arg30: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(5 : index) : i64
    %7 = llvm.mlir.constant(10 : index) : i64
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.getelementptr %1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.add %10, %0 : i64
    %12 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.sub %0, %5 : i64
    %15 = llvm.add %13, %14 : i64
    %16 = llvm.urem %15, %0 : i64
    %17 = llvm.sub %15, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg31) : i64 = (%4) to (%6) step (%5) {
          %97 = llvm.mul %4, %6 : i64
          %98 = llvm.add %97, %arg31 : i64
          %99 = llvm.getelementptr %18[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %2, %99 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg31) : i64 = (%4) to (%6) step (%5) {
          llvm.br ^bb1(%4 : i64)
        ^bb1(%97: i64):  // 2 preds: ^bb0, ^bb2
          %98 = llvm.icmp "slt" %97, %7 : i64
          llvm.cond_br %98, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %99 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %100 = llvm.mul %arg5, %4 : i64
          %101 = llvm.mul %97, %arg6 : i64
          %102 = llvm.add %100, %101 : i64
          %103 = llvm.getelementptr %99[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %104 = llvm.load %103 : !llvm.ptr -> f32
          %105 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %106 = llvm.mul %97, %arg12 : i64
          %107 = llvm.mul %arg31, %arg13 : i64
          %108 = llvm.add %106, %107 : i64
          %109 = llvm.getelementptr %105[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %110 = llvm.load %109 : !llvm.ptr -> f32
          %111 = llvm.mul %4, %6 : i64
          %112 = llvm.add %111, %arg31 : i64
          %113 = llvm.getelementptr %18[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %114 = llvm.load %113 : !llvm.ptr -> f32
          %115 = llvm.fmul %104, %110 : f32
          %116 = llvm.fadd %114, %115 : f32
          %117 = llvm.mul %4, %6 : i64
          %118 = llvm.add %117, %arg31 : i64
          %119 = llvm.getelementptr %18[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %116, %119 : f32, !llvm.ptr
          %120 = llvm.add %97, %5 : i64
          llvm.br ^bb1(%120 : i64)
        ^bb3:  // pred: ^bb1
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg31) : i64 = (%4) to (%6) step (%5) {
          %97 = llvm.mul %4, %6 : i64
          %98 = llvm.add %97, %arg31 : i64
          %99 = llvm.getelementptr %18[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %100 = llvm.load %99 : !llvm.ptr -> f32
          %101 = llvm.getelementptr %arg15[%arg16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %102 = llvm.mul %arg31, %arg18 : i64
          %103 = llvm.getelementptr %101[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %104 = llvm.load %103 : !llvm.ptr -> f32
          %105 = llvm.fadd %100, %104 : f32
          %106 = llvm.mul %4, %6 : i64
          %107 = llvm.add %106, %arg31 : i64
          %108 = llvm.getelementptr %18[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %105, %108 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    %19 = llvm.getelementptr %1[5] : (!llvm.ptr) -> !llvm.ptr, f32
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.add %20, %0 : i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.sub %0, %5 : i64
    %25 = llvm.add %23, %24 : i64
    %26 = llvm.urem %25, %0 : i64
    %27 = llvm.sub %25, %26 : i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg31) : i64 = (%4) to (%6) step (%5) {
          %97 = llvm.mul %4, %6 : i64
          %98 = llvm.add %97, %arg31 : i64
          %99 = llvm.getelementptr %18[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %100 = llvm.load %99 : !llvm.ptr -> f32
          %101 = llvm.intr.maximum(%100, %2) : (f32, f32) -> f32
          %102 = llvm.mul %4, %6 : i64
          %103 = llvm.add %102, %arg31 : i64
          %104 = llvm.getelementptr %28[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %101, %104 : f32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.call @free(%12) : (!llvm.ptr) -> ()
    %29 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.add %30, %0 : i64
    %32 = llvm.call @malloc(%31) : (i64) -> !llvm.ptr
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.sub %0, %5 : i64
    %35 = llvm.add %33, %34 : i64
    %36 = llvm.urem %35, %0 : i64
    %37 = llvm.sub %35, %36 : i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %39 = llvm.add %4, %4 : i64
    %40 = llvm.getelementptr %38[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %40 : f32, !llvm.ptr
    llvm.br ^bb1(%4 : i64)
  ^bb1(%41: i64):  // 2 preds: ^bb0, ^bb2
    %42 = llvm.icmp "slt" %41, %6 : i64
    llvm.cond_br %42, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %43 = llvm.mul %4, %6 : i64
    %44 = llvm.add %43, %41 : i64
    %45 = llvm.getelementptr %28[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.load %45 : !llvm.ptr -> f32
    %47 = llvm.getelementptr %arg20[%arg21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %48 = llvm.mul %41, %arg24 : i64
    %49 = llvm.mul %arg25, %4 : i64
    %50 = llvm.add %48, %49 : i64
    %51 = llvm.getelementptr %47[%50] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.load %51 : !llvm.ptr -> f32
    %53 = llvm.add %4, %4 : i64
    %54 = llvm.getelementptr %38[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %55 = llvm.load %54 : !llvm.ptr -> f32
    %56 = llvm.fmul %46, %52 : f32
    %57 = llvm.fadd %55, %56 : f32
    %58 = llvm.add %4, %4 : i64
    %59 = llvm.getelementptr %38[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %57, %59 : f32, !llvm.ptr
    %60 = llvm.add %41, %5 : i64
    llvm.br ^bb1(%60 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @free(%22) : (!llvm.ptr) -> ()
    %61 = llvm.add %4, %4 : i64
    %62 = llvm.getelementptr %38[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.load %62 : !llvm.ptr -> f32
    %64 = llvm.getelementptr %arg27[%arg28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %65 = llvm.mul %arg30, %4 : i64
    %66 = llvm.getelementptr %64[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %67 = llvm.load %66 : !llvm.ptr -> f32
    %68 = llvm.fadd %63, %67 : f32
    %69 = llvm.add %4, %4 : i64
    %70 = llvm.getelementptr %38[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %68, %70 : f32, !llvm.ptr
    %71 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %73 = llvm.add %72, %0 : i64
    %74 = llvm.call @malloc(%73) : (i64) -> !llvm.ptr
    %75 = llvm.ptrtoint %74 : !llvm.ptr to i64
    %76 = llvm.sub %0, %5 : i64
    %77 = llvm.add %75, %76 : i64
    %78 = llvm.urem %77, %0 : i64
    %79 = llvm.sub %77, %78 : i64
    %80 = llvm.inttoptr %79 : i64 to !llvm.ptr
    %81 = llvm.insertvalue %74, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.insertvalue %80, %81[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.insertvalue %4, %82[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.insertvalue %5, %83[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.insertvalue %5, %84[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.insertvalue %5, %85[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.insertvalue %5, %86[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.add %4, %4 : i64
    %89 = llvm.getelementptr %38[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %90 = llvm.load %89 : !llvm.ptr -> f32
    %91 = llvm.fneg %90 : f32
    %92 = llvm.intr.exp(%91) : (f32) -> f32
    %93 = llvm.fadd %92, %3 : f32
    %94 = llvm.fdiv %3, %93 : f32
    %95 = llvm.add %4, %4 : i64
    %96 = llvm.getelementptr %80[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %94, %96 : f32, !llvm.ptr
    llvm.call @free(%32) : (!llvm.ptr) -> ()
    llvm.return %87 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
