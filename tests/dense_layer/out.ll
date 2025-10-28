; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8

declare void @free(ptr)

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @model(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30) {
  %structArg127 = alloca { ptr, ptr }, align 8
  %structArg123 = alloca { ptr, ptr, ptr, ptr }, align 8
  %structArg118 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg = alloca { ptr }, align 8
  %.reloaded82 = alloca i64, align 8
  %.reloaded83 = alloca i64, align 8
  %.reloaded = alloca i64, align 8
  %.reloaded43 = alloca i64, align 8
  %.reloaded44 = alloca i64, align 8
  %.reloaded45 = alloca i64, align 8
  %.reloaded46 = alloca i64, align 8
  %.reloaded47 = alloca i64, align 8
  %32 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 5) to i64), i64 64))
  %33 = ptrtoint ptr %32 to i64
  %34 = add i64 %33, 63
  %35 = urem i64 %34, 64
  %36 = sub i64 %34, %35
  %37 = inttoptr i64 %36 to ptr
  br label %entry

entry:                                            ; preds = %31
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
  store ptr %37, ptr %gep_, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @model..omp_par, ptr %structArg)
  br label %omp.par.outlined.exit

omp.par.outlined.exit:                            ; preds = %omp_parallel
  br label %omp.par.exit.split

omp.par.exit.split:                               ; preds = %omp.par.outlined.exit
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %2, ptr %.reloaded, align 4
  store i64 %5, ptr %.reloaded43, align 4
  store i64 %6, ptr %.reloaded44, align 4
  store i64 %9, ptr %.reloaded45, align 4
  store i64 %12, ptr %.reloaded46, align 4
  store i64 %13, ptr %.reloaded47, align 4
  br label %omp_parallel122

omp_parallel122:                                  ; preds = %omp.par.exit.split
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  %gep_.reloaded43 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 1
  store ptr %.reloaded43, ptr %gep_.reloaded43, align 8
  %gep_.reloaded44 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 2
  store ptr %.reloaded44, ptr %gep_.reloaded44, align 8
  %gep_.reloaded45 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 3
  store ptr %.reloaded45, ptr %gep_.reloaded45, align 8
  %gep_.reloaded46 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 4
  store ptr %.reloaded46, ptr %gep_.reloaded46, align 8
  %gep_.reloaded47 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 5
  store ptr %.reloaded47, ptr %gep_.reloaded47, align 8
  %gep_119 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 6
  store ptr %1, ptr %gep_119, align 8
  %gep_120 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 7
  store ptr %8, ptr %gep_120, align 8
  %gep_121 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg118, i32 0, i32 8
  store ptr %37, ptr %gep_121, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @model..omp_par.1, ptr %structArg118)
  br label %omp.par.outlined.exit42

omp.par.outlined.exit42:                          ; preds = %omp_parallel122
  br label %omp.par.exit.split41

omp.par.exit.split41:                             ; preds = %omp.par.outlined.exit42
  %omp_global_thread_num48 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %16, ptr %.reloaded82, align 4
  store i64 %18, ptr %.reloaded83, align 4
  br label %omp_parallel126

omp_parallel126:                                  ; preds = %omp.par.exit.split41
  %gep_.reloaded82 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg123, i32 0, i32 0
  store ptr %.reloaded82, ptr %gep_.reloaded82, align 8
  %gep_.reloaded83 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg123, i32 0, i32 1
  store ptr %.reloaded83, ptr %gep_.reloaded83, align 8
  %gep_124 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg123, i32 0, i32 2
  store ptr %37, ptr %gep_124, align 8
  %gep_125 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg123, i32 0, i32 3
  store ptr %15, ptr %gep_125, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @model..omp_par.2, ptr %structArg123)
  br label %omp.par.outlined.exit81

omp.par.outlined.exit81:                          ; preds = %omp_parallel126
  br label %omp.par.exit.split80

omp.par.exit.split80:                             ; preds = %omp.par.outlined.exit81
  %38 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 5) to i64), i64 64))
  %39 = ptrtoint ptr %38 to i64
  %40 = add i64 %39, 63
  %41 = urem i64 %40, 64
  %42 = sub i64 %40, %41
  %43 = inttoptr i64 %42 to ptr
  %omp_global_thread_num84 = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel130

omp_parallel130:                                  ; preds = %omp.par.exit.split80
  %gep_128 = getelementptr { ptr, ptr }, ptr %structArg127, i32 0, i32 0
  store ptr %37, ptr %gep_128, align 8
  %gep_129 = getelementptr { ptr, ptr }, ptr %structArg127, i32 0, i32 1
  store ptr %43, ptr %gep_129, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @model..omp_par.3, ptr %structArg127)
  br label %omp.par.outlined.exit117

omp.par.outlined.exit117:                         ; preds = %omp_parallel130
  br label %omp.par.exit.split116

omp.par.exit.split116:                            ; preds = %omp.par.outlined.exit117
  call void @free(ptr %32)
  %44 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %45 = ptrtoint ptr %44 to i64
  %46 = add i64 %45, 63
  %47 = urem i64 %46, 64
  %48 = sub i64 %46, %47
  %49 = inttoptr i64 %48 to ptr
  %50 = getelementptr float, ptr %49, i64 0
  store float 0.000000e+00, ptr %50, align 4
  br label %51

51:                                               ; preds = %54, %omp.par.exit.split116
  %52 = phi i64 [ %69, %54 ], [ 0, %omp.par.exit.split116 ]
  %53 = icmp slt i64 %52, 5
  br i1 %53, label %54, label %70

54:                                               ; preds = %51
  %55 = add i64 0, %52
  %56 = getelementptr float, ptr %43, i64 %55
  %57 = load float, ptr %56, align 4
  %58 = getelementptr float, ptr %20, i64 %21
  %59 = mul i64 %52, %24
  %60 = mul i64 %25, 0
  %61 = add i64 %59, %60
  %62 = getelementptr float, ptr %58, i64 %61
  %63 = load float, ptr %62, align 4
  %64 = getelementptr float, ptr %49, i64 0
  %65 = load float, ptr %64, align 4
  %66 = fmul float %57, %63
  %67 = fadd float %65, %66
  %68 = getelementptr float, ptr %49, i64 0
  store float %67, ptr %68, align 4
  %69 = add i64 %52, 1
  br label %51

70:                                               ; preds = %51
  call void @free(ptr %38)
  %71 = getelementptr float, ptr %49, i64 0
  %72 = load float, ptr %71, align 4
  %73 = getelementptr float, ptr %27, i64 %28
  %74 = mul i64 %30, 0
  %75 = getelementptr float, ptr %73, i64 %74
  %76 = load float, ptr %75, align 4
  %77 = fadd float %72, %76
  %78 = getelementptr float, ptr %49, i64 0
  store float %77, ptr %78, align 4
  %79 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %80 = ptrtoint ptr %79 to i64
  %81 = add i64 %80, 63
  %82 = urem i64 %81, 64
  %83 = sub i64 %81, %82
  %84 = inttoptr i64 %83 to ptr
  %85 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %79, 0
  %86 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %85, ptr %84, 1
  %87 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %86, i64 0, 2
  %88 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, i64 1, 3, 0
  %89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, i64 1, 3, 1
  %90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, i64 1, 4, 0
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, i64 1, 4, 1
  %92 = getelementptr float, ptr %49, i64 0
  %93 = load float, ptr %92, align 4
  %94 = fneg float %93
  %95 = call float @llvm.exp.f32(float %94)
  %96 = fadd float %95, 1.000000e+00
  %97 = fdiv float 1.000000e+00, %96
  %98 = getelementptr float, ptr %84, i64 0
  store float %97, ptr %98, align 4
  call void @free(ptr %44)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %91
}

; Function Attrs: nounwind
define internal void @model..omp_par.3(ptr noalias %tid.addr85, ptr noalias %zero.addr86, ptr %0) #0 {
omp.par.entry87:
  %gep_ = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8
  %gep_1 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_2 = load ptr, ptr %gep_1, align 8
  %p.lastiter110 = alloca i32, align 4
  %p.lowerbound111 = alloca i64, align 8
  %p.upperbound112 = alloca i64, align 8
  %p.stride113 = alloca i64, align 8
  %tid.addr.local90 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr85, align 4
  store i32 %1, ptr %tid.addr.local90, align 4
  %tid91 = load i32, ptr %tid.addr.local90, align 4
  br label %omp.region.after_alloca97

omp.region.after_alloca97:                        ; preds = %omp.par.entry87
  br label %omp.region.after_alloca94

omp.region.after_alloca94:                        ; preds = %omp.region.after_alloca97
  br label %omp.par.region88

omp.par.region88:                                 ; preds = %omp.region.after_alloca94
  br label %omp.par.region96

omp.par.region96:                                 ; preds = %omp.par.region88
  br label %omp_loop.preheader98

omp_loop.preheader98:                             ; preds = %omp.par.region96
  store i64 0, ptr %p.lowerbound111, align 4
  store i64 4, ptr %p.upperbound112, align 4
  store i64 1, ptr %p.stride113, align 4
  %omp_global_thread_num114 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num114, i32 34, ptr %p.lastiter110, ptr %p.lowerbound111, ptr %p.upperbound112, ptr %p.stride113, i64 1, i64 0)
  %2 = load i64, ptr %p.lowerbound111, align 4
  %3 = load i64, ptr %p.upperbound112, align 4
  %4 = sub i64 %3, %2
  %5 = add i64 %4, 1
  br label %omp_loop.header99

omp_loop.header99:                                ; preds = %omp_loop.inc102, %omp_loop.preheader98
  %omp_loop.iv105 = phi i64 [ 0, %omp_loop.preheader98 ], [ %omp_loop.next107, %omp_loop.inc102 ]
  br label %omp_loop.cond100

omp_loop.cond100:                                 ; preds = %omp_loop.header99
  %omp_loop.cmp106 = icmp ult i64 %omp_loop.iv105, %5
  br i1 %omp_loop.cmp106, label %omp_loop.body101, label %omp_loop.exit103

omp_loop.exit103:                                 ; preds = %omp_loop.cond100
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num114)
  %omp_global_thread_num115 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num115)
  br label %omp_loop.after104

omp_loop.after104:                                ; preds = %omp_loop.exit103
  br label %omp.region.cont95

omp.region.cont95:                                ; preds = %omp_loop.after104
  br label %omp.par.pre_finalize89

omp.par.pre_finalize89:                           ; preds = %omp.region.cont95
  br label %omp.par.outlined.exit117.exitStub

omp_loop.body101:                                 ; preds = %omp_loop.cond100
  %6 = add i64 %omp_loop.iv105, %2
  %7 = mul i64 %6, 1
  %8 = add i64 %7, 0
  br label %omp.wsloop.region109

omp.wsloop.region109:                             ; preds = %omp_loop.body101
  %9 = add i64 0, %8
  %10 = getelementptr float, ptr %loadgep_, i64 %9
  %11 = load float, ptr %10, align 4
  %12 = call float @llvm.maximum.f32(float %11, float 0.000000e+00)
  %13 = add i64 0, %8
  %14 = getelementptr float, ptr %loadgep_2, i64 %13
  store float %12, ptr %14, align 4
  br label %omp.region.cont108

omp.region.cont108:                               ; preds = %omp.wsloop.region109
  br label %omp_loop.inc102

omp_loop.inc102:                                  ; preds = %omp.region.cont108
  %omp_loop.next107 = add nuw i64 %omp_loop.iv105, 1
  br label %omp_loop.header99

omp.par.outlined.exit117.exitStub:                ; preds = %omp.par.pre_finalize89
  ret void
}

; Function Attrs: nounwind
define internal void @model..omp_par.2(ptr noalias %tid.addr49, ptr noalias %zero.addr50, ptr %0) #0 {
omp.par.entry51:
  %gep_.reloaded82 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded82 = load ptr, ptr %gep_.reloaded82, align 8
  %gep_.reloaded83 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded83 = load ptr, ptr %gep_.reloaded83, align 8
  %gep_ = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_ = load ptr, ptr %gep_, align 8
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_2 = load ptr, ptr %gep_1, align 8
  %p.lastiter74 = alloca i32, align 4
  %p.lowerbound75 = alloca i64, align 8
  %p.upperbound76 = alloca i64, align 8
  %p.stride77 = alloca i64, align 8
  %tid.addr.local54 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr49, align 4
  store i32 %1, ptr %tid.addr.local54, align 4
  %tid55 = load i32, ptr %tid.addr.local54, align 4
  %2 = load i64, ptr %loadgep_.reloaded82, align 4
  %3 = load i64, ptr %loadgep_.reloaded83, align 4
  br label %omp.region.after_alloca61

omp.region.after_alloca61:                        ; preds = %omp.par.entry51
  br label %omp.region.after_alloca58

omp.region.after_alloca58:                        ; preds = %omp.region.after_alloca61
  br label %omp.par.region52

omp.par.region52:                                 ; preds = %omp.region.after_alloca58
  br label %omp.par.region60

omp.par.region60:                                 ; preds = %omp.par.region52
  br label %omp_loop.preheader62

omp_loop.preheader62:                             ; preds = %omp.par.region60
  store i64 0, ptr %p.lowerbound75, align 4
  store i64 4, ptr %p.upperbound76, align 4
  store i64 1, ptr %p.stride77, align 4
  %omp_global_thread_num78 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num78, i32 34, ptr %p.lastiter74, ptr %p.lowerbound75, ptr %p.upperbound76, ptr %p.stride77, i64 1, i64 0)
  %4 = load i64, ptr %p.lowerbound75, align 4
  %5 = load i64, ptr %p.upperbound76, align 4
  %6 = sub i64 %5, %4
  %7 = add i64 %6, 1
  br label %omp_loop.header63

omp_loop.header63:                                ; preds = %omp_loop.inc66, %omp_loop.preheader62
  %omp_loop.iv69 = phi i64 [ 0, %omp_loop.preheader62 ], [ %omp_loop.next71, %omp_loop.inc66 ]
  br label %omp_loop.cond64

omp_loop.cond64:                                  ; preds = %omp_loop.header63
  %omp_loop.cmp70 = icmp ult i64 %omp_loop.iv69, %7
  br i1 %omp_loop.cmp70, label %omp_loop.body65, label %omp_loop.exit67

omp_loop.exit67:                                  ; preds = %omp_loop.cond64
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num78)
  %omp_global_thread_num79 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num79)
  br label %omp_loop.after68

omp_loop.after68:                                 ; preds = %omp_loop.exit67
  br label %omp.region.cont59

omp.region.cont59:                                ; preds = %omp_loop.after68
  br label %omp.par.pre_finalize53

omp.par.pre_finalize53:                           ; preds = %omp.region.cont59
  br label %omp.par.outlined.exit81.exitStub

omp_loop.body65:                                  ; preds = %omp_loop.cond64
  %8 = add i64 %omp_loop.iv69, %4
  %9 = mul i64 %8, 1
  %10 = add i64 %9, 0
  br label %omp.wsloop.region73

omp.wsloop.region73:                              ; preds = %omp_loop.body65
  %11 = add i64 0, %10
  %12 = getelementptr float, ptr %loadgep_, i64 %11
  %13 = load float, ptr %12, align 4
  %14 = getelementptr float, ptr %loadgep_2, i64 %2
  %15 = mul i64 %10, %3
  %16 = getelementptr float, ptr %14, i64 %15
  %17 = load float, ptr %16, align 4
  %18 = fadd float %13, %17
  %19 = add i64 0, %10
  %20 = getelementptr float, ptr %loadgep_, i64 %19
  store float %18, ptr %20, align 4
  br label %omp.region.cont72

omp.region.cont72:                                ; preds = %omp.wsloop.region73
  br label %omp_loop.inc66

omp_loop.inc66:                                   ; preds = %omp.region.cont72
  %omp_loop.next71 = add nuw i64 %omp_loop.iv69, 1
  br label %omp_loop.header63

omp.par.outlined.exit81.exitStub:                 ; preds = %omp.par.pre_finalize53
  ret void
}

; Function Attrs: nounwind
define internal void @model..omp_par.1(ptr noalias %tid.addr7, ptr noalias %zero.addr8, ptr %0) #0 {
omp.par.entry9:
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded = load ptr, ptr %gep_.reloaded, align 8
  %gep_.reloaded43 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded43 = load ptr, ptr %gep_.reloaded43, align 8
  %gep_.reloaded44 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded44 = load ptr, ptr %gep_.reloaded44, align 8
  %gep_.reloaded45 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded45 = load ptr, ptr %gep_.reloaded45, align 8
  %gep_.reloaded46 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded46 = load ptr, ptr %gep_.reloaded46, align 8
  %gep_.reloaded47 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_.reloaded47 = load ptr, ptr %gep_.reloaded47, align 8
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_ = load ptr, ptr %gep_, align 8
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 7
  %loadgep_2 = load ptr, ptr %gep_1, align 8
  %gep_3 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 8
  %loadgep_4 = load ptr, ptr %gep_3, align 8
  %p.lastiter35 = alloca i32, align 4
  %p.lowerbound36 = alloca i64, align 8
  %p.upperbound37 = alloca i64, align 8
  %p.stride38 = alloca i64, align 8
  %tid.addr.local12 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr7, align 4
  store i32 %1, ptr %tid.addr.local12, align 4
  %tid13 = load i32, ptr %tid.addr.local12, align 4
  %2 = load i64, ptr %loadgep_.reloaded, align 4
  %3 = load i64, ptr %loadgep_.reloaded43, align 4
  %4 = load i64, ptr %loadgep_.reloaded44, align 4
  %5 = load i64, ptr %loadgep_.reloaded45, align 4
  %6 = load i64, ptr %loadgep_.reloaded46, align 4
  %7 = load i64, ptr %loadgep_.reloaded47, align 4
  br label %omp.region.after_alloca19

omp.region.after_alloca19:                        ; preds = %omp.par.entry9
  br label %omp.region.after_alloca16

omp.region.after_alloca16:                        ; preds = %omp.region.after_alloca19
  br label %omp.par.region10

omp.par.region10:                                 ; preds = %omp.region.after_alloca16
  br label %omp.par.region18

omp.par.region18:                                 ; preds = %omp.par.region10
  br label %omp_loop.preheader20

omp_loop.preheader20:                             ; preds = %omp.par.region18
  store i64 0, ptr %p.lowerbound36, align 4
  store i64 4, ptr %p.upperbound37, align 4
  store i64 1, ptr %p.stride38, align 4
  %omp_global_thread_num39 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num39, i32 34, ptr %p.lastiter35, ptr %p.lowerbound36, ptr %p.upperbound37, ptr %p.stride38, i64 1, i64 0)
  %8 = load i64, ptr %p.lowerbound36, align 4
  %9 = load i64, ptr %p.upperbound37, align 4
  %10 = sub i64 %9, %8
  %11 = add i64 %10, 1
  br label %omp_loop.header21

omp_loop.header21:                                ; preds = %omp_loop.inc24, %omp_loop.preheader20
  %omp_loop.iv27 = phi i64 [ 0, %omp_loop.preheader20 ], [ %omp_loop.next29, %omp_loop.inc24 ]
  br label %omp_loop.cond22

omp_loop.cond22:                                  ; preds = %omp_loop.header21
  %omp_loop.cmp28 = icmp ult i64 %omp_loop.iv27, %11
  br i1 %omp_loop.cmp28, label %omp_loop.body23, label %omp_loop.exit25

omp_loop.exit25:                                  ; preds = %omp_loop.cond22
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num39)
  %omp_global_thread_num40 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num40)
  br label %omp_loop.after26

omp_loop.after26:                                 ; preds = %omp_loop.exit25
  br label %omp.region.cont17

omp.region.cont17:                                ; preds = %omp_loop.after26
  br label %omp.par.pre_finalize11

omp.par.pre_finalize11:                           ; preds = %omp.region.cont17
  br label %omp.par.outlined.exit42.exitStub

omp_loop.body23:                                  ; preds = %omp_loop.cond22
  %12 = add i64 %omp_loop.iv27, %8
  %13 = mul i64 %12, 1
  %14 = add i64 %13, 0
  br label %omp.wsloop.region31

omp.wsloop.region31:                              ; preds = %omp_loop.body23
  br label %omp.wsloop.region32

omp.wsloop.region32:                              ; preds = %omp.wsloop.region33, %omp.wsloop.region31
  %15 = phi i64 [ %36, %omp.wsloop.region33 ], [ 0, %omp.wsloop.region31 ]
  %16 = icmp slt i64 %15, 10
  br i1 %16, label %omp.wsloop.region33, label %omp.wsloop.region34

omp.wsloop.region34:                              ; preds = %omp.wsloop.region32
  br label %omp.region.cont30

omp.region.cont30:                                ; preds = %omp.wsloop.region34
  br label %omp_loop.inc24

omp_loop.inc24:                                   ; preds = %omp.region.cont30
  %omp_loop.next29 = add nuw i64 %omp_loop.iv27, 1
  br label %omp_loop.header21

omp.wsloop.region33:                              ; preds = %omp.wsloop.region32
  %17 = getelementptr float, ptr %loadgep_, i64 %2
  %18 = mul i64 %3, 0
  %19 = mul i64 %15, %4
  %20 = add i64 %18, %19
  %21 = getelementptr float, ptr %17, i64 %20
  %22 = load float, ptr %21, align 4
  %23 = getelementptr float, ptr %loadgep_2, i64 %5
  %24 = mul i64 %15, %6
  %25 = mul i64 %14, %7
  %26 = add i64 %24, %25
  %27 = getelementptr float, ptr %23, i64 %26
  %28 = load float, ptr %27, align 4
  %29 = add i64 0, %14
  %30 = getelementptr float, ptr %loadgep_4, i64 %29
  %31 = load float, ptr %30, align 4
  %32 = fmul float %22, %28
  %33 = fadd float %31, %32
  %34 = add i64 0, %14
  %35 = getelementptr float, ptr %loadgep_4, i64 %34
  store float %33, ptr %35, align 4
  %36 = add i64 %15, 1
  br label %omp.wsloop.region32

omp.par.outlined.exit42.exitStub:                 ; preds = %omp.par.pre_finalize11
  ret void
}

; Function Attrs: nounwind
define internal void @model..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_ = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i64, align 8
  %p.upperbound = alloca i64, align 8
  %p.stride = alloca i64, align 8
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  br label %omp.region.after_alloca2

omp.region.after_alloca2:                         ; preds = %omp.par.entry
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.region.after_alloca2
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.par.region1
  store i64 0, ptr %p.lowerbound, align 4
  store i64 4, ptr %p.upperbound, align 4
  store i64 1, ptr %p.stride, align 4
  %omp_global_thread_num4 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num4, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i64 1, i64 0)
  %2 = load i64, ptr %p.lowerbound, align 4
  %3 = load i64, ptr %p.upperbound, align 4
  %4 = sub i64 %3, %2
  %5 = add i64 %4, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i64 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i64 %omp_loop.iv, %5
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num4)
  %omp_global_thread_num5 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num5)
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp_loop.after
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %omp.par.outlined.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %6 = add i64 %omp_loop.iv, %2
  %7 = mul i64 %6, 1
  %8 = add i64 %7, 0
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp_loop.body
  %9 = add i64 0, %8
  %10 = getelementptr float, ptr %loadgep_, i64 %9
  store float 0.000000e+00, ptr %10, align 4
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp.wsloop.region
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont3
  %omp_loop.next = add nuw i64 %omp_loop.iv, 1
  br label %omp_loop.header

omp.par.outlined.exit.exitStub:                   ; preds = %omp.par.pre_finalize
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #0

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64) #0

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) #0

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #2

; Function Attrs: nounwind
declare !callback !1 void @__kmpc_fork_call(ptr, i32, ptr, ...) #0

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!2}
!2 = !{i64 2, i64 -1, i64 -1, i1 true}
