; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous7 = external global [3 x i32]
@_anonymous6 = external global [3 x i32]
@_anonymous5 = external global [3 x i32]
@_anonymous4 = external global [3 x i32]
@_anonymous3 = external global [3 x i32]
@_anonymous2 = external global [3 x i32]
@_anonymous1 = external global [3 x i32]
@_anonymous0 = external global [3 x i32]
@in1_0_cons_buff_1 = external global [256 x bfloat]
@in1_0_cons_buff_0 = external global [256 x bfloat]
@in1_1_cons_buff_1 = external global [256 x bfloat]
@in1_1_cons_buff_0 = external global [256 x bfloat]
@in1_2_cons_buff_1 = external global [256 x bfloat]
@in1_2_cons_buff_0 = external global [256 x bfloat]
@in1_3_cons_buff_1 = external global [256 x bfloat]
@in1_3_cons_buff_0 = external global [256 x bfloat]
@in1_4_cons_buff_1 = external global [256 x bfloat]
@in1_4_cons_buff_0 = external global [256 x bfloat]
@in1_5_cons_buff_1 = external global [256 x bfloat]
@in1_5_cons_buff_0 = external global [256 x bfloat]
@in1_6_cons_buff_1 = external global [256 x bfloat]
@in1_6_cons_buff_0 = external global [256 x bfloat]
@in1_7_cons_buff_1 = external global [256 x bfloat]
@in1_7_cons_buff_0 = external global [256 x bfloat]
@in2_0_cons_buff_1 = external global [256 x bfloat]
@in2_0_cons_buff_0 = external global [256 x bfloat]
@in2_1_cons_buff_1 = external global [256 x bfloat]
@in2_1_cons_buff_0 = external global [256 x bfloat]
@in2_2_cons_buff_1 = external global [256 x bfloat]
@in2_2_cons_buff_0 = external global [256 x bfloat]
@in2_3_cons_buff_1 = external global [256 x bfloat]
@in2_3_cons_buff_0 = external global [256 x bfloat]
@in2_4_cons_buff_1 = external global [256 x bfloat]
@in2_4_cons_buff_0 = external global [256 x bfloat]
@in2_5_cons_buff_1 = external global [256 x bfloat]
@in2_5_cons_buff_0 = external global [256 x bfloat]
@in2_6_cons_buff_1 = external global [256 x bfloat]
@in2_6_cons_buff_0 = external global [256 x bfloat]
@in2_7_cons_buff_1 = external global [256 x bfloat]
@in2_7_cons_buff_0 = external global [256 x bfloat]
@out_0_buff_1 = external global [256 x bfloat]
@out_0_buff_0 = external global [256 x bfloat]
@out_1_buff_1 = external global [256 x bfloat]
@out_1_buff_0 = external global [256 x bfloat]
@out_2_buff_1 = external global [256 x bfloat]
@out_2_buff_0 = external global [256 x bfloat]
@out_3_buff_1 = external global [256 x bfloat]
@out_3_buff_0 = external global [256 x bfloat]
@out_4_buff_1 = external global [256 x bfloat]
@out_4_buff_0 = external global [256 x bfloat]
@out_5_buff_1 = external global [256 x bfloat]
@out_5_buff_0 = external global [256 x bfloat]
@out_6_buff_1 = external global [256 x bfloat]
@out_6_buff_0 = external global [256 x bfloat]
@out_7_buff_1 = external global [256 x bfloat]
@out_7_buff_0 = external global [256 x bfloat]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @op13_eltwise_add_bf16_vector(ptr, ptr, ptr, i32)

define void @core_0_2() {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %19, %0
  %2 = phi i64 [ %35, %19 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %36

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %5 = load i32, ptr @_anonymous0, align 4
  switch i32 %5, label %6 [
    i32 0, label %37
    i32 1, label %39
  ]

6:                                                ; preds = %37, %39, %4
  %7 = phi ptr [ %40, %39 ], [ %38, %37 ], [ @in1_0_cons_buff_0, %4 ]
  %8 = getelementptr [256 x bfloat], ptr %7, i32 0, i32 0
  br label %9

9:                                                ; preds = %6
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %10 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  switch i32 %10, label %11 [
    i32 0, label %41
    i32 1, label %43
  ]

11:                                               ; preds = %41, %43, %9
  %12 = phi ptr [ %44, %43 ], [ %42, %41 ], [ @in2_0_cons_buff_0, %9 ]
  %13 = getelementptr [256 x bfloat], ptr %12, i32 0, i32 0
  br label %14

14:                                               ; preds = %11
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  switch i32 %15, label %16 [
    i32 0, label %45
    i32 1, label %47
  ]

16:                                               ; preds = %45, %47, %14
  %17 = phi ptr [ %48, %47 ], [ %46, %45 ], [ @out_0_buff_0, %14 ]
  %18 = getelementptr [256 x bfloat], ptr %17, i32 0, i32 0
  br label %19

19:                                               ; preds = %16
  call void @op13_eltwise_add_bf16_vector(ptr %8, ptr %13, ptr %18, i32 256)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %20 = load i32, ptr @_anonymous0, align 4
  %21 = add i32 %20, 1
  %22 = icmp sge i32 %21, 2
  %23 = add i32 %20, -1
  %24 = select i1 %22, i32 %23, i32 %21
  store i32 %24, ptr @_anonymous0, align 4
  call void @llvm.aie2p.release(i32 50, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  call void @llvm.aie2p.release(i32 53, i32 1)
  %30 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  %31 = add i32 %30, 1
  %32 = icmp sge i32 %31, 2
  %33 = add i32 %30, -1
  %34 = select i1 %32, i32 %33, i32 %31
  store i32 %34, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  %35 = add i64 %2, 1
  br label %1

36:                                               ; preds = %1
  ret void

37:                                               ; preds = %4
  %38 = phi ptr [ @in1_0_cons_buff_0, %4 ]
  br label %6

39:                                               ; preds = %4
  %40 = phi ptr [ @in1_0_cons_buff_1, %4 ]
  br label %6

41:                                               ; preds = %9
  %42 = phi ptr [ @in2_0_cons_buff_0, %9 ]
  br label %11

43:                                               ; preds = %9
  %44 = phi ptr [ @in2_0_cons_buff_1, %9 ]
  br label %11

45:                                               ; preds = %14
  %46 = phi ptr [ @out_0_buff_0, %14 ]
  br label %16

47:                                               ; preds = %14
  %48 = phi ptr [ @out_0_buff_1, %14 ]
  br label %16
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
