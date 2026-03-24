add_executable(test_eden_fp4 test_eden_fp4.cu ../csrc/round_eden_fp4.cu ../csrc/round_four_six.cu)
target_compile_options(test_eden_fp4 PUBLIC -lineinfo -ffast-math)

add_executable(test_group_transform test_group_transform.cu ../csrc/group_transform.cu)
target_compile_options(test_group_transform PUBLIC -lineinfo -ffast-math)

add_executable(bench_group_transform bench_group_transform.cu ../csrc/rht128_ws.cu ../csrc/rht128.cu
../csrc/group_transform.cu ../csrc/rht128_tma.cu)
target_compile_options(bench_group_transform PUBLIC -lineinfo -ffast-math)
target_link_libraries(bench_group_transform PUBLIC CUDA::cuda_driver)

add_executable(bench_group_transform_eden bench_group_transform_eden.cu ../csrc/group_transform_and_eden.cu  ../csrc/rht128_eden.cu)
target_compile_options(bench_group_transform_eden PUBLIC -lineinfo -ffast-math)
target_link_libraries(bench_group_transform_eden PUBLIC CUDA::cuda_driver)

add_executable(bench_rht128_requant bench_rht128_requant.cu ../csrc/rht128_requant.cu ../csrc/group_transform_and_eden.cu)
target_compile_options(bench_rht128_requant PUBLIC -lineinfo -ffast-math)
target_link_libraries(bench_rht128_requant PUBLIC CUDA::cuda_driver)
