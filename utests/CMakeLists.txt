# subfolder containing all the important unit tests

# declaration of unit test executables
add_executable(UtestProc utest_proc.cpp)
target_link_libraries(UtestProc PUBLIC Histeq)
add_test(NAME test_proc COMMAND UtestProc)

add_executable(UtestOverwrite utest_overwrite.cpp)
target_link_libraries(UtestOverwrite PUBLIC Histeq)
add_test(NAME test_overwrite COMMAND UtestOverwrite)

add_executable(UtestNoiseLevel utest_noiseLevel.cpp)
target_link_libraries(UtestNoiseLevel PUBLIC Histeq)
add_test(NAME test_noiseLevel COMMAND UtestNoiseLevel)

add_executable(UtestBinarization utest_binarization.cpp)
target_link_libraries(UtestBinarization PUBLIC Histeq)
add_test(NAME test_binarization COMMAND UtestBinarization)

add_executable(UtestVector3 utest_vector3.cpp)
add_test(NAME test_vector3 COMMAND UtestVector3)

add_executable(UtestGenfilt utest_genfilt.cpp)
target_link_libraries(UtestGenfilt PUBLIC Genfilt)
add_test(NAME test_genfilt COMMAND UtestGenfilt)

add_executable(UtestMeanfilt utest_meanfilt.cpp)
target_link_libraries(UtestMeanfilt PUBLIC Meanfilt)
add_test(NAME test_meanfilt COMMAND UtestMeanfilt)

add_executable(UtestGaussfilt utest_gaussfilt.cpp)
target_link_libraries(UtestGaussfilt PUBLIC Gaussfilt)
add_test(NAME test_gaussfilt COMMAND UtestGaussfilt)

add_executable(UtestNormalizer utest_normalizer.cpp)
target_link_libraries(UtestNormalizer PUBLIC Normalizer)
add_test(NAME test_normalizer COMMAND UtestNormalizer)

# cuda specific test functions
if(${USE_CUDA})

	add_executable(UtestProcGpu utest_proc_gpu.cpp)
	target_link_libraries(UtestProcGpu PUBLIC Histeq)
	set_property(TARGET UtestProcGpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	add_test(NAME test_proc_gpu COMMAND UtestProcGpu)

	add_executable(UtestCdfGpu utest_cdf_gpu.cpp)
	target_link_libraries(UtestCdfGpu PUBLIC Histeq)
	set_property(TARGET UtestCdfGpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	add_test(NAME test_cdf_gpu COMMAND UtestCdfGpu)

	add_executable(UtestEqGpu utest_eq_gpu.cpp)
	target_link_libraries(UtestEqGpu PUBLIC Histeq)
	set_property(TARGET UtestEqGpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	add_test(NAME test_eq_gpu COMMAND UtestEqGpu)

	add_executable(UtestGpuFull utest_gpu_full.cpp)
	target_link_libraries(UtestGpuFull PUBLIC Histeq)
	set_property(TARGET UtestGpuFull PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	add_test(NAME test_gpu_full COMMAND UtestGpuFull)

endif()

